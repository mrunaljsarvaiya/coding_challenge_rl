# MIT License
#
# Copyright (c) 2023 Botian Xu, Tsinghua University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import torch
import torch.distributions as D
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import Unbounded, Composite

import omni_drones.utils.kit as kit_utils
from omni_drones.utils.torch import euler_to_quaternion, quat_axis
from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.envs.utils import create_visual_sphere
from omni_drones.robots.drone import MultirotorBase
from pxr import Gf


# ============================================================================
# Reference implementations copied verbatim from fly.py
# ============================================================================

def mod_2_pi(x : np.ndarray) -> np.ndarray:
    """
    Convert an angle in radians into range [-pi, pi]
    """
    return np.mod(x + np.pi, 2 * np.pi) - np.pi

class Scenario:
    """
    Base class for describing the test scenario.
    """

    def __init__(self):
        pass

    def params(self):
        """
        Return static parameters for the scenario
        """
        return dict(
            dt = 0.025,  # [s]
            total_time = 30.0,  # [s]

            drone_initial_position = np.array([-3.0, 0.0]),  # [m]
            drone_max_speed = 10.0,  # [m/s]
            drone_max_accel = 2.5,  # [m/s^2]

            obstacle_radius = 2.0,  # [m]

            desired_range_to_subject_cost = 10.0,
            desired_range_to_subject = 3.0,  # [m]

            desired_angle_to_subject_velocity_cost = 10.0,
            desired_angle_to_subject_velocity = 0.0 * np.pi,  # [rad]

            obstacle_cost = 500.0
        )

    def subject_position(self, t : float) -> np.ndarray:
        """
        Return the subject's (x, y) position given a time.
        """
        return np.array([0.0, 0.0])

    def obstacle_position(self, t : float) -> np.ndarray:
        """
        Return the obstacle's (x, y) position given a time.
        """
        return np.array([10.0, 10.0])

class FigureEightCosineObstacle(Scenario):
    """
    The subject moves along a figure eight curve:
        https://mathworld.wolfram.com/EightCurve.html

    Obstacle moves along a cosine wave.
    """

    def __init__(self,
                 loop_time : float = 25.0,  # [s]
                 loop_radius : float = 5.0,  # [m]
                 obstacle_period = 4.14):
        self.loop_time = loop_time
        self.loop_radius = loop_radius
        self.obstacle_period = obstacle_period

    def subject_position(self, t : float) -> np.ndarray:
        t_angle = t / self.loop_time * (2 * np.pi)
        x = self.loop_radius * np.sin(t_angle)
        y = x * np.cos(t_angle)
        return np.array([x, y]) + np.array([0, 1])

    def obstacle_position(self, t : float) -> np.ndarray:
        return np.array([11, -1]) + (t/25) * np.array([-21.1, 0]) +\
           np.cos(t * 2 * np.pi / self.obstacle_period) * np.array([0, 5])


# ============================================================================
# RL Environment
# ============================================================================

class DroneSubjectTrackEnv(IsaacEnv):
    r"""
    Subject-tracking and obstacle-avoidance RL environment.

    A 3D quadrotor must track a moving subject at a desired range and angle
    while avoiding a moving obstacle. The 2D tracking problem from the Skydio
    Figure-Eight challenge (fly.py) is embedded in the XY plane at a
    configurable fixed height (``fixed_z``).

    ## Observation (19D)

    - ``desired_position`` (3): target position minus drone position (env frame),
      i.e. vector from drone to desired point; target comes from the subject's
      state (range + angle from velocity vector).
    - ``drone_velocity`` (3): current linear velocity in env frame.
    - ``subject_velocity`` (3): subject linear velocity in env frame.
    - ``obstacle_position`` (3): obstacle position minus drone position (env frame).
    - ``obstacle_velocity`` (3): current obstacle velocity.
    - ``drone_quaternion`` (4): current orientation quaternion.

    ## Action (4D)

    Body rates (3) + thrust (1), processed by ``RateController``.

    ## Reward

    ``total_reward = base_reward + reward_reg``

    *base_reward* (per-step, consistent with ``compute_error_for_scenario``
    from fly.py):

    - Range error:    ``-0.5 * rl_dt * range_cost * (desired_range - actual_range)^2``
    - Angle error:    ``-0.5 * rl_dt * angle_cost * mod_2pi(residual)^2``
    - Obstacle error: ``-0.5 * rl_dt * obstacle_cost * max(0, radius - dist)^2``

    *reward_reg*:

    - Action magnitude penalty: ``-action_weight * ||action||^2``
    - Uprightness reward:       ``uprightness_weight * up_reward``
    - Z-height penalty:        ``-reward_z_height_weight * |z - fixed_z|``
    - Alive bonus:              ``reward_alive`` (small positive each step)

    On the step where the episode ends by horizon (full ``max_episode_length`` steps)
    without altitude failure, add ``reward_success`` once (full-trajectory bonus).

    Optional one-step penalties when an episode closes (config magnitudes, subtracted
    from reward on that step): ``reward_termination_penalty`` (altitude failure /
    ``terminated``), ``reward_truncation_penalty`` (time limit without failure /
    ``truncated`` only).

    ## Episode End

    Terminates when the drone altitude drops below 0.3 m, exceeds 10 m,
    or when ``max_episode_length`` is reached.

    ## Stats (episode-end indicators, 0/1 per completed episode)

    Logged means are the fraction of finished episodes in the batch: ``crashed_low``,
    ``crashed_high`` (altitude failure), and ``horizon_truncated`` (time limit without
    crash). Crash outcomes take precedence if both time limit and failure apply on the
    same step.
    """

    REQUIRED_CONFIG_PARAMS = [
        "loop_time",
        "loop_radius",
        "obstacle_period",
        "fixed_z",
        "desired_range_to_subject",
        "desired_angle_to_subject_velocity",
        "desired_range_to_subject_cost",
        "desired_angle_to_subject_velocity_cost",
        "obstacle_cost",
        "obstacle_radius",
        "reward_action_magnitude_weight",
        "reward_uprightness_weight",
        "reward_z_height_weight",
        "reward_termination_penalty",
        "reward_truncation_penalty",
        "reward_alive",
        "reward_success",
    ]

    def __init__(self, cfg, headless):
        missing = [p for p in self.REQUIRED_CONFIG_PARAMS if cfg.task.get(p, None) is None]
        if missing:
            raise ValueError(
                f"Missing required config parameters in task config: {missing}"
            )

        self._headless = headless

        self.loop_time_values = torch.tensor(
            list(cfg.task.loop_time), dtype=torch.float32
        )
        self.loop_radius_values = torch.tensor(
            list(cfg.task.loop_radius), dtype=torch.float32
        )
        self.obstacle_period_values = torch.tensor(
            list(cfg.task.obstacle_period), dtype=torch.float32
        )

        if self.loop_time_values.numel() == 0:
            raise ValueError("cfg.task.loop_time must be a non-empty list.")
        if self.loop_radius_values.numel() == 0:
            raise ValueError("cfg.task.loop_radius must be a non-empty list.")
        if self.obstacle_period_values.numel() == 0:
            raise ValueError("cfg.task.obstacle_period must be a non-empty list.")

        if not (
            self.loop_time_values.numel()
            == self.loop_radius_values.numel()
            == self.obstacle_period_values.numel()
        ):
            raise ValueError(
                "loop_time, loop_radius, and obstacle_period must all have the same length. "
                f"Got lengths: loop_time={self.loop_time_values.numel()}, "
                f"loop_radius={self.loop_radius_values.numel()}, "
                f"obstacle_period={self.obstacle_period_values.numel()}"
            )

        self.num_scenario_options = int(self.loop_time_values.numel())

        if not headless:
            print(
                "[DroneSubjectTrackEnv] scenario parameters: "
                f"loop_time={self.loop_time_values.tolist()}, "
                f"loop_radius={self.loop_radius_values.tolist()}, "
                f"obstacle_period={self.obstacle_period_values.tolist()}"
            )

        self.fixed_z = float(cfg.task.fixed_z)

        self.desired_range_to_subject = float(cfg.task.desired_range_to_subject)
        self.desired_angle_to_subject_velocity = float(
            cfg.task.desired_angle_to_subject_velocity
        )
        self.desired_range_to_subject_cost = float(
            cfg.task.desired_range_to_subject_cost
        )
        self.desired_angle_to_subject_velocity_cost = float(
            cfg.task.desired_angle_to_subject_velocity_cost
        )
        self.obstacle_cost_weight = float(cfg.task.obstacle_cost)
        self.obstacle_radius = float(cfg.task.obstacle_radius)

        self.reward_action_magnitude_weight = float(
            cfg.task.reward_action_magnitude_weight
        )
        self.reward_uprightness_weight = float(cfg.task.reward_uprightness_weight)
        self.reward_z_height_weight = float(cfg.task.reward_z_height_weight)
        self.reward_termination_penalty = float(cfg.task.reward_termination_penalty)
        self.reward_truncation_penalty = float(cfg.task.reward_truncation_penalty)
        self.reward_alive = float(cfg.task.reward_alive)
        self.reward_success = float(cfg.task.reward_success)

        super().__init__(cfg, headless)

        self.loop_time_values = self.loop_time_values.to(self.device)
        self.loop_radius_values = self.loop_radius_values.to(self.device)
        self.obstacle_period_values = self.obstacle_period_values.to(self.device)

        self.env_loop_time = torch.empty(self.num_envs, device=self.device)
        self.env_loop_radius = torch.empty(self.num_envs, device=self.device)
        self.env_obstacle_period = torch.empty(self.num_envs, device=self.device)
        self.env_scenario_idx = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

        default_idx = 0
        self.env_scenario_idx[:] = default_idx
        self.env_loop_time[:] = self.loop_time_values[default_idx]
        self.env_loop_radius[:] = self.loop_radius_values[default_idx]
        self.env_obstacle_period[:] = self.obstacle_period_values[default_idx]

        self.drone.initialize(track_contact_forces=False)

        sim_dt = float(self.cfg.sim.dt)
        substeps = int(self.cfg.sim.get("substeps", 1))
        self.rl_dt = sim_dt * substeps

        self.init_vels = torch.zeros_like(self.drone.get_velocities())
        self.init_joint_pos = self.drone.get_joint_positions(True)
        self.init_joint_vels = torch.zeros_like(self.drone.get_joint_velocities())

        self.init_rpy_dist = D.Uniform(
            torch.tensor([-.1, -.1, 0.], device=self.device) * torch.pi,
            torch.tensor([.1, .1, 0.], device=self.device) * torch.pi,
        )

        self.effort = torch.zeros(
            self.num_envs, 1, self.drone.action_spec.shape[-1],
            device=self.device,
        )

        self.alpha = 0.8

    # ----------------------------------------------------------------
    # Torch-vectorized trajectory methods (match numpy originals)
    # ----------------------------------------------------------------

    def _compute_subject_position(
        self,
        t: torch.Tensor,
        loop_time: torch.Tensor,
        loop_radius: torch.Tensor,
    ) -> torch.Tensor:
        """Subject position at time *t*.

        Mirrors ``FigureEightCosineObstacle.subject_position`` exactly.

        Args:
            t: (N,) time values.
            loop_time: (N,) per-env loop times.
            loop_radius: (N,) per-env loop radii.
        Returns:
            (N, 3) positions with z = ``fixed_z``.
        """
        t_angle = t / loop_time * (2 * torch.pi)
        x = loop_radius * torch.sin(t_angle)
        y = x * torch.cos(t_angle) + 1.0
        z = torch.full_like(x, self.fixed_z)
        return torch.stack([x, y, z], dim=-1)

    def _compute_obstacle_position(
        self,
        t: torch.Tensor,
        obstacle_period: torch.Tensor,
    ) -> torch.Tensor:
        """Obstacle position at time *t*.

        Mirrors ``FigureEightCosineObstacle.obstacle_position`` exactly.

        Args:
            t: (N,) time values.
            obstacle_period: (N,) per-env obstacle periods.
        Returns:
            (N, 3) positions with z = ``fixed_z``.
        """
        x = 11.0 + (t / 25.0) * (-21.1)
        y = -1.0 + torch.cos(t * 2 * torch.pi / obstacle_period) * 5.0
        z = torch.full_like(x, self.fixed_z)
        return torch.stack([x, y, z], dim=-1)

    def _compute_velocity_finite_diff(
        self,
        pos_func,
        t: torch.Tensor,
        delta_t: float = 0.001,
        **kwargs,
    ) -> torch.Tensor:
        """Velocity via forward finite difference.

        Args:
            pos_func: callable  (N,) -> (N, 3).
            t: (N,) time values.
            delta_t: finite-difference step size.
            **kwargs: additional per-env arguments forwarded to pos_func.
        Returns:
            (N, 3) velocity estimate.
        """
        return (pos_func(t + delta_t, **kwargs) - pos_func(t, **kwargs)) / delta_t

    def _compute_desired_position(
        self,
        subject_pos: torch.Tensor,
        subject_vel: torch.Tensor,
    ) -> torch.Tensor:
        """Desired drone position from the subject's state.

        Implements the math from fly.py lines 490-494::

            theta = arctan2(subject_velocity[1], subject_velocity[0])
            phi = desired_angle_to_subject_velocity
            unit_vec = [cos(theta + phi), sin(theta + phi)]
            desired_pos = subject_pos + desired_range * unit_vec

        Args:
            subject_pos: (N, 3) subject positions.
            subject_vel: (N, 3) subject velocities.
        Returns:
            (N, 3) desired drone positions (z = ``fixed_z``).
        """
        theta = torch.atan2(subject_vel[:, 1], subject_vel[:, 0])
        phi = self.desired_angle_to_subject_velocity
        cos_tp = torch.cos(theta + phi)
        sin_tp = torch.sin(theta + phi)

        desired_xy = subject_pos[:, :2] + self.desired_range_to_subject * torch.stack(
            [cos_tp, sin_tp], dim=-1
        )
        desired_z = torch.full(
            (subject_pos.shape[0], 1), self.fixed_z, device=subject_pos.device,
        )
        return torch.cat([desired_xy, desired_z], dim=-1)

    # ----------------------------------------------------------------
    # IsaacEnv interface
    # ----------------------------------------------------------------

    def _design_scene(self):
        drone_model_cfg = self.cfg.task.drone_model
        self.drone, self.controller = MultirotorBase.make(
            drone_model_cfg.name, drone_model_cfg.controller,
        )
        if self.controller is not None:
            self.controller = self.controller.to(self.device)

        kit_utils.create_ground_plane(
            "/World/defaultGroundPlane",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )

        t0 = torch.tensor([0.0], device=self.device)
        loop_time_0 = self.loop_time_values[:1].to(self.device)
        loop_radius_0 = self.loop_radius_values[:1].to(self.device)
        obstacle_period_0 = self.obstacle_period_values[:1].to(self.device)

        self.drone.spawn(translations=[
            (-3.0, 0.0, self.fixed_z),
        ])

        if not self._headless:
            self._setup_debug_visuals(t0, loop_time_0, loop_radius_0, obstacle_period_0)

        return ["/World/defaultGroundPlane"]

    def _setup_debug_visuals(
        self,
        t0: torch.Tensor,
        loop_time: torch.Tensor,
        loop_radius: torch.Tensor,
        obstacle_period: torch.Tensor,
    ):
        """Spawn collision-free debug spheres: subject (blue), desired (green), obstacle (red)."""
        subject_pos_0 = self._compute_subject_position(
            t0, loop_time, loop_radius
        )
        subject_vel_0 = self._compute_velocity_finite_diff(
            self._compute_subject_position, t0,
            loop_time=loop_time,
            loop_radius=loop_radius,
        )
        desired_pos_0 = self._compute_desired_position(
            subject_pos_0, subject_vel_0,
        ).squeeze(0)
        subject_pos_0 = subject_pos_0.squeeze(0)
        obstacle_pos_0 = self._compute_obstacle_position(
            t0, obstacle_period
        ).squeeze(0)

        self._debug_subject_prim = create_visual_sphere(
            prim_path="/World/debug_subject",
            radius=0.3,
            translation=(
                subject_pos_0[0].item(),
                subject_pos_0[1].item(),
                subject_pos_0[2].item(),
            ),
            color=(0.2, 0.45, 1.0),  # blue
        )
        self._debug_desired_prim = create_visual_sphere(
            prim_path="/World/debug_desired",
            radius=0.3,
            translation=(
                desired_pos_0[0].item(),
                desired_pos_0[1].item(),
                desired_pos_0[2].item(),
            ),
            color=(0.0, 0.85, 0.25),  # green
            opacity=0.4,
        )
        self._debug_obstacle_prim = create_visual_sphere(
            prim_path="/World/debug_obstacle",
            radius=self.obstacle_radius,
            translation=(
                obstacle_pos_0[0].item(),
                obstacle_pos_0[1].item(),
                obstacle_pos_0[2].item(),
            ),
            color=(1.0, 0.1, 0.1),  # red
        )

    def _update_debug_visuals(self):
        """Move the debug spheres to track the current env-0 positions."""
        if self._headless or not hasattr(self, "_debug_subject_prim"):
            return

        env_offset = self.envs_positions[0]  # (3,)

        subject_world = self._cached_subject_pos[0] + env_offset
        desired_world = self._cached_desired_pos[0] + env_offset
        obstacle_world = self._cached_obstacle_pos[0] + env_offset

        self._debug_subject_prim.GetAttribute("xformOp:translate").Set(
            Gf.Vec3d(subject_world[0].item(), subject_world[1].item(), subject_world[2].item())
        )
        self._debug_desired_prim.GetAttribute("xformOp:translate").Set(
            Gf.Vec3d(desired_world[0].item(), desired_world[1].item(), desired_world[2].item())
        )
        self._debug_obstacle_prim.GetAttribute("xformOp:translate").Set(
            Gf.Vec3d(obstacle_world[0].item(), obstacle_world[1].item(), obstacle_world[2].item())
        )

    def enable_play_trajectory_recording(self, max_frames: int = 10_000):
        """Start recording env-0 world-frame XY paths for a top-down Plotly animation (see ``save_play_trajectory_gif``)."""
        self._record_play_trajectory = True
        self._play_traj_max_frames = int(max_frames)
        self._play_traj_cap_warned = False
        self._play_traj_drone = []
        self._play_traj_subject = []
        self._play_traj_desired = []
        self._play_traj_obstacle = []

    def _record_play_trajectory_step(
        self,
        drone_pos: torch.Tensor,
        subject_pos: torch.Tensor,
        desired_pos: torch.Tensor,
        obstacle_pos: torch.Tensor,
    ):
        if not getattr(self, "_record_play_trajectory", False):
            return
        n = len(self._play_traj_drone)
        if n >= self._play_traj_max_frames:
            if not self._play_traj_cap_warned:
                print(
                    f"[DroneSubjectTrackEnv] Play trajectory recording cap reached "
                    f"({self._play_traj_max_frames} frames); further steps not stored."
                )
                self._play_traj_cap_warned = True
            return
        off = self.envs_positions[0]
        d = drone_pos[0, 0] + off
        s = subject_pos[0] + off
        des = desired_pos[0] + off
        obs = obstacle_pos[0] + off
        self._play_traj_drone.append((float(d[0].item()), float(d[1].item())))
        self._play_traj_subject.append((float(s[0].item()), float(s[1].item())))
        self._play_traj_desired.append((float(des[0].item()), float(des[1].item())))
        self._play_traj_obstacle.append((float(obs[0].item()), float(obs[1].item())))

    def save_play_trajectory_gif(self, path: str, dot_radius: float = 0.08, frame_stride: int = 10) -> bool:
        """Write a top-down XY interactive Plotly animation from recorded play trajectory."""
        from omni_drones.envs.drone_race.helper import plot_trajectory_html

        traj_d = getattr(self, "_play_traj_drone", [])
        if not traj_d:
            print("[DroneSubjectTrackEnv] No play trajectory recorded; skipping plot.")
            return False

        html_path = plot_trajectory_html(
            traj_d=traj_d,
            traj_s=self._play_traj_subject,
            traj_des=self._play_traj_desired,
            traj_o=self._play_traj_obstacle,
            r_obs=float(self.obstacle_radius),
            rl_dt=float(self.rl_dt),
            path=path,
            frame_stride=frame_stride,
        )
        n = len(traj_d)
        print(f"[DroneSubjectTrackEnv] Saved play trajectory HTML ({n} frames) to {html_path}")
        return True

    def _set_specs(self):
        observation_dim = 3 + 3 + 3 + 3 + 3 + 4  # 19

        self.observation_spec = Composite({
            "agents": {
                "observation": Unbounded((1, observation_dim), device=self.device),
                "intrinsics": self.drone.intrinsics_spec.unsqueeze(0).to(self.device),
            },
        }).expand(self.num_envs).to(self.device)

        self.action_spec = Composite({
            "agents": {
                "action": self.drone.action_spec.unsqueeze(0),
            },
        }).expand(self.num_envs).to(self.device)

        self.reward_spec = Composite({
            "agents": {
                "reward": Unbounded((1, 1)),
            },
        }).expand(self.num_envs).to(self.device)

        self.agent_spec["drone"] = AgentSpec(
            "drone", 1,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
            state_key=("agents", "intrinsics"),
        )

        stats_spec = Composite({
            "return": Unbounded(1),
            "episode_len": Unbounded(1),
            "drone_uprightness": Unbounded(1),
            "base_reward": Unbounded(1),
            "reward_reg": Unbounded(1),
            "reward_alive": Unbounded(1),
            "reward_success": Unbounded(1),
            "range_error": Unbounded(1),
            "angle_error": Unbounded(1),
            "obstacle_error": Unbounded(1),
            "reward_collision_minor": Unbounded(1),
            "crashed_low": Unbounded(1),
            "crashed_high": Unbounded(1),
            "horizon_truncated": Unbounded(1),
            "z_error": Unbounded(1),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.stats = stats_spec.zero()

    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids)
        self.effort[env_ids] = 0.0

        sampled_option_idx = torch.randint(
            low=0,
            high=self.num_scenario_options,
            size=(len(env_ids),),
            device=self.device,
        )

        self.env_scenario_idx[env_ids] = sampled_option_idx
        self.env_loop_time[env_ids] = self.loop_time_values[sampled_option_idx]
        self.env_loop_radius[env_ids] = self.loop_radius_values[sampled_option_idx]
        self.env_obstacle_period[env_ids] = self.obstacle_period_values[sampled_option_idx]

        start_pos = torch.tensor(
            [-3.0, 0.0, self.fixed_z], device=self.device
        ).expand(len(env_ids), -1)

        drone_rpy = self.init_rpy_dist.sample((*env_ids.shape, 1))
        drone_rot = euler_to_quaternion(drone_rpy)

        drone_start_pos_with_agent = start_pos.unsqueeze(1)
        env_positions_with_agent = self.envs_positions[env_ids].unsqueeze(1)

        self.drone.set_world_poses(
            drone_start_pos_with_agent + env_positions_with_agent,
            drone_rot, env_ids,
        )
        self.drone.set_velocities(
            torch.zeros(len(env_ids), 1, 6, device=self.device), env_ids,
        )
        self.drone.set_joint_positions(
            torch.zeros(len(env_ids), 1, 4, device=self.device), env_ids,
        )
        self.drone.set_joint_velocities(
            torch.zeros(len(env_ids), 1, 4, device=self.device), env_ids,
        )

        self.stats[env_ids] = 0.0

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")].clone()
        if self.controller is not None:
            root_state = self.drone.get_state()[..., :13]
            raw_actions = self.controller.scaled_to_raw(actions)
            rotor_cmds = self.controller(root_state, *raw_actions)
            self.drone.apply_action(rotor_cmds)
        else:
            raise RuntimeError(
                "No controller found. A RateController is required for this "
                "environment."
            )

    def _post_sim_step(self, tensordict: TensorDictBase):
        self.effort = tensordict[("agents", "action")].clone()

    def _build_robot_state(self) -> torch.Tensor:
        """Refresh drone kinematic caches and return a state tensor.

        Returns:
            (N, 1, 15): ``[linear_vel(3) | rot_matrix_flat(9) | angular_vel(3)]``
        """
        self.drone.get_state()
        lin_vel = self.drone.get_linear_velocity()    # (N, 1, 3)
        rot_mat = self.drone.get_rotation_matrix()    # (N, 1, 9)
        ang_vel = self.drone.get_angular_velocity()   # (N, 1, 3)
        return torch.cat([lin_vel, rot_mat, ang_vel], dim=-1)

    def _compute_state_and_obs(self):
        self.drone_state = self._build_robot_state()

        drone_vel = self.drone.get_linear_velocity()  # (N, 1, 3)
        drone_rot = self.drone.rot                    # (N, 1, 4)

        t = self.progress_buf.float() * self.rl_dt    # (N,)

        loop_time = self.env_loop_time
        loop_radius = self.env_loop_radius
        obstacle_period = self.env_obstacle_period

        subject_pos = self._compute_subject_position(t, loop_time, loop_radius)
        subject_vel = self._compute_velocity_finite_diff(
            self._compute_subject_position,
            t,
            loop_time=loop_time,
            loop_radius=loop_radius,
        )
        obstacle_pos = self._compute_obstacle_position(t, obstacle_period)
        obstacle_vel = self._compute_velocity_finite_diff(
            self._compute_obstacle_position,
            t,
            obstacle_period=obstacle_period,
        )
        desired_pos = self._compute_desired_position(subject_pos, subject_vel)

        self._cached_drone_vel = drone_vel
        self._cached_subject_pos = subject_pos
        self._cached_subject_vel = subject_vel
        self._cached_obstacle_pos = obstacle_pos
        self._cached_desired_pos = desired_pos

        self._update_debug_visuals()

        drone_pos = self.drone.pos  # (N, 1, 3), env frame — same as desired/obstacle
        self._record_play_trajectory_step(
            drone_pos, subject_pos, desired_pos, obstacle_pos,
        )

        desired_pos_rel = desired_pos.unsqueeze(1) - drone_pos
        obstacle_pos_rel = obstacle_pos.unsqueeze(1) - drone_pos
        subject_vel_rel = subject_vel.unsqueeze(1) - drone_vel
        obstacle_vel_rel = obstacle_vel.unsqueeze(1) - drone_vel

        self._cached_desired_pos_rel = desired_pos_rel
        self._cached_obstacle_pos_rel = obstacle_pos_rel
        self._cached_subject_vel_rel = subject_vel_rel
        self._cached_obstacle_vel_rel = obstacle_vel_rel

        obs = torch.cat([
            desired_pos_rel,       # (N, 1, 3)
            drone_vel,             # (N, 1, 3)
            subject_vel_rel,       # (N, 1, 3)
            obstacle_pos_rel,      # (N, 1, 3)
            obstacle_vel_rel,      # (N, 1, 3)
            drone_rot,             # (N, 1, 4)
        ], dim=-1)  # (N, 1, 19)

        return TensorDict(
            {
                "agents": {
                    "observation": obs,
                    "intrinsics": self.drone.intrinsics,
                },
                "stats": self.stats.clone(),
            },
            self.batch_size,
        )

    # ----------------------------------------------------------------
    # Reward helpers
    # ----------------------------------------------------------------

    def _get_uprightness_reward(self, drone_rot):
        """Reward for keeping the drone's z-axis aligned with world up.

        Args:
            drone_rot: (N, 4) drone quaternions.

        Returns:
            reward_up: (N,)
            drone_up:  (N, 3) — drone's z-axis in world frame (reused for stats).
        """
        drone_up = quat_axis(drone_rot, axis=2)
        reward = 0.5 * torch.square((drone_up[..., 2] + 1) / 2)
        return reward, drone_up

    @staticmethod
    def _mod_2_pi_torch(x: torch.Tensor) -> torch.Tensor:
        """Convert angles to [-pi, pi]. Matches ``mod_2_pi`` from fly.py."""
        return torch.remainder(x + torch.pi, 2 * torch.pi) - torch.pi

    # ----------------------------------------------------------------
    # Reward & done
    # ----------------------------------------------------------------

    def _compute_reward_and_done(self):
        drone_pos = self.drone.pos   # (N, 1, 3), env frame
        drone_rot = self.drone.rot   # (N, 1, 4)

        drone_pos_flat = drone_pos.squeeze(1)   # (N, 3)
        drone_rot_flat = drone_rot.squeeze(1)   # (N, 4)

        subject_pos = self._cached_subject_pos   # (N, 3)
        subject_vel = self._cached_subject_vel   # (N, 3)
        obstacle_pos = self._cached_obstacle_pos # (N, 3)

        desired_pos_rel = self._cached_desired_pos_rel
        obstacle_pos_rel = self._cached_obstacle_pos_rel
        subject_vel_rel = self._cached_subject_vel_rel
        obstacle_vel_rel = self._cached_obstacle_vel_rel

        # ==================================================================
        # base_reward  (per-step error matching fly.py, XY only)
        # ==================================================================

        diff_xy = drone_pos_flat[:, :2] - subject_pos[:, :2]           # (N, 2)
        actual_range = torch.norm(diff_xy, dim=-1)                     # (N,)
        range_residual = self.desired_range_to_subject - actual_range
        range_error = (
            0.5 * self.rl_dt
            * self.desired_range_to_subject_cost
            * torch.square(range_residual)
        )

        actual_angle = torch.atan2(diff_xy[:, 1], diff_xy[:, 0])
        subject_vel_angle = torch.atan2(
            subject_vel[:, 1], subject_vel[:, 0] + 1e-6,
        )
        angle_residual = self._mod_2_pi_torch(
            self.desired_angle_to_subject_velocity
            + subject_vel_angle
            - actual_angle
        )
        angle_error = (
            0.5 * self.rl_dt
            * self.desired_angle_to_subject_velocity_cost
            * torch.square(angle_residual)
        )

        range_to_obstacle_xy = torch.norm(
            drone_pos_flat[:, :2] - obstacle_pos[:, :2], dim=-1,
        )
        infringement = torch.clamp(
            self.obstacle_radius - range_to_obstacle_xy, min=0.0,
        )
        obstacle_error = (
            0.5 * self.rl_dt
            * self.obstacle_cost_weight
            * torch.square(infringement)
        )

        base_reward = -(range_error + angle_error + obstacle_error)

        # ==================================================================
        # Termination (needed before success bonus)
        # ==================================================================

        hasnan = torch.isnan(self.drone_state).any(-1)
        assert not hasnan.any(), "NaN detected in drone_state"

        crashed_low = drone_pos_flat[:, 2] < 0.3
        crashed_high = drone_pos_flat[:, 2] > 10.0
        terminated = crashed_low | crashed_high

        # ==================================================================
        # reward_reg
        # ==================================================================

        z_error = torch.abs(drone_pos_flat[:, 2] - self.fixed_z)
        z_error_penalty = -self.reward_z_height_weight * z_error

        # TODO Ideally the reward term penalizes (thrust_cmd - hover_thrust).
        action_sq = torch.sum(
            torch.square(self.effort.squeeze(1)), dim=-1,
        )  # (N,)
        action_penalty = -self.reward_action_magnitude_weight * action_sq

        reward_up, drone_up = self._get_uprightness_reward(drone_rot_flat)
        uprightness_reward = self.reward_uprightness_weight * reward_up

        alive_bonus = torch.full_like(base_reward, self.reward_alive)
        reward_reg = action_penalty + uprightness_reward + alive_bonus + z_error_penalty

        horizon_done = self.progress_buf >= self.max_episode_length
        episode_success = horizon_done & ~terminated
        success_bonus = self.reward_success * episode_success.float()

        truncated_episode = horizon_done & ~terminated
        episode_end_penalty = (
            -self.reward_termination_penalty * terminated.float()
            - self.reward_truncation_penalty * truncated_episode.float()
        )

        # Collision-minor penalty — vectorised equivalent of fly.py obs_avoidance_filter_step.
        # All quantities in the XY plane to match fly.py's 2-D planner.
        #
        # fly.py:  rel_pos = drone_pos - obs_pos  (points away from obstacle)
        #          rel_vel = drone_vel - obs_vel
        #          relative_vel_dot = dot(rel_pos, rel_vel) / |rel_pos|
        # Here:    obstacle_pos_rel = obs_pos - drone_pos  =>  rel_pos = -obstacle_pos_rel
        #          obstacle_vel_rel = obs_vel - drone_vel  =>  rel_vel = -obstacle_vel_rel
        # So dot(rel_pos, rel_vel) = dot(-obs_pos_rel, -obs_vel_rel)
        #                          = dot(obs_pos_rel, obs_vel_rel)   (double negation)
        obs_pos_rel_xy = obstacle_pos_rel.squeeze(1)[:, :2]   # (N, 2)
        obs_vel_rel_xy = obstacle_vel_rel.squeeze(1)[:, :2]   # (N, 2)
        relative_vel_dot = (
            torch.sum(obs_pos_rel_xy * obs_vel_rel_xy, dim=-1)
            / (range_to_obstacle_xy + 1e-6)
        )  # (N,)

        # fly.py: safety_margin = radius + max(2.0, 0.5 * speed)
        # margin grows with speed; minimum headroom is always 2.0 m
        drone_speed_xy = torch.norm(
            self._cached_drone_vel.squeeze(1)[:, :2], dim=-1
        )  # (N,)
        safety_margin_minor = (
            self.obstacle_radius + torch.clamp(0.5 * drone_speed_xy, min=2.0)
        )  # (N,)

        in_danger = (
            (range_to_obstacle_xy < safety_margin_minor)
            & (range_to_obstacle_xy > self.obstacle_radius)
            & (relative_vel_dot < 0)
        )
        magnitude = 20.0 * torch.exp(-1.5 * range_to_obstacle_xy) / (
            range_to_obstacle_xy + 1e-6
        )
        reward_collision_minor = -torch.where(
            in_danger, magnitude, torch.zeros_like(magnitude)
        )  # (N,) — negative penalty when drone is close and approaching the obstacle

        reward = base_reward + reward_reg + success_bonus + episode_end_penalty  # (N,)

        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
        done = truncated | terminated.unsqueeze(-1)

        # ==================================================================
        # Stats
        # ==================================================================

        self.stats["return"].add_(reward.unsqueeze(-1))
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)
        self.stats["drone_uprightness"].mul_(self.alpha).add_(
            (1 - self.alpha) * drone_up[..., 2].unsqueeze(-1)
        )
        self.stats["base_reward"].add_(base_reward.unsqueeze(-1))
        self.stats["reward_reg"].add_(reward_reg.unsqueeze(-1))
        self.stats["reward_alive"].add_(alive_bonus.unsqueeze(-1))
        self.stats["reward_success"].add_(success_bonus.unsqueeze(-1))
        self.stats["range_error"].add_(range_error.unsqueeze(-1))
        self.stats["angle_error"].add_(angle_error.unsqueeze(-1))
        self.stats["obstacle_error"].add_(obstacle_error.unsqueeze(-1))
        self.stats["reward_collision_minor"].add_(reward_collision_minor.unsqueeze(-1))
        self.stats["z_error"].add_(z_error.unsqueeze(-1))

        # Episode-end diagnostics (0/1 on the terminal step; mean over batch = fraction)
        d = done.squeeze(-1)
        self.stats["crashed_low"][d] = (terminated & crashed_low).float()[d].unsqueeze(-1)
        self.stats["crashed_high"][d] = (
            (terminated & crashed_high & ~crashed_low).float()[d].unsqueeze(-1)
        )
        self.stats["horizon_truncated"][d] = (
            (horizon_done & ~terminated).float()[d].unsqueeze(-1)
        )

        return TensorDict(
            {
                "agents": {
                    "reward": reward.unsqueeze(-1).unsqueeze(-1),
                },
                "done": done,
                "terminated": terminated.unsqueeze(-1),
                "truncated": truncated,
                "stats": self.stats.clone(),
            },
            self.batch_size,
        )