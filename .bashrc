echo "[.bashrc] sourced — HOME=${HOME} uid=$(id -u)"
export OMNI_DRONES_DIR="/workspace/omni_drones"
export ISAACSIM_PATH="/workspace/isaacsim"
[ -f "$ISAACSIM_PATH/setup_conda_env.sh" ] && source "$ISAACSIM_PATH/setup_conda_env.sh"

cd $OMNI_DRONES_DIR

# check if venv exists at /opt/venv/bin/activate. If it does activate it. If it does't create a new venv at that location 
if [ -f /opt/venv/bin/activate ]; then
  echo "Activating existing venv"
  source /opt/venv/bin/activate

  # check if isaacsim exists using pip show isaacsim. If it doesn't exist echo insructions to run the install script.
  if ! pip show isaaclab > /dev/null; then
    echo "IsaacLab is not installed. You probably need to run the install script!"
    echo "Execute the bash function omni_drones_install"
  fi
else
  echo "Creating a new venv. Make sure to run the install script!" 
  echo "Execute the bash function omni_drones_install"
  /opt/conda/envs/sim/bin/python -m venv /opt/venv
  source /opt/venv/bin/activate
fi


function omni_drones_install() {
  echo "\n"
  echo "----------------------------------------"
  echo "Installing OmniDrones"
  echo "Expect some red warnings/errors, as long as the install script completes successfully, you're good to go!"
  read -rp "Press Enter to continue..."

  cd $OMNI_DRONES_DIR/docker
  ./install

  # check if pip show isaaclab is succesfful 
  if ! pip show isaaclab > /dev/null; then
    echo "----------------------------------------"
    echo "IsaacLab is not installed! There is an issue with the install script!"   
    return 1
  fi
  echo "OmniDrones installed successfully"
  echo "----------------------------------------"
  echo "Note that it's okay to see errors of the following format:"
  echo "ERROR: pip's dependency resolver does not currently take into account all the packages...."
  echo "Any other errors are not expected"
  echo "----------------------------------------"
  echo "re-source your bash using: source ~/.bashrc"
}

function play_obs_avoidance(){
  # Optional arg: scenario index 0–4 (matches cfg/task/DroneSubjectTrack.yaml lists).
  local idx="${1:-}"
  local loop_time=(25.0 24.0 28.0 30.0 30.0)
  local loop_radius=(5.0 4.9 5.2 4.4 6.0)
  local obstacle_period=(28.0 21.0 15.0 15.0 23.0)
  local scenario_overrides=()
  if [ -n "$idx" ]; then
    if ! [[ "$idx" =~ ^[0-4]$ ]]; then
      echo "play_obs_avoidance: optional index must be a single digit 0–4" >&2
      return 1
    fi
    scenario_overrides=(
      "task.loop_time=[${loop_time[$idx]}]"
      "task.loop_radius=[${loop_radius[$idx]}]"
      "task.obstacle_period=[${obstacle_period[$idx]}]"
    )
  fi
  cd /workspace/omni_drones/scripts
  python play.py task.env.num_envs=1 algo.checkpoint_path=/workspace/omni_drones/checkpoint.pt task=DroneSubjectTrack headless=false "${scenario_overrides[@]}" total_frames=60000
}


function viz_trajectory(){
  python3 -m http.server 8080 --directory /workspace/omni_drones/scripts
}

function train_obs_avoidance(){
  local headless=${1:-true}
  cd /workspace/omni_drones/scripts
  if [ -n "${WANDB_API_KEY:-}" ]; then
    python train.py algo=ppo task=DroneSubjectTrack headless=${headless}
  else
    python train.py algo=ppo wandb.mode=disabled task=DroneSubjectTrack headless=${headless}
  fi
}