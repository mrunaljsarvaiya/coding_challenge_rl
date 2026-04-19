import numpy as np
import plotly.graph_objects as go


def plot_trajectory_html(
    traj_d: list,
    traj_s: list,
    traj_des: list,
    traj_o: list,
    r_obs: float,
    rl_dt: float,
    path: str,
    frame_stride: int = 10,
) -> str:
    """
    Build and save an interactive Plotly animation of the drone obstacle-avoidance trajectory.

    Each animation frame directly indexes traj_d[i], traj_s[i], traj_des[i], traj_o[i],
    mirroring the matplotlib _draw_frame(i) approach.  The obstacle is drawn as a circle
    shape in data coordinates so its radius is correct regardless of zoom level.

    Args:
        traj_d:       List of (x, y) drone positions.
        traj_s:       List of (x, y) subject positions.
        traj_des:     List of (x, y) desired positions.
        traj_o:       List of (x, y) obstacle centre positions.
        r_obs:        Obstacle radius in metres.
        rl_dt:        Simulation timestep in seconds.
        path:         Output path (extension replaced with .html).
        frame_stride: Use every Nth recorded step as an animation frame.

    Returns:
        Absolute path of the written HTML file.
    """
    n = len(traj_d)

    drone_arr    = np.array(traj_d)
    subject_arr  = np.array(traj_s)
    desired_arr  = np.array(traj_des)
    obstacle_arr = np.array(traj_o)

    step = max(1, int(frame_stride))
    frame_indices = list(range(0, n, step))
    if frame_indices[-1] != n - 1:
        frame_indices.append(n - 1)

    all_x = (drone_arr[:, 0].tolist() + subject_arr[:, 0].tolist() +
             desired_arr[:, 0].tolist() + obstacle_arr[:, 0].tolist())
    all_y = (drone_arr[:, 1].tolist() + subject_arr[:, 1].tolist() +
             desired_arr[:, 1].tolist() + obstacle_arr[:, 1].tolist())
    margin  = r_obs + 1.0
    x_range = [min(all_x) - margin, max(all_x) + margin]
    y_range = [min(all_y) - margin, max(all_y) + margin]

    def _dot(x, y, name, color, size=10):
        return go.Scatter(
            x=[x], y=[y], mode='markers', name=name,
            marker=dict(color=color, size=size),
            showlegend=True,
        )

    def _obs_circle(cx, cy):
        """Circle in data coordinates — equivalent to matplotlib Circle(center, r_obs)."""
        return dict(
            type='circle', xref='x', yref='y',
            x0=cx - r_obs, y0=cy - r_obs,
            x1=cx + r_obs, y1=cy + r_obs,
            line=dict(color='#FF5555', width=1.5),
            fillcolor='rgba(255,85,85,0.15)',
        )

    frames = [
        go.Frame(
            data=[
                _dot(traj_d[i][0],   traj_d[i][1],   'drone',   '#5555FF'),
                _dot(traj_s[i][0],   traj_s[i][1],   'subject', '#CCCC00'),
                _dot(traj_des[i][0], traj_des[i][1], 'desired', '#55FF55'),
            ],
            layout=go.Layout(shapes=[_obs_circle(traj_o[i][0], traj_o[i][1])]),
            name=f'{i * rl_dt:.3f}',
        )
        for i in frame_indices
    ]

    i0 = frame_indices[0]
    fig = go.Figure(
        data=[
            # Animated dot traces (0-2) — replaced by each frame
            _dot(traj_d[i0][0],   traj_d[i0][1],   'drone',   '#5555FF'),
            _dot(traj_s[i0][0],   traj_s[i0][1],   'subject', '#CCCC00'),
            _dot(traj_des[i0][0], traj_des[i0][1], 'desired', '#55FF55'),
            # Full path lines — static, always visible
            go.Scatter(x=drone_arr[:, 0],    y=drone_arr[:, 1],    mode='lines', name='drone path',    line=dict(color='#AAAAFF', width=1), showlegend=False),
            go.Scatter(x=subject_arr[:, 0],  y=subject_arr[:, 1],  mode='lines', name='subject path',  line=dict(color='#FFFF55', width=1), showlegend=False),
            go.Scatter(x=desired_arr[:, 0],  y=desired_arr[:, 1],  mode='lines', name='desired path',  line=dict(color='#AAFFAA', width=1), showlegend=False),
            go.Scatter(x=obstacle_arr[:, 0], y=obstacle_arr[:, 1], mode='lines', name='obstacle path', line=dict(color='#FFAAAA', width=1), showlegend=False),
        ],
        frames=frames,
    )

    frame_ms = int(step * rl_dt * 1000)
    fig.update_layout(
        title='Drone Obstacle Avoidance Trajectory',
        template='plotly_dark',
        height=700,
        shapes=[_obs_circle(traj_o[i0][0], traj_o[i0][1])],
        xaxis=dict(range=x_range, dtick=1.0, showline=False),
        yaxis=dict(range=y_range, dtick=1.0, showline=False, scaleanchor='x', scaleratio=1),
        updatemenus=[dict(
            type='buttons', showactive=False,
            buttons=[
                dict(label='Play',  method='animate',
                     args=[None, dict(frame=dict(duration=frame_ms, redraw=True), fromcurrent=True, mode='immediate')]),
                dict(label='Pause', method='animate',
                     args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate', transition=dict(duration=0))]),
            ],
        )],
        sliders=[dict(
            steps=[dict(method='animate',
                        args=[[f.name], dict(mode='immediate', frame=dict(duration=frame_ms, redraw=True), transition=dict(duration=0))],
                        label=f.name)
                   for f in frames],
            active=0, x=0.1, y=0, xanchor='left', yanchor='top', len=0.9,
        )],
    )

    html_path = path.rsplit('.', 1)[0] + '.html' if '.' in path else path + '.html'
    fig.write_html(html_path)
    return html_path
