"""
Visualization routines for WMR system identification results.

All plotting functions receive pre-computed arrays and render matplotlib figures.
Matches the style of Figures 3 and 4 in the paper.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend (safe for headless environments)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os


def plot_training_data(dataset: dict, save_path: str = None) -> None:
    """
    Plot the training data: commands and measured robot responses.
    Shows the multi-step command structure and resulting velocity responses.
    """
    d = dataset['train']
    t = d['t']
    u = d['u']
    y = d['y']

    # Use at most 500 points for clarity
    idx = slice(0, min(500, len(t)))

    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Training Data: Commands and Robot Responses", fontsize=13)

    axes[0].step(t[idx], u[idx, 0], where='post', color='steelblue')
    axes[0].set_ylabel("v_cmd (m/s)")
    axes[0].grid(True, alpha=0.4)

    axes[1].step(t[idx], u[idx, 1], where='post', color='darkorange')
    axes[1].set_ylabel("ω_cmd (rad/s)")
    axes[1].grid(True, alpha=0.4)

    axes[2].plot(t[idx], y[idx, 0], color='steelblue', linewidth=1.2)
    axes[2].set_ylabel("v_actual (m/s)")
    axes[2].grid(True, alpha=0.4)

    axes[3].plot(t[idx], y[idx, 1], color='darkorange', linewidth=1.2)
    axes[3].set_ylabel("ω_actual (rad/s)")
    axes[3].set_xlabel("Time (s)")
    axes[3].grid(True, alpha=0.4)

    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_velocity_fit(
    t: np.ndarray,
    y_true: np.ndarray,
    y_hat: np.ndarray,
    metrics: dict,
    title: str = "ARX One-Step-Ahead Prediction",
    save_path: str = None,
) -> None:
    """
    Plot true vs predicted velocities for both channels.

    Args:
        t:       time vector, shape (N,)
        y_true:  shape (N, 2) — [v_true, omega_true]
        y_hat:   shape (N, 2) — [v_hat, omega_hat], may contain NaN for warmup
        metrics: output of arx_model.score()
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle(title, fontsize=13)

    for i, (ax, name, unit) in enumerate(zip(
        axes,
        ['Linear velocity v', 'Angular velocity ω'],
        ['m/s', 'rad/s'],
    )):
        fit_key = 'fit_v' if i == 0 else 'fit_omega'
        rmse_key = 'rmse_v' if i == 0 else 'rmse_omega'
        label_true = f"True ({name})"
        label_pred = f"ARX predicted  (fit={metrics[fit_key]:.1f}%, RMSE={metrics[rmse_key]:.4f} {unit})"

        ax.plot(t, y_true[:, i], color='steelblue', linewidth=1.2, label=label_true)
        ax.plot(t, y_hat[:, i], color='crimson', linewidth=1.0, linestyle='--', label=label_pred)
        ax.set_ylabel(f"{name} ({unit})")
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.4)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_trajectory_comparison(
    comparison: dict,
    title: str = "Path Comparison: Ground Truth vs ARX Model",
    save_path: str = None,
) -> None:
    """
    XY path plot comparing kinematic (ideal), ground truth, and ARX model paths.

    Matches Figure 4 in the paper.
    """
    gt = comparison['ground_truth']
    arx = comparison['arx_model']
    kin = comparison['kinematic']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=13)

    for ax, title_sub in zip(axes, ['Path (XY)', 'Position Error over Time']):
        ax.set_title(title_sub)

    ax = axes[0]
    ax.plot(kin['x'], kin['y'], 'g--', linewidth=1.5, label='Ideal (kinematic)', alpha=0.7)
    ax.plot(gt['x'], gt['y'], 'b-', linewidth=2.0, label='Ground truth (physics)', alpha=0.8)
    ax.plot(arx['x'], arx['y'], 'r-', linewidth=1.5, label='ARX model', alpha=0.8)

    # Mark start
    ax.plot(gt['x'][0], gt['y'][0], 'ko', markersize=6)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)
    ax.set_aspect('equal')

    ax = axes[1]
    t = gt['t']
    ax.plot(t, comparison['position_error'], 'r-', linewidth=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position error (m)")
    ax.grid(True, alpha=0.4)
    ax.set_title(f"Final error: {comparison['position_error'][-1]:.3f} m")

    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_model_order_study(
    orders: list,
    fit_v: list,
    fit_omega: list,
    save_path: str = None,
) -> None:
    """
    Plot fit% vs ARX order for both output channels.

    Used to justify the paper's selection of order 9 for diagonal channels.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(orders, fit_v, 'b-o', label='v (linear velocity)', markersize=5)
    ax.plot(orders, fit_omega, 'r-s', label='ω (angular velocity)', markersize=5)

    ax.axhline(85, color='b', linestyle=':', alpha=0.5, label='Paper lower bound (v)')
    ax.axhline(67, color='r', linestyle=':', alpha=0.5, label='Paper lower bound (ω)')

    ax.set_xlabel("ARX diagonal order (na_diag = nb_diag)")
    ax.set_ylabel("Fit % (one-step-ahead)")
    ax.set_title("Model Order Study: Fit% vs ARX Order")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)
    ax.set_xticks(orders)

    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_residuals(
    t: np.ndarray,
    y_true: np.ndarray,
    y_hat: np.ndarray,
    save_path: str = None,
) -> None:
    """
    Plot residuals (prediction error) for both channels.

    Good residuals should look like white noise; systematic structure
    indicates the model order is too low or the system is nonlinear.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 6))
    fig.suptitle("Prediction Residuals", fontsize=13)

    names = ['v (m/s)', 'ω (rad/s)']
    colors = ['steelblue', 'darkorange']

    for i in range(2):
        residual = y_true[:, i] - y_hat[:, i]
        mask = ~np.isnan(residual)
        r = residual[mask]
        t_r = t[mask]

        axes[i, 0].plot(t_r, r, color=colors[i], linewidth=0.8, alpha=0.8)
        axes[i, 0].axhline(0, color='k', linestyle='--', linewidth=0.8)
        axes[i, 0].set_ylabel(f"Residual {names[i]}")
        axes[i, 0].set_xlabel("Time (s)")
        axes[i, 0].set_title(f"Residual time series ({names[i]})")
        axes[i, 0].grid(True, alpha=0.4)

        axes[i, 1].hist(r, bins=40, color=colors[i], edgecolor='white', alpha=0.8)
        axes[i, 1].set_xlabel(f"Residual {names[i]}")
        axes[i, 1].set_ylabel("Count")
        axes[i, 1].set_title(f"Residual histogram (std={np.std(r):.4f})")
        axes[i, 1].grid(True, alpha=0.4)

    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_fan_trajectories(
    robot,
    arx_model,
    dt: float = 0.1,
    v_fixed: float = 0.5,
    omega_start: float = 0.0,
    omega_finals: list = None,
    n_steps: int = 30,
    save_path: str = None,
) -> None:
    """
    Reproduce Figure 3 from the paper: a fan of 3-second trajectories
    with fixed v_in, ω_in = 0 initially, then stepping to different ω_final.

    For each ω_final, the command sequence is:
      first half: (v_fixed, 0)
      second half: (v_fixed, omega_final)
    """
    from trajectory_generator import TrajectoryGenerator
    tgen = TrajectoryGenerator(arx_model, dt=dt)

    if omega_finals is None:
        omega_finals = np.linspace(-2.0, 2.0, 9)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_title(f"Fan of {n_steps * dt:.0f}-second trajectories\n"
                 f"v_in={v_fixed} m/s, ω_in=0→ω_final (paper Fig. 3 style)")

    cmap = plt.cm.coolwarm
    colors = [cmap(i / (len(omega_finals) - 1)) for i in range(len(omega_finals))]

    for omega_f, color in zip(omega_finals, colors):
        half = n_steps // 2
        v_cmds = np.full(n_steps, v_fixed)
        omega_cmds = np.concatenate([np.zeros(half), np.full(n_steps - half, omega_f)])

        robot.reset()
        traj_gt = {'x': np.zeros(n_steps), 'y': np.zeros(n_steps)}
        for k in range(n_steps):
            s = robot.step(v_cmds[k], omega_cmds[k])
            traj_gt['x'][k] = s['x']
            traj_gt['y'][k] = s['y']

        robot.reset()
        traj_arx = tgen.generate(v_cmds, omega_cmds, warmup_robot=robot)

        ax.plot(traj_gt['x'], traj_gt['y'], '-', color=color, linewidth=2.0, alpha=0.8)
        ax.plot(traj_arx['x'], traj_arx['y'], '--', color=color, linewidth=1.5, alpha=0.7)

    # Legend
    gt_patch = mpatches.Patch(color='gray', label='Ground truth (solid)')
    arx_patch = mpatches.Patch(color='gray', linestyle='--', label='ARX model (dashed)')
    ax.legend(handles=[gt_patch, arx_patch], fontsize=9)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(omega_finals[0], omega_finals[-1]))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='ω_final (rad/s)')

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.4)
    ax.set_aspect('equal')

    plt.tight_layout()
    _save_or_show(fig, save_path)


def _save_or_show(fig, save_path: str) -> None:
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)
