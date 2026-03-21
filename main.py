"""
Main script: WMR kinodynamic modeling via System Identification.

Reproduces the experimental results of:
  Lee, Paulik, Krishnan - "Wheeled Mobile Robot Modeling for Local Navigation
  Using System Identification" - MWSCAS 2023

Pipeline:
  1. Create physics-based robot simulator (Pioneer P3-DX parameters)
  2. Generate random multi-step training and validation datasets
  3. Fit MIMO ARX model (paper's asymmetric order configuration)
  4. Evaluate model fit percentage on validation data
  5. (Optional) Model order study: sweep ARX order 1-15
  6. Generate and compare trajectories against ground truth
  7. Produce paper-equivalent figures in ./results/
"""

import os
import numpy as np

from robot_simulator import DifferentialDriveRobot
from data_generator import DataGenerator
from arx_model import MIMOARXModel
from trajectory_generator import TrajectoryGenerator
import visualize as viz


# ─────────────────────────────── Configuration ────────────────────────────────

CONFIG = {
    # Pioneer P3-DX physical parameters
    'wheel_radius':         0.0975,  # m
    'wheel_base':           0.381,   # m (distance between wheels)
    'max_wheel_speed':      1.5,     # m/s
    'motor_time_constant':  0.1,     # s  (1st-order lag τ; pole = 1 - dt/τ = 0.9)
    'dt':                   0.01,    # simulation timestep (paper uses 0.01s)

    # Training data generation
    # Paper: "50 two-second duration multi-step sequences" → 50×200 = 10000 samples
    # Use 8000 training samples at dt=0.01s (80 seconds of data)
    'n_samples':        10000,
    'train_split':      0.8,
    'v_range':          (-1.0, 1.0),
    'omega_range':      (-1.5, 1.5),
    'min_hold_steps':   100,   # 1 second (matches paper's 2-s intervals minimum)
    'max_hold_steps':   200,   # 2 seconds
    'rng_seed':         42,

    # ARX model — paper's best configuration (Section IV)
    'na_diag':   9,   # diagonal AR order (y11, y22)
    'na_cross':  3,   # cross AR order (y12, y21)
    'nb_diag':   9,   # diagonal exogenous order (u11, u22)
    'nb_cross':  1,   # cross exogenous order (u12, u21)
    'ridge_alpha': 1e-8,

    # Evaluation
    'run_order_study':    True,
    'order_study_range':  list(range(1, 16)),

    # Output
    'output_dir': './results',
}


# ──────────────────────────────── Main pipeline ────────────────────────────────

def main():
    cfg = CONFIG
    os.makedirs(cfg['output_dir'], exist_ok=True)
    outdir = cfg['output_dir']

    print("=" * 65)
    print("WMR Kinodynamic Modeling via System Identification")
    print("Lee, Paulik, Krishnan — MWSCAS 2023 (Python Implementation)")
    print("=" * 65)

    # ── Step 1: Create robot ──────────────────────────────────────────────────
    robot = DifferentialDriveRobot(
        wheel_radius=cfg['wheel_radius'],
        wheel_base=cfg['wheel_base'],
        max_wheel_speed=cfg['max_wheel_speed'],
        motor_time_constant=cfg['motor_time_constant'],
        dt=cfg['dt'],
    )
    print(f"\n[1] Robot: Pioneer P3-DX | dt={cfg['dt']}s | "
          f"motor τ={cfg['motor_time_constant']}s | pole={robot._pole:.3f}")

    # ── Step 2: Generate training/validation data ─────────────────────────────
    print(f"\n[2] Generating data: {cfg['n_samples']} steps "
          f"({cfg['train_split']*100:.0f}% train / "
          f"{(1-cfg['train_split'])*100:.0f}% val)...")

    gen = DataGenerator(robot, rng_seed=cfg['rng_seed'])
    dataset = gen.generate_dataset(
        n_samples=cfg['n_samples'],
        split=cfg['train_split'],
        v_range=cfg['v_range'],
        omega_range=cfg['omega_range'],
        min_hold_steps=cfg['min_hold_steps'],
        max_hold_steps=cfg['max_hold_steps'],
    )

    n_train = len(dataset['train']['t'])
    n_val = len(dataset['val']['t'])
    print(f"   Training samples: {n_train} | Validation samples: {n_val}")

    viz.plot_training_data(dataset, save_path=f"{outdir}/training_data.png")

    # ── Step 3: Fit ARX model ─────────────────────────────────────────────────
    print(f"\n[3] Fitting MIMO ARX model "
          f"(na_diag={cfg['na_diag']}, na_cross={cfg['na_cross']}, "
          f"nb_diag={cfg['nb_diag']}, nb_cross={cfg['nb_cross']})...")

    arx = MIMOARXModel(
        na_diag=cfg['na_diag'],
        na_cross=cfg['na_cross'],
        nb_diag=cfg['nb_diag'],
        nb_cross=cfg['nb_cross'],
        ridge_alpha=cfg['ridge_alpha'],
    )
    arx.fit(dataset['train']['y'], dataset['train']['u'])
    print("   Model fitted.")

    # Check stability
    stability = arx.check_stability()
    print(f"   Stability: v-channel={'STABLE' if stability['stable_v'] else 'UNSTABLE'} "
          f"(max |eig|={stability['max_eigval_v']:.4f}), "
          f"ω-channel={'STABLE' if stability['stable_omega'] else 'UNSTABLE'} "
          f"(max |eig|={stability['max_eigval_omega']:.4f})")

    # ── Step 4: Validate on held-out data ─────────────────────────────────────
    print("\n[4] Evaluating one-step-ahead prediction on validation set...")

    val_y = dataset['val']['y']
    val_u = dataset['val']['u']
    val_t = dataset['val']['t']

    # One-step prediction requires at least max_lag history from training
    # Prepend last max_lag training samples as context
    context_y = dataset['train']['y'][-arx.max_lag:]
    context_u = dataset['train']['u'][-arx.max_lag:]
    full_val_y = np.vstack([context_y, val_y])
    full_val_u = np.vstack([context_u, val_u])

    y_hat_full = arx.predict(full_val_y, full_val_u, mode='one_step')
    # Trim context
    y_hat_val = y_hat_full[arx.max_lag:]
    # Re-align: predictions start at max_lag within the trimmed window
    # The first max_lag rows of y_hat_val will be NaN due to warmup
    metrics = arx.score(val_y, y_hat_val)

    print(f"   Fit%: v={metrics['fit_v']:.1f}%  ω={metrics['fit_omega']:.1f}%")
    print(f"   RMSE: v={metrics['rmse_v']:.4f} m/s  ω={metrics['rmse_omega']:.4f} rad/s")
    print(f"   R²:   v={metrics['r2_v']:.4f}  ω={metrics['r2_omega']:.4f}")

    # Paper target: v: 85-95%, ω: 67-96%
    _check_targets(metrics)

    t_shifted = np.arange(len(val_y)) * cfg['dt']
    viz.plot_velocity_fit(
        t_shifted, val_y, y_hat_val, metrics,
        title="ARX One-Step-Ahead Prediction (Validation Set)",
        save_path=f"{outdir}/velocity_fit.png",
    )
    viz.plot_residuals(
        t_shifted, val_y, y_hat_val,
        save_path=f"{outdir}/residuals.png",
    )

    # ── Step 5 (optional): Model order study ─────────────────────────────────
    if cfg['run_order_study']:
        print("\n[5] Model order study (sweep ARX diagonal order 1-15)...")
        orders, fit_v_list, fit_omega_list = _model_order_study(
            robot, gen, dataset, cfg, arx.max_lag
        )
        viz.plot_model_order_study(
            orders, fit_v_list, fit_omega_list,
            save_path=f"{outdir}/model_order_study.png",
        )
    else:
        print("\n[5] Model order study skipped (set run_order_study=True to enable).")

    # ── Step 6: Trajectory comparison ────────────────────────────────────────
    print("\n[6] Generating trajectory comparisons...")

    tgen = TrajectoryGenerator(arx, dt=cfg['dt'])

    for traj_name in ['figure8', 'square', 'spiral']:
        print(f"   Trajectory: {traj_name}")
        v_cmds, omega_cmds = gen.generate_trajectory_commands(
            trajectory_type=traj_name, n_samples=5000   # 50 seconds at dt=0.01
        )
        robot.reset()
        comparison = tgen.compare_with_ground_truth(robot, v_cmds, omega_cmds)

        final_err = comparison['position_error'][-1]
        mean_err = np.mean(comparison['position_error'][arx.max_lag:])
        print(f"   Final position error: {final_err:.4f} m  |  Mean: {mean_err:.4f} m")

        viz.plot_trajectory_comparison(
            comparison,
            title=f"Path Comparison: {traj_name.title()} Trajectory",
            save_path=f"{outdir}/trajectory_{traj_name}.png",
        )

    # ── Step 7: Fan plot (paper Figure 3 equivalent) ─────────────────────────
    print("\n[7] Generating fan trajectories (paper Fig. 3 style)...")
    viz.plot_fan_trajectories(
        robot, arx, dt=cfg['dt'],
        omega_finals=np.linspace(-2.0, 2.0, 9),
        n_steps=300,   # 3 seconds at dt=0.01s
        save_path=f"{outdir}/fan_trajectories.png",
    )

    print(f"\nDone. All figures saved to: {outdir}/")
    print("\nResults summary:")
    print(f"  v  fit%: {metrics['fit_v']:.1f}%  (paper: 85.1–94.76%)")
    print(f"  ω  fit%: {metrics['fit_omega']:.1f}%  (paper: 67.2–95.68%)")


# ──────────────────────────────── Helpers ─────────────────────────────────────

def _check_targets(metrics: dict) -> None:
    v_ok = metrics['fit_v'] >= 80
    o_ok = metrics['fit_omega'] >= 60
    if v_ok and o_ok:
        print("   [OK] Both channels meet paper-level performance targets.")
    else:
        if not v_ok:
            print(f"   [WARN] v fit% ({metrics['fit_v']:.1f}%) below 80% target.")
        if not o_ok:
            print(f"   [WARN] ω fit% ({metrics['fit_omega']:.1f}%) below 60% target.")


def _model_order_study(robot, gen, dataset, cfg, max_lag_base):
    """Sweep ARX diagonal order and record fit% for each."""
    orders = cfg['order_study_range']
    fit_v_list = []
    fit_omega_list = []

    val_y = dataset['val']['y']
    val_u = dataset['val']['u']
    context_y_base = dataset['train']['y'][-max_lag_base:]
    context_u_base = dataset['train']['u'][-max_lag_base:]

    for order in orders:
        model = MIMOARXModel(
            na_diag=order,
            na_cross=min(3, order),
            nb_diag=order,
            nb_cross=min(1, order),
            ridge_alpha=cfg['ridge_alpha'],
        )
        model.fit(dataset['train']['y'], dataset['train']['u'])

        ml = model.max_lag
        context_y = dataset['train']['y'][-ml:]
        context_u = dataset['train']['u'][-ml:]
        full_y = np.vstack([context_y, val_y])
        full_u = np.vstack([context_u, val_u])

        y_hat = model.predict(full_y, full_u, mode='one_step')
        y_hat_val = y_hat[ml:]
        m = model.score(val_y, y_hat_val)

        fit_v_list.append(m['fit_v'])
        fit_omega_list.append(m['fit_omega'])
        print(f"   Order {order:2d}: fit_v={m['fit_v']:.1f}%  fit_ω={m['fit_omega']:.1f}%")

    return orders, fit_v_list, fit_omega_list


if __name__ == '__main__':
    main()
