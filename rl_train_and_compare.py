import os, sys
# Windows: prevent duplicate OpenMP runtime crash (torch + scipy/numpy both ship libiomp5)
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
# Force UTF-8 output on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

"""
RL-based Lazy Phase Compensator -- Training, Evaluation & Paper Comparison
==========================================================================

What the paper says (Lee, Paulik, Krishnan -- MWSCAS 2023)
----------------------------------------------------------
- A differential-drive robot suffers a "lazy phase": finite motor acceleration
  causes actual velocities to lag behind commanded velocities.
- The paper models this with a 1st-order discrete lag (pole p = 0.90) and
  uses MIMO ARX system identification to *predict* the resulting trajectories.
- ARX fit: v ~ 85-95%, omega ~ 67-96% on flat terrain.
- The paper does NOT actively compensate -- it predicts where the robot will go.

Our RL contribution
-------------------
1. Flat terrain (paper setting): An SAC agent learns to issue *pre-emptive*
   commands that, after passing through the motor lag, keep the robot on the
   kinematic reference trajectory -- beating the uncompensated kinematic
   baseline and showing active improvement over passive ARX prediction.

2. 3D terrain (novel extension): On slopes the effective motor time constant
   grows (more load going uphill).  The ARX model -- trained on flat ground --
   cannot adapt.  The RL agent observes the local slope and compensates,
   maintaining trajectory accuracy where ARX degrades.

Comparison methods
------------------
  KINEMATIC  - No compensation: command = desired velocity directly.
               Robot lags; position error accumulates.
  ARX        - Paper method: ARX model *predicts* what the robot will do
               under kinematic commands (no closed-loop correction).
  RL-SAC     - Our trained agent: closed-loop, compensates for lag & slope.

Usage
-----
  # Train from scratch (~ 5-10 min on CPU)
  python rl_train_and_compare.py --train

  # Evaluate a saved model, generate all comparison plots
  python rl_train_and_compare.py --eval

  # Open PyBullet GUI and watch the robot navigate 3D terrain
  python rl_train_and_compare.py --gui

  # All-in-one: train, evaluate, visualise
  python rl_train_and_compare.py --train --eval --gui
"""

import argparse
import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pybullet as p

# -- Stable Baselines 3 --------------------------------------------------------
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback, BaseCallback
)

# -- Project modules -----------------------------------------------------------
from rl_lazy_phase_env import (
    LazyPhaseEnv3D, kinematic_integrate,
    DT, V_MAX, W_MAX, MOTOR_TAU, BASE_H,
)
from robot_simulator import DifferentialDriveRobot, _wrap_to_pi
from arx_model import MIMOARXModel
from data_generator import DataGenerator
from trajectory_generator import TrajectoryGenerator, _kinematic_trajectory

# -- Paths ---------------------------------------------------------------------
RESULTS_DIR = './results'
MODEL_DIR   = os.path.join(RESULTS_DIR, 'rl_models')
MODEL_PATH  = os.path.join(MODEL_DIR, 'sac_lazy_phase_final')

FAN_V      = 0.30     # m/s  -- fixed forward speed for fan trajectories
FAN_OMEGAS = np.linspace(-1.8, 1.8, 7)   # range of omega steps (rad/s)
FAN_STEPS  = 300      # 3 seconds at dt=0.01 s


# ==============================================================================
#  Reward-logging callback
# ==============================================================================

class RewardLogCallback(BaseCallback):
    """Records mean episode reward every `log_freq` steps."""

    def __init__(self, log_freq: int = 5000):
        super().__init__(verbose=0)
        self.log_freq = log_freq
        self.rewards  = []
        self.steps    = []
        self._ep_buf  = []

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info:
                self._ep_buf.append(info['episode']['r'])
        if self.num_timesteps % self.log_freq == 0 and self._ep_buf:
            self.rewards.append(float(np.mean(self._ep_buf[-20:])))
            self.steps.append(self.num_timesteps)
        return True


# ==============================================================================
#  Training
# ==============================================================================

def make_env(terrain_type: str, seed: int, max_steps: int = 3000):
    def _init():
        env = LazyPhaseEnv3D(
            terrain_type=terrain_type,
            use_gui=False,
            max_steps=max_steps,
            traj_type='random',
            seed=seed,
            slope_effect=True,
        )
        return Monitor(env)
    return _init


def train(n_steps: int = 300_000):
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 65)
    print("RL Lazy-Phase Compensator  --  SAC Training")
    print("=" * 65)
    print(f"  Total timesteps : {n_steps:,}")
    print(f"  Phase 1 (flat)  : {int(n_steps * 0.6):,} steps")
    print(f"  Phase 2 (mixed) : {int(n_steps * 0.4):,} steps")
    print(f"  Output dir      : {MODEL_DIR}")
    print()

    reward_cb = RewardLogCallback(log_freq=5_000)

    # -- Phase 1: flat terrain (learn basic lazy-phase compensation) -----------
    env_flat = DummyVecEnv([make_env('flat', seed=0)])
    eval_flat = DummyVecEnv([make_env('flat', seed=99, max_steps=500)])

    model = SAC(
        'MlpPolicy',
        env_flat,
        learning_rate=3e-4,
        buffer_size=200_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        ent_coef='auto',
        train_freq=1,
        gradient_steps=1,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=0,
    )

    ckpt_cb = CheckpointCallback(
        save_freq=50_000, save_path=MODEL_DIR, name_prefix='sac_flat'
    )
    eval_cb = EvalCallback(
        eval_flat,
        best_model_save_path=os.path.join(MODEL_DIR, 'best'),
        eval_freq=25_000, n_eval_episodes=5, verbose=0,
    )

    print("[Phase 1] Training on flat terrain ...")
    t0 = time.time()
    model.learn(
        total_timesteps=int(n_steps * 0.6),
        callback=[ckpt_cb, eval_cb, reward_cb],
        progress_bar=True,
    )
    print(f"          Done in {time.time()-t0:.0f}s")

    # -- Phase 2: mixed/3D terrain (fine-tune for slope adaptation) ------------
    env_3d = DummyVecEnv([make_env('mixed', seed=1)])
    model.set_env(env_3d)

    ckpt_cb2 = CheckpointCallback(
        save_freq=50_000, save_path=MODEL_DIR, name_prefix='sac_mixed'
    )

    print("\n[Phase 2] Fine-tuning on mixed (3D) terrain ...")
    t1 = time.time()
    model.learn(
        total_timesteps=int(n_steps * 0.4),
        callback=[ckpt_cb2, reward_cb],
        progress_bar=True,
        reset_num_timesteps=False,
    )
    print(f"          Done in {time.time()-t1:.0f}s")

    # -- Save model ------------------------------------------------------------
    model.save(MODEL_PATH)
    print(f"\nModel saved -> {MODEL_PATH}.zip")

    # -- Save training curve ---------------------------------------------------
    if reward_cb.steps:
        _plot_training_curve(reward_cb.steps, reward_cb.rewards)

    return model, reward_cb


# ==============================================================================
#  ARX model (paper baseline)  --  quick re-train
# ==============================================================================

def _build_arx_model() -> MIMOARXModel:
    """Train the paper's ARX model on analytic flat-terrain data."""
    print("\n[ARX]  Training paper ARX model on flat terrain ...")
    robot = DifferentialDriveRobot(
        wheel_radius=0.0975, wheel_base=0.381,
        max_wheel_speed=1.5, motor_time_constant=0.1, dt=DT
    )
    gen = DataGenerator(robot, rng_seed=42)
    ds  = gen.generate_dataset(n_samples=8000, split=0.8,
                               v_range=(-1.0, 1.0), omega_range=(-1.5, 1.5),
                               min_hold_steps=100, max_hold_steps=200)
    arx = MIMOARXModel(na_diag=9, na_cross=3, nb_diag=9, nb_cross=1)
    arx.fit(ds['train']['y'], ds['train']['u'])
    m = arx.score(
        ds['val']['y'],
        arx.predict(
            np.vstack([ds['train']['y'][-arx.max_lag:], ds['val']['y']]),
            np.vstack([ds['train']['u'][-arx.max_lag:], ds['val']['u']]),
            mode='one_step',
        )[arx.max_lag:],
    )
    print(f"       ARX fit -- v: {m['fit_v']:.1f}%  w: {m['fit_omega']:.1f}%  "
          f"(paper target: 85-95%% / 67-96%%)")
    return arx


# ==============================================================================
#  Evaluation helpers
# ==============================================================================

def _run_episode(
    model,                  # SAC model or None (-> kinematic baseline)
    v_cmds: np.ndarray,
    omega_cmds: np.ndarray,
    terrain_type: str = 'flat',
    slope_effect: bool = True,
) -> dict:
    """
    Run one episode in PyBullet.

    If model is None, the agent issues the desired (kinematic) commands
    directly -- this demonstrates the uncompensated lazy phase.
    """
    ref_traj = kinematic_integrate(v_cmds, omega_cmds)
    N = len(v_cmds)

    env = LazyPhaseEnv3D(
        terrain_type=terrain_type,
        use_gui=False,
        max_steps=N,
        traj_type='fan',          # we override the ref traj manually
        fan_v=float(v_cmds[0]),
        fan_omega=float(omega_cmds[0]),
        fan_steps=N,
        slope_effect=slope_effect,
        seed=0,
    )
    env._ref_traj = ref_traj
    obs, _ = env.reset()
    env._ref_traj = ref_traj   # re-apply (reset may regenerate it)

    xs, ys, ths, vs, ws = [], [], [], [], []

    for k in range(N):
        if model is None:
            # Kinematic baseline: pass desired velocity directly
            action = np.array([
                np.clip(float(v_cmds[k]) / V_MAX,    -1, 1),
                np.clip(float(omega_cmds[k]) / W_MAX, -1, 1),
            ], dtype=np.float32)
        else:
            action, _ = model.predict(obs, deterministic=True)

        obs, _, terminated, truncated, _ = env.step(action)
        st = env.get_robot_state()
        xs.append(st['x']); ys.append(st['y']); ths.append(st['theta'])
        vs.append(st['v']); ws.append(st['omega'])
        if terminated or truncated:
            break

    env.close()
    n_actual = len(xs)
    return {
        'x':     np.array(xs),
        'y':     np.array(ys),
        'theta': np.array(ths),
        'v':     np.array(vs),
        'omega': np.array(ws),
        't':     np.arange(n_actual) * DT,
    }


def _arx_predict_trajectory(
    arx: MIMOARXModel,
    v_cmds: np.ndarray,
    omega_cmds: np.ndarray,
) -> dict:
    """
    Run the analytic robot simulator for ARX warmup then free-run predict.
    Returns {x, y, theta, v_hat, omega_hat, t}.
    """
    robot = DifferentialDriveRobot(
        wheel_radius=0.0975, wheel_base=0.381,
        max_wheel_speed=1.5, motor_time_constant=0.1, dt=DT
    )
    tgen = TrajectoryGenerator(arx, dt=DT)
    return tgen.generate(v_cmds, omega_cmds, warmup_robot=robot)


def _position_error(actual: dict, ref: dict) -> np.ndarray:
    n = min(len(actual['x']), len(ref['x']))
    return np.sqrt(
        (actual['x'][:n] - ref['x'][:n])**2 +
        (actual['y'][:n] - ref['y'][:n])**2
    )


# ==============================================================================
#  Evaluation & Comparison
# ==============================================================================

def evaluate(model_path: str = MODEL_PATH):
    print("\n" + "=" * 65)
    print("Evaluation & Paper Comparison")
    print("=" * 65)

    # Load RL model
    if not os.path.exists(model_path + '.zip'):
        print(f"[WARN] No saved model at {model_path}.zip  ->  using untrained model")
        env_tmp = LazyPhaseEnv3D(terrain_type='flat', use_gui=False, max_steps=300)
        model = SAC('MlpPolicy', env_tmp, verbose=0)
        env_tmp.close()
    else:
        model = SAC.load(model_path)
        print(f"Loaded model: {model_path}.zip")

    # ARX baseline
    arx = _build_arx_model()

    # -- Fan trajectory test (paper Figure 3 style) ----------------------------
    print("\n[1/4] Fan trajectories -- flat terrain (paper Figure 3 equivalent)")
    results_flat = _run_fan_evaluation(model, arx, terrain='flat')

    print("\n[2/4] Fan trajectories -- 3D (hills) terrain (novel extension)")
    results_3d = _run_fan_evaluation(model, arx, terrain='hills')

    # -- Velocity tracking during a hard turn transition ------------------------
    print("\n[3/4] Velocity tracking during lazy-phase transition")
    results_vel = _run_velocity_comparison(model, terrain='flat')

    # -- Generate all figures ---------------------------------------------------
    print("\n[4/4] Generating comparison figures ...")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    _plot_fan_comparison(results_flat, terrain_label='Flat Terrain',
                         save_path=os.path.join(RESULTS_DIR, 'rl_fan_flat.png'))
    _plot_fan_comparison(results_3d,   terrain_label='Hills Terrain (3D)',
                         save_path=os.path.join(RESULTS_DIR, 'rl_fan_3d.png'))
    _plot_position_error(results_flat, results_3d,
                         save_path=os.path.join(RESULTS_DIR, 'rl_position_error.png'))
    _plot_velocity_tracking(results_vel,
                            save_path=os.path.join(RESULTS_DIR, 'rl_velocity_tracking.png'))
    _plot_terrain_3d_paths(results_flat, results_3d)

    # -- Print summary table ----------------------------------------------------
    _print_summary(results_flat, results_3d)


def _run_fan_evaluation(model, arx, terrain: str) -> dict:
    """
    For each omega in FAN_OMEGAS run all three methods and collect results.
    """
    kinematic_paths, arx_paths, rl_paths, ref_paths = [], [], [], []
    kin_errs, arx_errs, rl_errs = [], [], []

    for omega_f in FAN_OMEGAS:
        v_cmds     = np.full(FAN_STEPS, FAN_V)
        omega_cmds = np.full(FAN_STEPS, omega_f)

        ref = {'x': kinematic_integrate(v_cmds, omega_cmds)[:, 0],
               'y': kinematic_integrate(v_cmds, omega_cmds)[:, 1]}

        # Kinematic baseline (no compensation)
        kin_res = _run_episode(None, v_cmds, omega_cmds,
                               terrain_type=terrain, slope_effect=(terrain != 'flat'))
        # ARX prediction (paper method -- analytic, always flat-terrain model)
        arx_res = _arx_predict_trajectory(arx, v_cmds, omega_cmds)
        # RL agent
        rl_res  = _run_episode(model, v_cmds, omega_cmds,
                               terrain_type=terrain, slope_effect=(terrain != 'flat'))

        kin_errs.append(float(np.mean(_position_error(kin_res, ref)[-100:])))
        arx_errs.append(float(np.mean(_position_error(
            {'x': arx_res['x'], 'y': arx_res['y']}, ref)[-100:])))
        rl_errs.append(float(np.mean(_position_error(rl_res, ref)[-100:])))

        ref_paths.append(ref)
        kinematic_paths.append(kin_res)
        arx_paths.append(arx_res)
        rl_paths.append(rl_res)

        print(f"  omega={omega_f:+.1f}  |  kin err={kin_errs[-1]:.4f}m  "
              f"arx err={arx_errs[-1]:.4f}m  rl err={rl_errs[-1]:.4f}m")

    return {
        'omega_finals': FAN_OMEGAS,
        'ref':   ref_paths,
        'kin':   kinematic_paths,
        'arx':   arx_paths,
        'rl':    rl_paths,
        'kin_errs': kin_errs,
        'arx_errs': arx_errs,
        'rl_errs':  rl_errs,
    }


def _run_velocity_comparison(model, terrain: str = 'flat') -> dict:
    """
    Hard velocity transition: robot at rest, then step v=0.6, omega=0.
    Then at t=1s switch to v=0.6, omega=1.5 (hard turn onset).
    Records commanded vs actual velocities for all three methods.
    """
    N = 400  # 4 s
    v_cmds     = np.concatenate([np.full(100, 0.0),  # 1s stopped
                                  np.full(100, 0.6),  # 1s straight
                                  np.full(200, 0.6)]) # 2s turning
    omega_cmds = np.concatenate([np.full(100, 0.0),
                                  np.full(100, 0.0),
                                  np.full(200, 1.5)])  # sudden hard turn

    kin_res = _run_episode(None,  v_cmds, omega_cmds, terrain, (terrain != 'flat'))
    rl_res  = _run_episode(model, v_cmds, omega_cmds, terrain, (terrain != 'flat'))

    return {
        'v_cmd':   v_cmds[:min(N, len(kin_res['v']))],
        'w_cmd':   omega_cmds[:min(N, len(kin_res['v']))],
        'kin_v':   kin_res['v'],
        'kin_w':   kin_res['omega'],
        'rl_v':    rl_res['v'],
        'rl_w':    rl_res['omega'],
        't':       np.arange(len(kin_res['v'])) * DT,
    }


# ==============================================================================
#  Plotting
# ==============================================================================

C_KIN = '#2980b9'   # blue
C_ARX = '#e67e22'   # orange
C_RL  = '#27ae60'   # green
C_REF = '#95a5a6'   # grey


def _plot_training_curve(steps, rewards):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, rewards, color=C_RL, linewidth=2)
    ax.set_xlabel('Training steps')
    ax.set_ylabel('Mean episode reward')
    ax.set_title('SAC Training Curve -- Lazy Phase Compensator')
    ax.grid(True, alpha=0.3)
    path = os.path.join(RESULTS_DIR, 'rl_training_curve.png')
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"Training curve -> {path}")


def _plot_fan_comparison(results: dict, terrain_label: str, save_path: str):
    """
    Fan trajectory comparison: one subplot per omega value, show all 3 methods.
    Mirrors paper Figure 3 but adds the RL-controlled path.
    """
    n_fan = len(results['omega_finals'])
    ncols = min(4, n_fan)
    nrows = (n_fan + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows),
                              squeeze=False)
    fig.suptitle(f'Fan Trajectory Comparison -- {terrain_label}\n'
                 f'(v_in = {FAN_V} m/s, step omega from 0)', fontsize=13, y=1.01)

    for idx, omega_f in enumerate(results['omega_finals']):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]

        ref = results['ref'][idx]
        kin = results['kin'][idx]
        arx = results['arx'][idx]
        rl  = results['rl'][idx]

        ax.plot(ref['x'],  ref['y'],   '--', color=C_REF, lw=1.5,
                label='Kinematic ideal')
        ax.plot(kin['x'],  kin['y'],   '-',  color=C_KIN, lw=1.8, alpha=0.85,
                label='No comp. (kin cmd)')
        ax.plot(arx['x'],  arx['y'],   '-',  color=C_ARX, lw=1.8, alpha=0.85,
                label='ARX pred. (paper)')
        ax.plot(rl['x'],   rl['y'],    '-',  color=C_RL,  lw=2.2,
                label='RL-SAC (ours)')

        ax.set_title(f'omega = {omega_f:+.1f} rad/s', fontsize=10)
        ax.set_xlabel('x (m)', fontsize=8)
        ax.set_ylabel('y (m)', fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.set_aspect('equal', 'datalim')
        ax.tick_params(labelsize=7)

    # Hide unused axes
    for idx in range(n_fan, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    # Single legend below the figure
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4,
               fontsize=10, bbox_to_anchor=(0.5, -0.04))

    fig.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"Fan comparison plot -> {save_path}")


def _plot_position_error(flat_res: dict, terrain_3d_res: dict, save_path: str):
    """
    Time-series position error: shows lazy-phase induced deviation and RL improvement.
    Two sub-rows: flat terrain (paper comparison) and 3D terrain (novel).
    """
    omega_idx = len(FAN_OMEGAS) // 2  # pick the middle omega (largest turn)
    t = np.arange(FAN_STEPS) * DT

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle('Position Error vs Time -- Lazy Phase Compensation\n'
                 f'(v={FAN_V} m/s, omega={FAN_OMEGAS[omega_idx]:.1f} rad/s step)',
                 fontsize=12)

    for ax, res, label in zip(axes,
                               [flat_res, terrain_3d_res],
                               ['Flat terrain (paper setting)',
                                'Hills terrain (3D -- novel extension)']):
        ref  = res['ref'][omega_idx]
        kin  = res['kin'][omega_idx]
        arx  = res['arx'][omega_idx]
        rl   = res['rl'][omega_idx]

        n = min(FAN_STEPS, len(kin['x']), len(arx['x']), len(rl['x']))
        e_kin = _position_error(kin, ref)[:n]
        e_arx = _position_error({'x': arx['x'], 'y': arx['y']}, ref)[:n]
        e_rl  = _position_error(rl,  ref)[:n]

        ax.plot(t[:n], e_kin, color=C_KIN, lw=2.0, label='No compensation (kinematic cmd)')
        ax.plot(t[:n], e_arx, color=C_ARX, lw=2.0, label='ARX prediction (paper)')
        ax.plot(t[:n], e_rl,  color=C_RL,  lw=2.5, label='RL-SAC compensator (ours)')

        # Paper range shading
        ax.axhspan(0, 0.05, alpha=0.08, color=C_RL, label='RL target: <5 cm')
        ax.set_ylabel('Position error (m)', fontsize=10)
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    axes[-1].set_xlabel('Time (s)', fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"Position error plot -> {save_path}")


def _plot_velocity_tracking(res: dict, save_path: str):
    """
    Show commanded vs actual velocities.  The RL agent pre-empts the command
    to compensate for lag; the kinematic baseline shows raw lag deviation.
    """
    t = res['t']
    n = len(t)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig.suptitle('Velocity Tracking -- Lazy Phase Compensation\n'
                 '(hard turn onset at t = 2 s)', fontsize=12)

    for ax, cmd_key, kin_key, rl_key, ylabel, title in [
        (axes[0], 'v_cmd', 'kin_v', 'rl_v',
         'Linear velocity (m/s)', 'Forward velocity v'),
        (axes[1], 'w_cmd', 'kin_w', 'rl_w',
         'Angular velocity (rad/s)', 'Angular velocity omega'),
    ]:
        cmd = res[cmd_key][:n]
        kin = res[kin_key][:n]
        rl  = res[rl_key][:n]
        n_  = min(len(cmd), len(kin), len(rl))

        ax.plot(t[:n_], cmd[:n_], '--', color=C_REF, lw=2.0, label='Commanded (desired)')
        ax.plot(t[:n_], kin[:n_], '-',  color=C_KIN, lw=2.0, label='No compensation -- actual')
        ax.plot(t[:n_], rl[:n_],  '-',  color=C_RL,  lw=2.5, label='RL-SAC -- actual')

        ax.axvline(2.0, color='red', lw=1.2, linestyle=':', alpha=0.7,
                   label='Lazy phase onset (hard turn)')
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (s)', fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"Velocity tracking plot -> {save_path}")


def _plot_terrain_3d_paths(flat_res: dict, hills_res: dict):
    """3D surface plots showing robot paths on flat and hilly terrain."""
    from terrain_generator import TerrainGenerator

    omega_idx = len(FAN_OMEGAS) // 2

    for res, ttype, fname in [
        (flat_res,  'flat',  'rl_terrain_3d_flat.png'),
        (hills_res, 'hills', 'rl_terrain_3d_hills.png'),
    ]:
        gen = TerrainGenerator(size=20.0, resolution=128, seed=42)
        gen.heights = gen._generate_heights(ttype)

        kin = res['kin'][omega_idx]
        rl  = res['rl'][omega_idx]
        ref_xy = res['ref'][omega_idx]
        ref_dict = {'x': ref_xy['x'], 'y': ref_xy['y']}

        robot_paths = {
            'Kinematic ideal':      ref_dict,
            'No comp. (kin cmd)':   {'x': kin['x'], 'y': kin['y']},
            'RL-SAC (ours)':        {'x': rl['x'],  'y': rl['y']},
        }
        save = os.path.join(RESULTS_DIR, fname)
        gen.visualize_3d(
            robot_paths=robot_paths,
            save_path=save,
            title=f'3D Terrain Path Comparison -- {ttype.title()}',
        )
        print(f"3D terrain plot -> {save}")


def _print_summary(flat_res: dict, hills_res: dict):
    """Print comparison table matching the paper's reporting style."""
    print()
    print("=" * 65)
    print("Results Summary -- Mean position error (m) over last 1 s")
    print("=" * 65)
    print(f"{'Method':<28} {'Flat terrain':>14} {'Hills terrain':>14}")
    print("-" * 65)

    for label, kin_k, arx_k, rl_k in [
        ('No compensation (kin cmd)', 'kin_errs', None, None),
        ('ARX prediction (paper)',    None, 'arx_errs', None),
        ('RL-SAC compensator (ours)', None, None, 'rl_errs'),
    ]:
        if kin_k:
            flat_e  = float(np.mean(flat_res[kin_k]))
            hills_e = float(np.mean(hills_res[kin_k]))
        elif arx_k:
            flat_e  = float(np.mean(flat_res[arx_k]))
            hills_e = float(np.mean(hills_res[arx_k]))
        else:
            flat_e  = float(np.mean(flat_res[rl_k]))
            hills_e = float(np.mean(hills_res[rl_k]))
        print(f"{label:<28} {flat_e:>14.4f} {hills_e:>14.4f}")

    print("-" * 65)
    paper_arx = float(np.mean(flat_res['arx_errs']))
    paper_kin = float(np.mean(flat_res['kin_errs']))
    rl_flat   = float(np.mean(flat_res['rl_errs']))
    rl_3d     = float(np.mean(hills_res['rl_errs']))
    kin_3d    = float(np.mean(hills_res['kin_errs']))

    print(f"\nRL improvement vs no-compensation (flat) : "
          f"{100*(paper_kin-rl_flat)/max(paper_kin,1e-9):.1f}%")
    print(f"RL improvement vs no-compensation (3D)   : "
          f"{100*(kin_3d-rl_3d)/max(kin_3d,1e-9):.1f}%")
    print(f"ARX fit% (paper reports 85.1-94.76% v, 67.2-95.68% omega)")
    print("=" * 65)


# ==============================================================================
#  GUI visualisation (actual robot simulation)
# ==============================================================================

def run_gui_simulation(model_path: str = MODEL_PATH, terrain_type: str = 'hills'):
    """
    Open PyBullet GUI and run the trained RL agent navigating 3D terrain.
    The robot is visible, the camera follows it.  Press Ctrl-C to stop.
    """
    print("\n" + "=" * 65)
    print(f"GUI Simulation  --  terrain: {terrain_type}")
    print("Close the PyBullet window or press Ctrl-C to exit.")
    print("=" * 65)

    if os.path.exists(model_path + '.zip'):
        model = SAC.load(model_path)
        print(f"Loaded RL model: {model_path}.zip")
    else:
        print("[WARN] No saved model -- running kinematic (uncompensated) baseline")
        model = None

    # Build a longer figure-8 reference for visual appeal
    N = 6000  # 60 s
    vc = np.full(N, 0.40)
    wc = np.zeros(N)
    half = N // 2
    wc[:half] =  0.80
    wc[half:] = -0.80

    env = LazyPhaseEnv3D(
        terrain_type=terrain_type,
        use_gui=True,
        max_steps=N,
        traj_type='figure8',
        slope_effect=True,
        seed=7,
    )
    obs, _ = env.reset()

    print("Simulation running ...  (robot navigates in PyBullet window)")
    step = 0
    try:
        while step < N:
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                ref_idx = min(step, len(env._ref_traj) - 1)
                vd = float(env._ref_traj[ref_idx, 3])
                wd = float(env._ref_traj[ref_idx, 4])
                action = np.array([vd / 1.0, wd / 2.0], dtype=np.float32)

            obs, _, terminated, truncated, _ = env.step(action)

            # Capture a frame every 50 steps and save (optional)
            if step % 100 == 0:
                st = env.get_robot_state()
                print(f"  step={step:5d}  x={st['x']:+.2f}  y={st['y']:+.2f} "
                      f" v={st['v']:.3f}  omega={st['omega']:+.3f}")

            if terminated or truncated:
                print("  Episode ended -- resetting ...")
                obs, _ = env.reset()

            step += 1
            time.sleep(DT * 0.5)   # slow down to ~2x real time for visibility

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        env.close()


# ==============================================================================
#  Capture & save simulation frames (for paper-quality figures)
# ==============================================================================

def capture_simulation_frames(
    model_path: str = MODEL_PATH,
    terrain_type: str = 'hills',
    n_frames: int = 8,
    output_dir: str = None,
):
    """
    Run a short simulation episode and save N evenly spaced camera frames.
    Frames are saved as PNG images that can be arranged in a figure grid.
    """
    output_dir = output_dir or os.path.join(RESULTS_DIR, 'sim_frames')
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(model_path + '.zip'):
        model = SAC.load(model_path)
    else:
        model = None

    N = 300  # 3-second episode for frame capture
    env = LazyPhaseEnv3D(
        terrain_type=terrain_type,
        use_gui=False,        # off-screen render
        max_steps=N,
        traj_type='fan',
        fan_v=FAN_V,
        fan_omega=1.2,
        fan_steps=N,
        slope_effect=True,
        seed=3,
    )
    obs, _ = env.reset()

    save_at = set(np.linspace(0, N - 1, n_frames, dtype=int))
    frame_idx = 0

    for k in range(N):
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            ref_idx = min(k, len(env._ref_traj) - 1)
            vd = float(env._ref_traj[ref_idx, 3])
            wd = float(env._ref_traj[ref_idx, 4])
            action = np.array([vd / 1.0, wd / 2.0], dtype=np.float32)

        obs, _, terminated, truncated, _ = env.step(action)

        if k in save_at:
            frame = env.render()
            path  = os.path.join(output_dir, f'frame_{frame_idx:03d}_t{k*DT:.2f}s.png')
            # Use matplotlib to save
            fig, ax = plt.subplots(figsize=(6.4, 4.8))
            ax.imshow(frame)
            ax.axis('off')
            ax.set_title(f't = {k*DT:.2f} s  |  terrain: {terrain_type}', fontsize=9)
            fig.tight_layout(pad=0.1)
            fig.savefig(path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            frame_idx += 1

        if terminated or truncated:
            break

    env.close()

    # Build a contact-sheet figure from all captured frames
    pngs = sorted(
        [os.path.join(output_dir, f) for f in os.listdir(output_dir)
         if f.endswith('.png')],
    )
    if pngs:
        ncols = 4
        nrows = (len(pngs) + ncols - 1) // ncols
        fig2, axes2 = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        for i, ax2 in enumerate(np.array(axes2).flat):
            if i < len(pngs):
                img = plt.imread(pngs[i])
                ax2.imshow(img)
                ax2.axis('off')
            else:
                ax2.set_visible(False)
        fig2.suptitle(f'PyBullet Simulation -- {terrain_type.title()} Terrain',
                      fontsize=14)
        sheet_path = os.path.join(RESULTS_DIR, f'rl_sim_frames_{terrain_type}.png')
        fig2.savefig(sheet_path, dpi=100, bbox_inches='tight')
        plt.close(fig2)
        print(f"Simulation frame sheet -> {sheet_path}")

    return output_dir


# ==============================================================================
#  Entry point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='RL Lazy-Phase Compensator -- train, evaluate, visualise'
    )
    parser.add_argument('--train',   action='store_true',
                        help='Train SAC agent from scratch')
    parser.add_argument('--eval',    action='store_true',
                        help='Evaluate saved model and generate comparison plots')
    parser.add_argument('--gui',     action='store_true',
                        help='Open PyBullet GUI and watch the robot navigate')
    parser.add_argument('--frames',  action='store_true',
                        help='Capture simulation frames (off-screen render)')
    parser.add_argument('--terrain', default='hills',
                        choices=['flat', 'hills', 'ramps', 'mixed'],
                        help='Terrain type for GUI / frame capture (default: hills)')
    parser.add_argument('--steps',   type=int, default=300_000,
                        help='Training timesteps (default: 300 000)')
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR,   exist_ok=True)

    if not any([args.train, args.eval, args.gui, args.frames]):
        # Default: do everything
        args.train  = True
        args.eval   = True
        args.frames = True

    if args.train:
        train(n_steps=args.steps)

    if args.eval:
        evaluate(MODEL_PATH)

    if args.frames:
        print("\n[Frames] Capturing off-screen simulation frames ...")
        capture_simulation_frames(MODEL_PATH, terrain_type=args.terrain)

    if args.gui:
        run_gui_simulation(MODEL_PATH, terrain_type=args.terrain)

    if args.eval or args.train:
        print(f"\nAll outputs saved to: {RESULTS_DIR}/")
        print("Key figures:")
        for f in [
            'rl_training_curve.png',
            'rl_fan_flat.png',
            'rl_fan_3d.png',
            'rl_position_error.png',
            'rl_velocity_tracking.png',
            'rl_terrain_3d_hills.png',
            f'rl_sim_frames_{args.terrain}.png',
        ]:
            fpath = os.path.join(RESULTS_DIR, f)
            status = '_' if os.path.exists(fpath) else '-'
            print(f"  {status} {f}")


if __name__ == '__main__':
    main()
