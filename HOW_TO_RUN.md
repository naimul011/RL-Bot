# How to Run — RL Bot: Lazy Phase Compensation Project

> Complete guide for running every script in this project, from installation
> through training, evaluation, GUI simulation, and visualization.

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Installation](#2-installation)
3. [Quick-Start Cheatsheet](#3-quick-start-cheatsheet)
4. [Script Reference](#4-script-reference)
   - 4.1 [main.py — Paper Reproduction (ARX)](#41-mainpy--paper-reproduction-arx)
   - 4.2 [rl_train_and_compare.py — RL Training & Evaluation](#42-rl_train_and_comparepy--rl-training--evaluation)
   - 4.3 [terrain_generator.py — Terrain Preview](#43-terrain_generatorpy--terrain-preview)
   - 4.4 [web_visualizer.py — Browser Animation](#44-web_visualizerpy--browser-animation)
5. [Training the RL Agent](#5-training-the-rl-agent)
6. [Running the PyBullet GUI Simulation](#6-running-the-pybullet-gui-simulation)
7. [Capturing Simulation Frames](#7-capturing-simulation-frames)
8. [Evaluating a Saved Model](#8-evaluating-a-saved-model)
9. [Understanding the Output Files](#9-understanding-the-output-files)
10. [Troubleshooting](#10-troubleshooting)
11. [Module Map (for developers)](#11-module-map-for-developers)

---

## 1. Project Structure

```
RL-Bot-main/
│
├── main.py                    # Paper ARX reproduction pipeline
├── rl_train_and_compare.py    # RL training, evaluation, GUI, frames
├── rl_lazy_phase_env.py       # Gymnasium environment (MDP)
├── terrain_generator.py       # PyBullet heightfield terrain
│
├── arx_model.py               # MIMO ARX model (paper method)
├── data_generator.py          # Training data generation
├── robot_simulator.py         # Analytic diff-drive simulator
├── pybullet_simulator.py      # PyBullet-backed robot simulator
├── trajectory_generator.py    # ARX trajectory free-run
├── visualize.py               # Matplotlib plot helpers
│
├── robot_3d.urdf              # Pioneer P3-DX robot (4-caster 3D)
├── robot.urdf                 # Pioneer P3-DX (flat terrain, 2-caster)
│
├── web_visualizer.py          # Flask browser animation server
│
├── results/                   # All output figures and models
│   ├── rl_models/             # Saved SAC checkpoints
│   │   ├── sac_lazy_phase_final.zip    ← best model (active)
│   │   ├── sac_mixed_230000_steps.zip  ← same model (backup)
│   │   ├── sac_mixed_280000_steps.zip
│   │   ├── sac_flat_150000_steps.zip
│   │   └── best/              ← EvalCallback best checkpoint
│   ├── sim_frames/            # Individual PyBullet frame PNGs
│   └── *.png                  # All comparison / terrain figures
│
├── requirements.txt
├── REPORT.md                  # Technical comparison report
└── HOW_TO_RUN.md              # This file
```

---

## 2. Installation

### 2.1 Prerequisites

- **Python 3.9 – 3.11** (3.11 recommended; 3.12 has PyBullet issues on Windows)
- **Anaconda** or any virtual environment manager
- Windows 10/11 or Linux (macOS also works)

### 2.2 Create a clean environment (recommended)

```bash
conda create -n rlbot python=3.11 -y
conda activate rlbot
```

### 2.3 Install dependencies

```bash
cd "path/to/RL-Bot-main"
pip install numpy scipy matplotlib pybullet gymnasium stable-baselines3 torch
```

Or use the requirements file (adds Flask for the web visualizer):

```bash
pip install -r requirements.txt
pip install gymnasium stable-baselines3 torch
```

### 2.4 Verify installation

```bash
python -c "import pybullet, gymnasium, stable_baselines3, torch; print('All OK')"
```

Expected output:
```
pybullet build time: Oct 21 2025 11:46:21
All OK
```

### 2.5 Windows-only: fix console encoding

All scripts handle this automatically. If you see `UnicodeEncodeError` with
`τ` or `ω` characters, force UTF-8 before running:

```cmd
set PYTHONIOENCODING=utf-8
```

Or add this once in your terminal profile:

```cmd
chcp 65001
```

---

## 3. Quick-Start Cheatsheet

| Goal | Command |
|------|---------|
| Reproduce the paper (ARX only) | `python main.py` |
| Train RL agent from scratch | `python rl_train_and_compare.py --train` |
| Evaluate saved model + plots | `python rl_train_and_compare.py --eval` |
| Watch robot in GUI (hills) | `python rl_train_and_compare.py --gui` |
| Watch robot in GUI (flat) | `python rl_train_and_compare.py --gui --terrain flat` |
| Capture frame contact sheet | `python rl_train_and_compare.py --frames` |
| All at once (train+eval+gui) | `python rl_train_and_compare.py --train --eval --gui` |
| Preview all terrain types | `python terrain_generator.py` |
| Browser animation | `python web_visualizer.py` then open `http://localhost:5000` |

---

## 4. Script Reference

### 4.1 `main.py` — Paper Reproduction (ARX)

Reproduces the MIMO ARX system identification pipeline from the paper.
**No GPU required.** Runs in ~30 seconds.

```bash
python main.py
```

**What it does:**

1. Creates a `DifferentialDriveRobot` (Pioneer P3-DX analytic model)
2. Generates 8,000-sample random multi-step training dataset
3. Fits the MIMO ARX model (order 9 diagonal, 3 cross, 1 cross-exog)
4. Evaluates NFIR fit% on validation set
5. (Optional) Sweeps ARX model order 1–15 to show order sensitivity
6. Generates 6 reference trajectories (figure-8, square, spiral, etc.)
7. Compares ARX-predicted vs. ground-truth paths
8. Saves all figures to `results/`

**Expected output (printed):**

```
[1] Robot: Pioneer P3-DX | dt=0.01s | tau=0.10s | pole=0.900
[2] Generating training data: 8000 samples ...
[3] Fitting ARX model ...
    Fit -- v: 98.8%  omega: 98.8%
    [OK] Both channels meet paper-level performance targets.
[4] Generating comparison trajectories ...
Done. All figures saved to: results/
  v  fit%: 98.8%  (paper: 85.1-94.76%)
  omega fit%: 98.8%  (paper: 67.2-95.68%)
```

**Outputs in `results/`:**

| File | Description |
|------|-------------|
| `training_data.png` | Multi-step commands + velocity responses |
| `velocity_fit.png` | ARX predicted vs actual velocities |
| `residuals.png` | Prediction residuals |
| `model_order_study.png` | Fit% vs ARX order sweep |
| `fan_trajectories.png` | Fan of 9 trajectories (paper Fig. 3 style) |
| `trajectory_figure8.png` | Figure-8 path comparison |
| `trajectory_square.png` | Square path comparison |
| `trajectory_spiral.png` | Spiral path comparison |
| `trajectory_slalom.png` | Slalom path comparison |
| `trajectory_step_burst.png` | Step-burst path comparison |
| `trajectory_zigzag.png` | Zigzag path comparison |

---

### 4.2 `rl_train_and_compare.py` — RL Training & Evaluation

The main RL pipeline. Has four independent flags that can be combined freely.

```
usage: rl_train_and_compare.py [-h] [--train] [--eval] [--gui] [--frames]
                               [--terrain {flat,hills,ramps,mixed}]
                               [--steps STEPS]
```

| Flag | Description |
|------|-------------|
| `--train` | Train SAC agent from scratch (~15 min CPU) |
| `--eval` | Evaluate saved model, generate all comparison plots |
| `--gui` | Open PyBullet GUI window, watch robot navigate live |
| `--frames` | Capture 8 off-screen frames + contact sheet |
| `--terrain` | Terrain for `--gui` and `--frames` (default: `hills`) |
| `--steps` | Training timesteps (default: 300,000) |

**Outputs in `results/`:**

| File | Produced by | Description |
|------|-------------|-------------|
| `rl_fan_flat.png` | `--eval` | 7-fan trajectories on flat terrain |
| `rl_fan_3d.png` | `--eval` | 7-fan trajectories on hills terrain |
| `rl_position_error.png` | `--eval` | Position error vs time, all methods |
| `rl_velocity_tracking.png` | `--eval` | Velocity step response comparison |
| `rl_terrain_3d_flat.png` | `--eval` | 3D terrain surface + robot paths (flat) |
| `rl_terrain_3d_hills.png` | `--eval` | 3D terrain surface + robot paths (hills) |
| `rl_sim_frames_hills.png` | `--frames` | Contact sheet of 8 simulation frames |
| `sim_frames/frame_*.png` | `--frames` | Individual frames |
| `rl_models/sac_lazy_phase_final.zip` | `--train` | Best saved model |
| `rl_models/sac_flat_*_steps.zip` | `--train` | Checkpoints (every 50k, flat phase) |
| `rl_models/sac_mixed_*_steps.zip` | `--train` | Checkpoints (every 50k, mixed phase) |

---

### 4.3 `terrain_generator.py` — Terrain Preview

Run standalone to preview all four terrain types as 2D colormaps and 3D surfaces.

```bash
python terrain_generator.py
```

**What it does:** generates `flat`, `hills`, `ramps`, `mixed` terrain heightfields
and saves 2D + 3D visualisation PNGs to `results/`.

**Outputs:**

```
results/terrain_2d_flat.png
results/terrain_2d_hills.png
results/terrain_2d_ramps.png
results/terrain_2d_mixed.png
results/terrain_3d_flat.png
results/terrain_3d_hills.png
results/terrain_3d_ramps.png
results/terrain_3d_mixed.png
```

No PyBullet GUI is opened — this is pure NumPy/Matplotlib.

---

### 4.4 `web_visualizer.py` — Browser Animation

Trains 6 models (Kinematic, FIR, ARX-3, ARX-9, ARMAX, OE) and serves an
interactive browser animation at `http://localhost:5000`.

```bash
pip install flask        # one-time install
python web_visualizer.py
```

Then open your browser to: **`http://localhost:5000`**

> Note: This runs an HTTP server. Press `Ctrl+C` in the terminal to stop it.

---

## 5. Training the RL Agent

### 5.1 Train with default settings (300k steps)

```bash
python rl_train_and_compare.py --train
```

Training uses a **two-phase curriculum**:

```
Phase 1 — Flat terrain   180,000 steps   (~9 min on CPU)
Phase 2 — Mixed terrain  120,000 steps   (~6 min on CPU)
                         ─────────────────────────────
Total                    300,000 steps   (~15 min)
```

You will see a `tqdm` progress bar in the terminal. SAC checkpoints are saved
every 50,000 steps automatically:

```
results/rl_models/sac_flat_50000_steps.zip
results/rl_models/sac_flat_100000_steps.zip
results/rl_models/sac_flat_150000_steps.zip
results/rl_models/sac_mixed_230000_steps.zip   ← Phase 2 starts at 180k
results/rl_models/sac_mixed_280000_steps.zip
results/rl_models/sac_lazy_phase_final.zip      ← final save
```

The best model (by eval reward) is also saved to `results/rl_models/best/`.

### 5.2 Train with more steps for better performance

```bash
python rl_train_and_compare.py --train --steps 600000
```

### 5.3 Resume from an existing checkpoint

The pipeline does not have a built-in `--resume` flag. To continue from a
checkpoint, edit the `MODEL_PATH` constant in `rl_train_and_compare.py`:

```python
# Line ~91 in rl_train_and_compare.py
MODEL_PATH = './results/rl_models/sac_mixed_230000_steps'  # load this
```

Then in a Python script or the REPL:

```python
from stable_baselines3 import SAC
from rl_lazy_phase_env import LazyPhaseEnv3D

model = SAC.load('./results/rl_models/sac_mixed_230000_steps')
env = LazyPhaseEnv3D(terrain_type='mixed', use_gui=False, max_steps=3000)
model.set_env(env)
model.learn(total_timesteps=100_000, reset_num_timesteps=False)
model.save('./results/rl_models/sac_extended')
env.close()
```

### 5.4 Monitor training speed

The simulator runs at roughly **1,000 steps/second** on CPU. Expected rates:

| Hardware | Steps/sec | 300k steps |
|----------|-----------|------------|
| Modern laptop CPU | ~900–1,200 | ~5–6 min |
| Older desktop CPU | ~500–800 | ~8–10 min |
| GPU (CUDA) | ~1,000–2,000* | ~3–5 min |

> *PyBullet physics runs on CPU even with a GPU. The GPU only accelerates
> the neural network forward/backward passes (a minor fraction of total time).

---

## 6. Running the PyBullet GUI Simulation

The GUI opens a real-time 3D window showing the robot navigating terrain.
The camera follows the robot automatically.

### 6.1 Watch the RL agent on hills terrain (default)

```bash
python rl_train_and_compare.py --gui
```

### 6.2 Watch on different terrains

```bash
python rl_train_and_compare.py --gui --terrain flat
python rl_train_and_compare.py --gui --terrain ramps
python rl_train_and_compare.py --gui --terrain mixed
```

### 6.3 What you will see

- A **blue Pioneer P3-DX** robot with dark wheels and grey ball casters
- The terrain rendered with colour-coded elevation
  - Flat: light green
  - Hills: medium green
  - Ramps: sandy brown
  - Mixed: forest green
- The robot traces a **figure-8** trajectory (60 seconds, 6,000 steps)
- Camera follows from above-rear, updating every step
- Terminal prints robot pose and velocity every 100 steps:

```
step=    0  x=+0.00  y=+0.01  v=-0.086  omega=+0.125
step=  100  x=+0.32  y=+0.13  v=0.461   omega=+0.871
step=  200  x=+0.52  y=+0.49  v=0.414   omega=+0.879
...
```

### 6.4 Controls in the PyBullet GUI window

| Action | How |
|--------|-----|
| Rotate camera | Left-click + drag |
| Zoom | Scroll wheel |
| Pan | Right-click + drag |
| Reset camera | Middle-click |
| Stop simulation | Close the window, or `Ctrl+C` in terminal |

### 6.5 Speed control

By default the simulation runs at **2× real time** (sleeps 5 ms between 10 ms steps).
To change this, edit `rl_train_and_compare.py` line ~770:

```python
time.sleep(DT * 0.5)   # 0.5 → 2× real time
                        # 1.0 → real time
                        # 0.0 → as fast as possible
```

---

## 7. Capturing Simulation Frames

Runs the simulation in **off-screen mode** (no window) and saves PNG frames.

```bash
python rl_train_and_compare.py --frames
python rl_train_and_compare.py --frames --terrain mixed
```

**What it saves:**

```
results/sim_frames/frame_000_t0.00s.png    ← individual frames
results/sim_frames/frame_001_t0.42s.png
...
results/sim_frames/frame_007_t2.99s.png
results/rl_sim_frames_hills.png            ← 2×4 contact sheet
```

Each frame is 640×480 rendered via `p.getCameraImage()` with an overhead-rear
camera angle. The contact sheet is ready to embed in reports or presentations.

---

## 8. Evaluating a Saved Model

```bash
python rl_train_and_compare.py --eval
```

### 8.1 What evaluation does

1. **Loads** `results/rl_models/sac_lazy_phase_final.zip`
2. **Re-trains** the ARX baseline on 8,000 flat-terrain samples (takes ~5 s)
3. **Runs** 7 fan trajectories × 2 terrains × 3 methods = **42 PyBullet episodes**
4. **Prints** per-trajectory error table + summary
5. **Saves** 6 comparison figures to `results/`

### 8.2 Swapping which model is evaluated

Copy any checkpoint to the `sac_lazy_phase_final` path:

```bash
# Use the 230k mixed model (best overall)
cp results/rl_models/sac_mixed_230000_steps.zip \
   results/rl_models/sac_lazy_phase_final.zip

# Use the flat-terrain specialist
cp results/rl_models/sac_flat_150000_steps.zip \
   results/rl_models/sac_lazy_phase_final.zip
```

Then re-run `--eval`.

### 8.3 Reading the summary table

```
=================================================================
Results Summary -- Mean position error (m) over last 1 s
=================================================================
Method                         Flat terrain  Hills terrain
-----------------------------------------------------------------
No compensation (kin cmd)            0.0791         0.2600
ARX prediction (paper)               0.0283         0.0283
RL-SAC compensator (ours)            0.0862         0.2327
-----------------------------------------------------------------
RL improvement vs no-compensation (flat) : -9.0%
RL improvement vs no-compensation (3D)   : 10.5%
```

- **No compensation**: robot receives the desired velocity directly — motor lag causes deviation
- **ARX prediction**: paper's open-loop predictor (low error = good prediction of the lagged path)
- **RL-SAC**: closed-loop agent actively compensating for lag

---

## 9. Understanding the Output Files

### 9.1 Complete `results/` directory after running everything

```
results/
│
├── # Paper reproduction (main.py)
├── training_data.png
├── velocity_fit.png
├── residuals.png
├── model_order_study.png
├── fan_trajectories.png
├── trajectory_figure8.png
├── trajectory_square.png
├── trajectory_spiral.png
├── trajectory_slalom.png
├── trajectory_step_burst.png
├── trajectory_zigzag.png
│
├── # Terrain previews (terrain_generator.py)
├── terrain_2d_{flat,hills,ramps,mixed}.png
├── terrain_3d_{flat,hills,ramps,mixed}.png
│
├── # RL evaluation (rl_train_and_compare.py --eval)
├── rl_fan_flat.png           ← 7-trajectory fan, flat terrain
├── rl_fan_3d.png             ← 7-trajectory fan, hills terrain
├── rl_position_error.png     ← error vs time, both terrains
├── rl_velocity_tracking.png  ← velocity step response
├── rl_terrain_3d_flat.png    ← 3D surface + paths (flat)
├── rl_terrain_3d_hills.png   ← 3D surface + paths (hills)
│
├── # Simulation frames (rl_train_and_compare.py --frames)
├── rl_sim_frames_hills.png   ← 2×4 contact sheet
├── sim_frames/
│   ├── frame_000_t0.00s.png
│   └── ...frame_007_t2.99s.png
│
└── # RL models (rl_train_and_compare.py --train)
    rl_models/
    ├── sac_lazy_phase_final.zip      ← model used by --eval and --gui
    ├── sac_mixed_230000_steps.zip    ← best checkpoint (flat 0.054m, hills 0.339m)
    ├── sac_mixed_280000_steps.zip
    ├── sac_flat_150000_steps.zip
    ├── sac_flat_100000_steps.zip
    ├── sac_flat_50000_steps.zip
    └── best/                         ← EvalCallback best-ever model
```

### 9.2 Saved model format

Models are saved as `.zip` files by stable-baselines3. They contain:
- Actor/critic network weights (`policy.pth`)
- Replay buffer metadata
- Hyperparameters

Load a model at any time:

```python
from stable_baselines3 import SAC
model = SAC.load('./results/rl_models/sac_lazy_phase_final')
# model.predict(obs, deterministic=True) → action
```

---

## 10. Troubleshooting

### `KMP_DUPLICATE_LIB_OK` / OpenMP crash

```
OMP: Error #15: Initializing libiomp5md.dll...
```

**Fix:** Already handled automatically. If it still crashes:

```bash
set KMP_DUPLICATE_LIB_OK=TRUE   # Windows CMD
export KMP_DUPLICATE_LIB_OK=TRUE  # bash/zsh
python rl_train_and_compare.py --eval
```

---

### `UnicodeEncodeError` with Greek letters

```
UnicodeEncodeError: 'charmap' codec can't encode character '\u03c4'
```

**Fix:**

```bash
set PYTHONIOENCODING=utf-8   # Windows CMD
chcp 65001                   # change code page (Windows)
python main.py
```

---

### PyBullet GUI window does not open

Occurs in remote/headless environments (SSH without X11 forwarding).

**Fix:** Use `--frames` instead of `--gui` (off-screen rendering):

```bash
python rl_train_and_compare.py --frames --terrain hills
```

Or forward X11 if on Linux:
```bash
ssh -X user@host
python rl_train_and_compare.py --gui
```

---

### Robot falls through terrain / floats above it

This is the heightfield z-centering issue. It was fixed in `terrain_generator.py`:

```python
# terrain_generator.py line ~111
h_min = float(self.heights.min())
h_max = float(self.heights.max())
self._z_base = (h_min + h_max) * 0.5    # ← critical fix
body_id = p.createMultiBody(..., basePosition=[0.0, 0.0, self._z_base], ...)
```

If you modify the terrain and the robot floats, verify this formula is intact.

---

### `ModuleNotFoundError: No module named 'stable_baselines3'`

```bash
pip install stable-baselines3
```

### `ModuleNotFoundError: No module named 'gymnasium'`

```bash
pip install gymnasium
```

### Training is very slow (< 200 steps/sec)

This is usually caused by PyBullet being in GUI mode during training.
The training always uses `use_gui=False` — verify no `--gui` flag is passed
alongside `--train`. Check for background processes consuming CPU.

---

### `FileNotFoundError: robot_3d.urdf`

Make sure you are running the scripts from the project root:

```bash
cd "path/to/RL-Bot-main"
python rl_train_and_compare.py --gui
```

Do not run from a different directory. Alternatively, use an absolute path:

```bash
python "C:/Users/PFT2248-2/Documents/RL-Bot-main/RL-Bot-main/rl_train_and_compare.py" --gui
```

---

## 11. Module Map (for developers)

```
main.py
  └─ uses: robot_simulator.py
           data_generator.py
           arx_model.py
           trajectory_generator.py
           visualize.py

rl_train_and_compare.py
  └─ uses: rl_lazy_phase_env.py       ← Gymnasium MDP wrapper
             └─ terrain_generator.py  ← heightfield terrain
             └─ robot_3d.urdf         ← Pioneer P3-DX URDF
           arx_model.py               ← paper baseline
           robot_simulator.py         ← ARX warmup simulation
           data_generator.py          ← ARX training data
           trajectory_generator.py    ← ARX free-run prediction
           stable_baselines3.SAC      ← RL algorithm

web_visualizer.py
  └─ uses: robot_simulator.py
           arx_model.py
           trajectory_generator.py
           flask (HTTP server)
```

### Extending the RL environment

To add a new terrain type, edit `terrain_generator.py`:

```python
# In _generate_heights(), add a new elif branch:
elif terrain_type == 'steps':
    # create staircase terrain
    ...
```

Then use it anywhere:
```bash
python rl_train_and_compare.py --gui --terrain steps
python rl_train_and_compare.py --train --steps 300000
# (also update make_env() in rl_train_and_compare.py to include the new type)
```

To change the reference trajectory shape, add a new method to `LazyPhaseEnv3D`:

```python
# In rl_lazy_phase_env.py
def _ref_lemniscate(self) -> np.ndarray:
    """Bernoulli lemniscate (figure-8 on a curve)."""
    ...
```

And register it in `_make_reference()`:

```python
elif self.traj_type == 'lemniscate':
    return self._ref_lemniscate()
```

---

*Last updated: 2026-04-14*
*Python 3.11 | PyBullet 3.2.5 | stable-baselines3 2.8.0 | gymnasium 1.2.3 | torch 2.11.0*
