# RL-Bot

Python implementation of **Wheeled Mobile Robot (WMR) kinodynamic modeling via MIMO ARX System Identification**, reproducing the results of:

> Lee, Paulik, Krishnan — *"Wheeled Mobile Robot Modeling for Local Navigation Using System Identification"* — MWSCAS 2023

The project replaces the original Webots simulator with a pure-Python physics engine and fits a **MIMO ARX model** that learns the robot's dynamics from data.

---

## Overview

A differential-drive robot (Pioneer P3-DX) has finite motor acceleration — wheels do not instantly reach commanded speeds. This "lazy phase" creates memory effects that are captured by a higher-order ARX model. The pipeline:

1. **Physics simulator** (ground truth) — first-order motor lag + Euler pose integration
2. **Training data generation** — random piecewise-constant multi-step command sequences
3. **MIMO ARX fitting** — ridge-regularized least squares, asymmetric order configuration
4. **Validation** — one-step-ahead prediction fit% on held-out data
5. **Trajectory comparison** — free-run ARX simulation vs ground truth (figure-8, square, spiral, fan)

Expected performance (matching the paper):

| Channel | Fit% target | RMSE |
|---------|-------------|------|
| Linear velocity `v` | 85–95% | < 0.05 m/s |
| Angular velocity `ω` | 67–96% | < 0.08 rad/s |

---

## Project Structure

```
RL-Bot/
├── main.py                  # Full offline pipeline (trains, validates, saves figures)
├── web_visualizer.py        # Live browser animation at localhost:5000
├── robot_simulator.py       # Physics-based ground truth (Pioneer P3-DX)
├── data_generator.py        # Multi-step training/validation data generation
├── arx_model.py             # MIMO ARX system identification
├── trajectory_generator.py  # ARX free-run trajectory synthesis & comparison
├── visualize.py             # Matplotlib figure generation
├── requirements.txt
└── results/                 # Output figures (created on first run)
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full offline pipeline (saves figures to ./results/)
python main.py

# OR launch the interactive browser visualizer
python web_visualizer.py
# Open http://localhost:5000
```

---

## Web Visualizer

`web_visualizer.py` trains the ARX model, generates trajectory data, and serves an interactive Canvas-based animation.

**Features:**
- Real-time robot animation for figure-8, square, spiral, and fan trajectories
- Three overlaid paths: **Kinematic Ideal** (dashed), **Ground Truth** (cyan), **ARX Model** (orange)
- Robot icons pointing in their current heading direction
- Adjustable playback speed (1× – 80×)
- Live stats: simulation time, position error, heading error

**Controls:**

| Control | Action |
|---------|--------|
| Trajectory dropdown | Switch between trajectory patterns |
| ▶ Play / ⏸ Pause | Start/stop animation |
| ↺ Reset | Restart from the beginning |
| Speed slider | 1× to 80× playback speed |

---

## Robot Model

**Pioneer P3-DX differential-drive robot**

| Parameter | Value |
|-----------|-------|
| Wheel radius | 0.0975 m |
| Wheel base (2L) | 0.381 m |
| Max wheel speed | 1.5 m/s |
| Motor time constant τ | 0.1 s |
| Simulation timestep dt | 0.01 s |
| Motor pole p = 1 − dt/τ | 0.9 |

**Motor dynamics (the "lazy phase"):**
```
v_wheel(k) = p · v_wheel(k−1) + (1−p) · v_cmd(k)
```

This discrete low-pass filter creates transient response memory that motivates higher-order ARX models.

---

## ARX Model Configuration

The MIMO ARX model predicts `[v_actual, ω_actual]` from command history `[v_cmd, ω_cmd]`:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `na_diag` | 9 | Self-feedback AR order |
| `na_cross` | 3 | Cross-channel AR order |
| `nb_diag` | 9 | Self-channel exogenous order |
| `nb_cross` | 1 | Cross-channel exogenous order |
| `ridge_alpha` | 1e-8 | L2 regularization |

Order 9 captures 9 × 0.01 s = 90 ms of history, sufficient to cover the ~300 ms motor settling time.

---

## Results

Sample output figures saved to `./results/`:

| Figure | Description |
|--------|-------------|
| `training_data.png` | Command sequences and measured robot responses |
| `velocity_fit.png` | One-step-ahead predictions vs true velocities |
| `residuals.png` | Prediction error time series and histograms |
| `model_order_study.png` | Fit% vs ARX order (justifies order = 9) |
| `trajectory_figure8.png` | XY path comparison — figure-8 |
| `trajectory_square.png` | XY path comparison — square |
| `trajectory_spiral.png` | XY path comparison — spiral |
| `fan_trajectories.png` | Multiple 3-second fan trajectories (paper Fig. 3) |

---

## Dependencies

- `numpy` — numerical computation
- `scipy` — least-squares solver
- `matplotlib` — offline figure generation
- `statsmodels` — statistical utilities
- `flask` — web visualizer server

---

## Reference

```
@inproceedings{lee2023wmr,
  title     = {Wheeled Mobile Robot Modeling for Local Navigation Using System Identification},
  author    = {Lee, Paulik, Krishnan},
  booktitle = {MWSCAS 2023},
}
```
