# REPORT2 - Post-Training Results Update

Date: 2026-04-16

## 1. Training Status

Training is complete.

- Command used:
  - `python rl_train_and_compare.py --train --steps 600000`
- Phases completed:
  - Phase 1 (flat): 360,000 steps
  - Phase 2 (mixed): 240,000 steps
- Final model checkpoint:
  - `results/rl_models/sac_lazy_phase_final.zip`

## 2. Evaluation Run

Evaluation command:

- `python rl_train_and_compare.py --eval`

ARX fit during evaluation:

- v fit: 98.8%
- w fit: 98.8%

## 3. Updated Summary Metrics

Mean position error over the last 1 s:

| Method | Flat terrain (m) | Hills terrain (m) |
|---|---:|---:|
| No compensation (kin cmd) | 0.0790 | 0.2600 |
| ARX prediction (paper) | 0.0283 | 0.0283 |
| RL-SAC compensator (ours) | 0.0521 | 0.3705 |

Computed RL change vs no compensation:

- Flat: +34.0% improvement
- Hills (3D): -42.5% regression

## 4. Per-Trajectory RL Errors (from latest eval)

Flat terrain RL errors:

| omega (rad/s) | RL error (m) |
|---:|---:|
| -1.8 | 0.1355 |
| -1.2 | 0.0442 |
| -0.6 | 0.0161 |
| -0.0 | 0.0350 |
| +0.6 | 0.0195 |
| +1.2 | 0.0544 |
| +1.8 | 0.0601 |

Hills terrain RL errors:

| omega (rad/s) | RL error (m) |
|---:|---:|
| -1.8 | 0.6200 |
| -1.2 | 0.2550 |
| -0.6 | 0.0372 |
| -0.0 | 0.0353 |
| +0.6 | 0.0716 |
| +1.2 | 0.5349 |
| +1.8 | 1.0391 |

## 5. Figures Generated

Latest plots are available in:

- `results/rl_training_curve.png`
- `results/rl_fan_flat.png`
- `results/rl_fan_3d.png`
- `results/rl_position_error.png`
- `results/rl_velocity_tracking.png`
- `results/rl_terrain_3d_flat.png`
- `results/rl_terrain_3d_hills.png`

## 6. Conclusion from This Run

The 600k-step model improved flat-terrain tracking significantly, but degraded badly on hills terrain in this evaluation. The current best practical model for 3D terrain tracking may still be an earlier checkpoint, so checkpoint selection by terrain-specific validation is recommended before replacing the deployed model.
