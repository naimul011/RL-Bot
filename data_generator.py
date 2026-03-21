"""
Training and validation data generation for WMR system identification.

Generates random piecewise-constant (multi-step) command sequences and
runs the physics simulator to produce ground-truth input/output datasets.

Based on: Lee, Paulik, Krishnan, MWSCAS 2023 - Section III & IV
"""

import numpy as np
from robot_simulator import DifferentialDriveRobot


class DataGenerator:
    """
    Generate SI training/validation datasets using random multi-step sequences.

    The multi-step (piecewise-constant) commands ensure that the system is
    persistently excited across a wide frequency range, which is necessary
    for reliable ARX parameter estimation.
    """

    def __init__(
        self,
        robot: DifferentialDriveRobot,
        rng_seed: int = 42,
    ):
        self.robot = robot
        self.rng = np.random.default_rng(rng_seed)

    def _generate_multistep_sequence(
        self,
        n_samples: int,
        v_range: tuple = (-1.0, 1.0),
        omega_range: tuple = (-1.5, 1.5),
        min_hold_steps: int = 10,
        max_hold_steps: int = 40,
    ) -> tuple:
        """
        Generate a piecewise-constant (multi-step) command sequence.

        Each step randomly selects a (v, omega) pair and holds it for
        a random duration. This ensures transient responses are frequently
        excited, which is critical for learning dynamics.

        Returns:
            (v_cmds, omega_cmds): each shape (n_samples,)
        """
        v_cmds = np.zeros(n_samples)
        omega_cmds = np.zeros(n_samples)

        k = 0
        while k < n_samples:
            hold = int(self.rng.integers(min_hold_steps, max_hold_steps + 1))
            end = min(k + hold, n_samples)
            v_cmd = self.rng.uniform(*v_range)
            omega_cmd = self.rng.uniform(*omega_range)
            v_cmds[k:end] = v_cmd
            omega_cmds[k:end] = omega_cmd
            k = end

        return v_cmds, omega_cmds

    def generate_dataset(
        self,
        n_samples: int = 5000,
        split: float = 0.7,
        v_range: tuple = (-1.0, 1.0),
        omega_range: tuple = (-1.5, 1.5),
        min_hold_steps: int = 10,
        max_hold_steps: int = 40,
    ) -> dict:
        """
        Generate a complete dataset and split into train/validation.

        Args:
            n_samples:      total number of timesteps
            split:          fraction used for training (rest for validation)
            v_range:        (min, max) for linear velocity commands (m/s)
            omega_range:    (min, max) for angular velocity commands (rad/s)
            min_hold_steps: minimum steps per constant command segment
            max_hold_steps: maximum steps per constant command segment

        Returns:
            dict with keys 'train', 'val', 'full', each containing:
                'u': shape (N, 2)  — [v_in, omega_in]
                'y': shape (N, 2)  — [v_out, omega_out]
                't': shape (N,)    — time vector
        """
        v_cmds, omega_cmds = self._generate_multistep_sequence(
            n_samples, v_range, omega_range, min_hold_steps, max_hold_steps
        )

        self.robot.reset()
        sim = self.robot.simulate(v_cmds, omega_cmds)

        t = np.arange(n_samples) * self.robot.dt
        u = np.column_stack([v_cmds, omega_cmds])
        y = np.column_stack([sim['v'], sim['omega']])

        split_idx = int(n_samples * split)

        def _slice(a, start, end):
            return a[start:end]

        return {
            'train': {
                'u': _slice(u, 0, split_idx),
                'y': _slice(y, 0, split_idx),
                't': _slice(t, 0, split_idx),
            },
            'val': {
                'u': _slice(u, split_idx, n_samples),
                'y': _slice(y, split_idx, n_samples),
                't': _slice(t, split_idx, n_samples),
            },
            'full': {'u': u, 'y': y, 't': t},
        }

    def generate_trajectory_commands(
        self,
        trajectory_type: str = 'figure8',
        n_samples: int = 500,
        dt: float = None,
    ) -> tuple:
        """
        Generate structured command sequences for trajectory evaluation.

        Args:
            trajectory_type: 'figure8', 'square', 'spiral', 'straight', 'random'
            n_samples:       number of timesteps
            dt:              timestep (defaults to robot's dt)

        Returns:
            (v_cmds, omega_cmds): each shape (n_samples,)
        """
        if dt is None:
            dt = self.robot.dt

        t = np.arange(n_samples) * dt

        if trajectory_type == 'figure8':
            v_cmds = np.full(n_samples, 0.5)
            omega_cmds = 0.8 * np.sin(2 * np.pi * t / (n_samples * dt / 2))

        elif trajectory_type == 'square':
            v_cmds = np.full(n_samples, 0.4)
            omega_cmds = np.zeros(n_samples)
            seg = n_samples // 8
            for i in range(8):
                start = i * seg
                end = min(start + seg, n_samples)
                if i % 2 == 1:  # turn segments
                    omega_cmds[start:end] = 1.0

        elif trajectory_type == 'spiral':
            v_cmds = np.linspace(0.1, 1.0, n_samples)
            omega_cmds = np.full(n_samples, 0.5)

        elif trajectory_type == 'straight':
            v_cmds = np.full(n_samples, 0.6)
            omega_cmds = np.zeros(n_samples)

        elif trajectory_type == 'random':
            v_cmds, omega_cmds = self._generate_multistep_sequence(n_samples)

        else:
            raise ValueError(f"Unknown trajectory_type: {trajectory_type}")

        return v_cmds, omega_cmds
