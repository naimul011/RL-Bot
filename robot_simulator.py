"""
Physics-based differential-drive robot simulator.

Serves as the ground-truth plant, replacing Webots in the paper.
The key physics is motor dynamics: finite acceleration limits produce
the "lazy phase" that makes ARX modeling non-trivial.

Based on: Lee, Paulik, Krishnan, MWSCAS 2023
"""

import numpy as np


class DifferentialDriveRobot:
    """
    High-fidelity differential-drive robot simulation.

    Physical parameters default to Pioneer P3-DX specs.
    Motor dynamics are modeled as a first-order lag (low-pass filter) with
    a speed saturation limit. This introduces the "lazy phase" memory that
    motivates higher-order ARX models.

    The first-order lag gives:
        v_wheel(k) = (1 - dt/tau) * v_wheel(k-1) + (dt/tau) * v_wheel_cmd(k)

    At dt=0.01s, tau=0.1s: pole = 0.9 → ~0.3s settling time (3 time constants).
    Order-9 ARX captures 9 * 0.01s = 0.09s, sufficient for the transient dynamics.
    This matches the paper's sampling interval (0.01s) and order selection (9).
    """

    def __init__(
        self,
        wheel_radius: float = 0.0975,   # meters
        wheel_base: float = 0.381,       # 2L, meters between wheel centers
        max_wheel_speed: float = 1.5,    # m/s per wheel
        motor_time_constant: float = 0.1,  # seconds (1st-order lag time constant)
        dt: float = 0.01,               # simulation timestep (paper uses 0.01s)
    ):
        self.r = wheel_radius
        self.L = wheel_base / 2.0        # half-track width
        self.max_wheel_speed = max_wheel_speed
        self.tau = motor_time_constant
        self.dt = dt

        # First-order lag pole: p = 1 - dt/tau
        self._pole = max(0.0, 1.0 - dt / motor_time_constant)

        # State
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.vR = 0.0   # right wheel velocity (m/s)
        self.vL = 0.0   # left wheel velocity (m/s)

    def reset(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.theta = theta
        self.vR = 0.0
        self.vL = 0.0

    def _apply_motor_dynamics(self, v_cmd: float, v_current: float) -> float:
        """
        First-order lag motor model (the 'lazy phase').

        v_wheel(k) = p * v_wheel(k-1) + (1-p) * v_cmd(k)

        With p = 1 - dt/tau, this implements a discrete-time RC-like low-pass
        filter. DC gain = 1.0 (steady-state v_actual == v_cmd), ensuring the
        ARX model has the correct steady-state behavior in free-run mode.
        """
        p = self._pole
        v_new = p * v_current + (1.0 - p) * v_cmd
        return np.clip(v_new, -self.max_wheel_speed, self.max_wheel_speed)

    def step(self, v_cmd: float, omega_cmd: float) -> dict:
        """
        Advance simulation by one timestep.

        Args:
            v_cmd:     commanded linear velocity (m/s)
            omega_cmd: commanded angular velocity (rad/s)

        Returns:
            State dict with actual velocities and pose.
        """
        # Inverse kinematics: body velocity -> wheel velocity commands
        vR_cmd = v_cmd + omega_cmd * self.L
        vL_cmd = v_cmd - omega_cmd * self.L

        # Apply motor dynamics (finite acceleration)
        self.vR = self._apply_motor_dynamics(vR_cmd, self.vR)
        self.vL = self._apply_motor_dynamics(vL_cmd, self.vL)

        # Forward kinematics: wheel velocities -> body velocities
        v_actual = (self.vR + self.vL) / 2.0
        omega_actual = (self.vR - self.vL) / (2.0 * self.L)

        # Integrate pose (Euler integration)
        self.x += v_actual * np.cos(self.theta) * self.dt
        self.y += v_actual * np.sin(self.theta) * self.dt
        self.theta += omega_actual * self.dt
        self.theta = _wrap_to_pi(self.theta)

        return {
            'x': self.x,
            'y': self.y,
            'theta': self.theta,
            'v': v_actual,
            'omega': omega_actual,
            'vR': self.vR,
            'vL': self.vL,
        }

    def simulate(self, v_cmds: np.ndarray, omega_cmds: np.ndarray) -> dict:
        """
        Batch simulate over a full command sequence.

        Args:
            v_cmds:     shape (N,) linear velocity commands
            omega_cmds: shape (N,) angular velocity commands

        Returns:
            Dict of arrays, each shape (N,), with keys:
            x, y, theta, v, omega, vR, vL
        """
        N = len(v_cmds)
        assert len(omega_cmds) == N

        self.reset()
        history = {k: np.zeros(N) for k in ('x', 'y', 'theta', 'v', 'omega', 'vR', 'vL')}

        for k in range(N):
            state = self.step(v_cmds[k], omega_cmds[k])
            for key in history:
                history[key][k] = state[key]

        return history


def _wrap_to_pi(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi
