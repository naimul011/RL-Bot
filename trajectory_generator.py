"""
Trajectory generation using the fitted ARX model.

Integrates the ARX model's velocity predictions with kinematic equations
to produce (x, y, theta) trajectories. Warmup steps use the physics
simulator to initialize the ARX history buffers from a valid state.

Based on: Lee, Paulik, Krishnan, MWSCAS 2023 - Section III-A (eq. 4)
"""

import numpy as np
from arx_model import MIMOARXModel
from robot_simulator import DifferentialDriveRobot, _wrap_to_pi


class TrajectoryGenerator:
    """
    Generate robot trajectories using an ARX model for velocity prediction.

    The ARX model predicts (v_actual, omega_actual) at each step; these are
    then integrated using the kinematic equations to obtain the robot's pose.
    """

    def __init__(self, arx_model: MIMOARXModel, dt: float = 0.1):
        assert arx_model._is_fitted, "ARX model must be fitted before use."
        self.model = arx_model
        self.dt = dt
        self.max_lag = arx_model.max_lag

    def generate(
        self,
        v_cmds: np.ndarray,
        omega_cmds: np.ndarray,
        warmup_robot: DifferentialDriveRobot = None,
        x0: float = 0.0,
        y0: float = 0.0,
        theta0: float = 0.0,
    ) -> dict:
        """
        Generate a trajectory from a sequence of velocity commands.

        Phase 1 (warmup): run the physics simulator for max_lag steps to
        build a valid initial history buffer for the ARX model.

        Phase 2 (ARX free-run): for each remaining step, predict velocities
        with the ARX model (using own past predictions), then integrate pose.

        Args:
            v_cmds:        shape (N,) linear velocity commands (m/s)
            omega_cmds:    shape (N,) angular velocity commands (rad/s)
            warmup_robot:  physics simulator for warmup phase
            x0, y0, theta0: initial pose

        Returns:
            dict with arrays: x, y, theta, v_hat, omega_hat, t
        """
        N = len(v_cmds)
        assert len(omega_cmds) == N

        U = np.column_stack([v_cmds, omega_cmds])

        x = np.zeros(N)
        y = np.zeros(N)
        theta = np.zeros(N)
        v_hat_arr = np.zeros(N)
        omega_hat_arr = np.zeros(N)

        # --- Phase 1: Warmup with physics simulator ---
        if warmup_robot is not None:
            warmup_robot.reset(x0, y0, theta0)
            Y_buf = np.zeros((self.max_lag, 2))
            U_buf = np.zeros((self.max_lag, 2))

            for k in range(min(self.max_lag, N)):
                state = warmup_robot.step(v_cmds[k], omega_cmds[k])
                x[k] = state['x']
                y[k] = state['y']
                theta[k] = state['theta']
                Y_buf[k, 0] = state['v']
                Y_buf[k, 1] = state['omega']
                U_buf[k, 0] = v_cmds[k]
                U_buf[k, 1] = omega_cmds[k]
                v_hat_arr[k] = state['v']
                omega_hat_arr[k] = state['omega']

            cur_x = warmup_robot.x
            cur_y = warmup_robot.y
            cur_theta = warmup_robot.theta
            start_k = self.max_lag

        else:
            # No warmup robot: start from zero history, integrate from given pose
            Y_buf = np.zeros((self.max_lag, 2))
            U_buf = np.zeros((self.max_lag, 2))
            cur_x, cur_y, cur_theta = x0, y0, theta0
            start_k = self.max_lag

        # --- Phase 2: ARX free-run simulation ---
        for k in range(start_k, N):
            # Predict velocities using ARX model
            y_hat = self.model._predict_from_buffer(Y_buf, U_buf)
            v_pred = y_hat[0]
            omega_pred = y_hat[1]

            # Kinematic integration (eq. 4 in paper)
            cur_x += v_pred * np.cos(cur_theta) * self.dt
            cur_y += v_pred * np.sin(cur_theta) * self.dt
            cur_theta = _wrap_to_pi(cur_theta + omega_pred * self.dt)

            x[k] = cur_x
            y[k] = cur_y
            theta[k] = cur_theta
            v_hat_arr[k] = v_pred
            omega_hat_arr[k] = omega_pred

            # Shift history buffers (use predicted velocities)
            Y_buf = np.roll(Y_buf, -1, axis=0)
            Y_buf[-1, 0] = v_pred
            Y_buf[-1, 1] = omega_pred
            U_buf = np.roll(U_buf, -1, axis=0)
            U_buf[-1] = U[k]

        t = np.arange(N) * self.dt

        return {
            'x': x,
            'y': y,
            'theta': theta,
            'v_hat': v_hat_arr,
            'omega_hat': omega_hat_arr,
            't': t,
        }

    def compare_with_ground_truth(
        self,
        robot: DifferentialDriveRobot,
        v_cmds: np.ndarray,
        omega_cmds: np.ndarray,
        x0: float = 0.0,
        y0: float = 0.0,
        theta0: float = 0.0,
    ) -> dict:
        """
        Run both the physics simulator and ARX trajectory generator on the
        same command sequence, then compare the resulting paths.

        Also computes a purely kinematic (ideal) trajectory that assumes
        instantaneous velocity response (no dynamics), matching the paper's
        'Ideal Path' reference.

        Returns:
            dict with 'ground_truth', 'arx_model', 'kinematic', and error arrays
        """
        # Ground truth: full physics simulation
        robot.reset(x0, y0, theta0)
        gt = robot.simulate(v_cmds, omega_cmds)
        gt['t'] = np.arange(len(v_cmds)) * self.dt

        # ARX trajectory (warmup with same robot, then free-run)
        robot.reset(x0, y0, theta0)
        arx = self.generate(v_cmds, omega_cmds, warmup_robot=robot, x0=x0, y0=y0, theta0=theta0)

        # Kinematic (ideal): integrate commands directly, no dynamics
        kinematic = _kinematic_trajectory(v_cmds, omega_cmds, self.dt, x0, y0, theta0)

        # Position error: Euclidean distance between ARX and ground truth
        pos_error = np.sqrt((gt['x'] - arx['x']) ** 2 + (gt['y'] - arx['y']) ** 2)

        # Heading error (wrapped to [-pi, pi])
        heading_error = np.abs(np.array([_wrap_to_pi(a - b) for a, b in zip(gt['theta'], arx['theta'])]))

        return {
            'ground_truth': gt,
            'arx_model': arx,
            'kinematic': kinematic,
            'position_error': pos_error,
            'heading_error': heading_error,
        }


def _kinematic_trajectory(
    v_cmds: np.ndarray,
    omega_cmds: np.ndarray,
    dt: float,
    x0: float = 0.0,
    y0: float = 0.0,
    theta0: float = 0.0,
) -> dict:
    """
    Pure kinematic trajectory: integrate commands directly (no motor dynamics).

    This is the 'Ideal Path' reference in the paper — it assumes velocities
    are achieved instantaneously (infinite motor bandwidth).
    """
    N = len(v_cmds)
    x = np.zeros(N)
    y = np.zeros(N)
    theta = np.zeros(N)

    cx, cy, ct = x0, y0, theta0
    for k in range(N):
        cx += v_cmds[k] * np.cos(ct) * dt
        cy += v_cmds[k] * np.sin(ct) * dt
        ct = _wrap_to_pi(ct + omega_cmds[k] * dt)
        x[k] = cx
        y[k] = cy
        theta[k] = ct

    return {'x': x, 'y': y, 'theta': theta, 't': np.arange(N) * dt}
