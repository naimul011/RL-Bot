"""
PyBullet-based differential-drive robot simulator.

Implements the same external interface as `DifferentialDriveRobot` in
`robot_simulator.py` so existing data generation and ARX pipeline code can
switch backends without further changes.
"""

import numpy as np

try:
    import pybullet as p
except ImportError as exc:
    raise ImportError(
        "pybullet is required for PyBulletDifferentialDriveRobot. "
        "Install with: pip install pybullet"
    ) from exc


class PyBulletDifferentialDriveRobot:
    """
    Differential-drive simulator backed by PyBullet rigid-body dynamics.

    We keep motor lag behavior identical to the analytic simulator:
        v_wheel(k) = p * v_wheel(k-1) + (1-p) * v_cmd(k)
    Then map wheel speeds to body linear/angular velocity and apply them in
    PyBullet using base velocity control.
    """

    def __init__(
        self,
        wheel_radius: float = 0.0975,
        wheel_base: float = 0.381,
        max_wheel_speed: float = 1.5,
        motor_time_constant: float = 0.1,
        dt: float = 0.01,
        use_gui: bool = False,
    ):
        self.r = wheel_radius
        self.L = wheel_base / 2.0
        self.max_wheel_speed = max_wheel_speed
        self.tau = motor_time_constant
        self.dt = dt
        self._pole = max(0.0, 1.0 - dt / motor_time_constant)

        self._cid = p.connect(p.GUI if use_gui else p.DIRECT)
        p.setTimeStep(self.dt, physicsClientId=self._cid)
        p.setGravity(0.0, 0.0, -9.81, physicsClientId=self._cid)

        # Ground plane with low friction to avoid introducing unintended slip drag.
        self._plane_id = p.createCollisionShape(
            p.GEOM_PLANE, physicsClientId=self._cid
        )
        self._plane_body = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=self._plane_id,
            physicsClientId=self._cid,
        )
        
        # Load the differential-drive robot from URDF
        import os
        urdf_path = os.path.join(os.path.dirname(__file__), "robot.urdf")
        self._body_id = p.loadURDF(
            urdf_path,
            basePosition=[0.0, 0.0, 0.1175],  # 0.0975 + 0.02
            baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, 0.0]),
            physicsClientId=self._cid,
        )

        # Constrain roll/pitch so platform behaves like planar mobile robot.
        p.changeDynamics(
            self._body_id,
            -1,
            linearDamping=0.0,
            angularDamping=0.0,
            lateralFriction=0.8,
            physicsClientId=self._cid,
        )
        p.changeDynamics(
            self._body_id,
            0,
            lateralFriction=1.0,
            physicsClientId=self._cid,
        )
        p.changeDynamics(
            self._body_id,
            1,
            lateralFriction=1.0,
            physicsClientId=self._cid,
        )

        self.vR = 0.0
        self.vL = 0.0

    def __del__(self):
        if hasattr(self, "_cid") and self._cid is not None:
            try:
                p.disconnect(physicsClientId=self._cid)
            except Exception:
                pass

    def reset(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0) -> None:
        self.vR = 0.0
        self.vL = 0.0
        quat = p.getQuaternionFromEuler([0.0, 0.0, theta])
        p.resetBasePositionAndOrientation(
            self._body_id,
            [x, y, 0.1175],
            quat,
            physicsClientId=self._cid,
        )
        p.resetBaseVelocity(
            self._body_id,
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            physicsClientId=self._cid,
        )
        p.setJointMotorControlArray(
            self._body_id,
            [0, 1],
            p.VELOCITY_CONTROL,
            targetVelocities=[0.0, 0.0],
            forces=[0.0, 0.0],
            physicsClientId=self._cid,
        )

    def _apply_motor_dynamics(self, v_cmd: float, v_current: float) -> float:
        pcoef = self._pole
        v_new = pcoef * v_current + (1.0 - pcoef) * v_cmd
        return float(np.clip(v_new, -self.max_wheel_speed, self.max_wheel_speed))

    def step(self, v_cmd: float, omega_cmd: float) -> dict:
        vR_cmd = v_cmd + omega_cmd * self.L
        vL_cmd = v_cmd - omega_cmd * self.L

        # Desired angular velocity for the wheels
        wR_cmd = vR_cmd / self.r
        wL_cmd = vL_cmd / self.r

        # The max force/torque is related to the time constant in some way.
        # But we also have `_apply_motor_dynamics`, so we can just use the calculated
        # vR and vL and force PyBullet wheels to those velocities to exactly match
        # the motor dynamics from the paper/analytic model.

        self.vR = self._apply_motor_dynamics(vR_cmd, self.vR)
        self.vL = self._apply_motor_dynamics(vL_cmd, self.vL)

        target_wL = self.vL / self.r
        target_wR = self.vR / self.r

        p.setJointMotorControlArray(
            self._body_id,
            [0, 1],
            p.VELOCITY_CONTROL,
            targetVelocities=[target_wL, target_wR],
            forces=[10.0, 10.0],
            physicsClientId=self._cid,
        )

        p.stepSimulation(physicsClientId=self._cid)

        # Retrieve actual base state from the physics engine
        pos, quat = p.getBasePositionAndOrientation(
            self._body_id, physicsClientId=self._cid
        )
        vel, ang_vel = p.getBaseVelocity(
            self._body_id, physicsClientId=self._cid
        )
        _, _, yaw = p.getEulerFromQuaternion(quat)

        # Compute actual v and omega from the body
        v_actual = vel[0] * np.cos(yaw) + vel[1] * np.sin(yaw)
        omega_actual = ang_vel[2]

        return {
            'x': float(pos[0]),
            'y': float(pos[1]),
            'theta': float(_wrap_to_pi(yaw)),
            'v': float(v_actual),
            'omega': float(omega_actual),
            'vR': float(self.vR),
            'vL': float(self.vL),
        }

    def simulate(self, v_cmds: np.ndarray, omega_cmds: np.ndarray) -> dict:
        n = len(v_cmds)
        assert len(omega_cmds) == n

        self.reset()
        history = {k: np.zeros(n) for k in ('x', 'y', 'theta', 'v', 'omega', 'vR', 'vL')}

        for k in range(n):
            s = self.step(float(v_cmds[k]), float(omega_cmds[k]))
            for key in history:
                history[key][k] = s[key]

        return history


def _wrap_to_pi(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi
