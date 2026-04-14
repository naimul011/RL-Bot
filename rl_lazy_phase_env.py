"""
Gymnasium environment: lazy-phase compensation on 3D terrain.

The Lazy Phase (paper §III-A)
------------------------------
When a differential-drive robot is commanded to change its velocity, the
motors need finite time to respond.  The paper models this as a 1st-order
discrete lag:

    v_wheel(k) = p * v_wheel(k-1) + (1-p) * v_cmd(k)
    p = 1 - dt / tau     (pole;  tau = 0.1 s, dt = 0.01 s → p = 0.90)

This "lazy phase" causes the actual trajectory to deviate from the kinematic
ideal.  The ARX model in the paper accurately *predicts* this deviation but
does not compensate for it.

RL Contribution
---------------
An SAC agent observes the current tracking error and the wheel-velocity state
(which encodes how "lagged" the motors currently are) and learns to issue
*pre-emptive* commands that, after passing through the lag filter, deliver the
desired velocities — minimising path deviation.

3D Terrain Extension
---------------------
On a slope the drive wheels experience higher torque loads going uphill,
effectively *increasing* the motor time constant:

    tau_eff(slope) = tau_base * (1 + K_SLOPE * max(0, slope_fwd))

The ARX model from the paper was trained on flat ground and degrades on slopes
because it cannot adapt to this non-stationarity.  The RL agent observes the
local slope and adapts its pre-emption accordingly.

Observation space (17-D)
------------------------
 [0]    dist          – Euclidean distance to reference point
 [1-2]  (dx_l, dy_l) – position error in robot-local frame
 [3]    dtheta        – heading error, wrapped to (−π, π]
 [4-5]  (v, ω)        – actual body linear / angular velocity
 [6-7]  (vR, vL)      – wheel velocities  ← lag state
 [8-9]  (v_r, ω_r)    – reference velocities at current time step
 [10-11](δv, δω)      – velocity error  (actual − reference)
 [12-13](s_fwd,s_lat) – terrain slope in robot forward / lateral directions
 [14]   cross_track   – signed lateral deviation from reference path
 [15]   t_remain      – normalised episode time remaining  (1 → 0)
 [16]   t_ref         – normalised reference index  (0 → 1)

Action space (2-D, continuous, ∈ [−1, 1])
------------------------------------------
 [0] → v_cmd   = action[0] * V_MAX    (±1.0 m/s)
 [1] → ω_cmd   = action[1] * W_MAX    (±2.0 rad/s)
"""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p

from terrain_generator import TerrainGenerator

# ── robot constants (Pioneer P3-DX) ──────────────────────────────────────────
WHEEL_R   = 0.0975    # m
HALF_TRACK = 0.1905   # m  (= wheelbase / 2)
MAX_WHEEL_V = 1.5     # m/s  per wheel
MOTOR_TAU  = 0.1      # s  base time constant
DT         = 0.01     # s  simulation / control timestep
BASE_H     = 0.1175   # m  base-link centre above flat ground

# ── action limits ─────────────────────────────────────────────────────────────
V_MAX = 1.0           # m/s
W_MAX = 2.0           # rad/s

# ── slope effect on motor lag ─────────────────────────────────────────────────
K_SLOPE = 1.5         # tau_eff = tau * (1 + K_SLOPE * clip(slope_fwd, 0, ∞))


def _wrap(angle: float) -> float:
    """Wrap angle to (−π, π]."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def kinematic_integrate(
    v_cmds: np.ndarray,
    w_cmds: np.ndarray,
    dt: float = DT,
    x0: float = 0.0,
    y0: float = 0.0,
    th0: float = 0.0,
) -> np.ndarray:
    """
    Integrate kinematics with no motor lag → ideal reference trajectory.

    Returns
    -------
    ndarray of shape (N, 5): columns = [x, y, theta, v, omega]
    """
    N = len(v_cmds)
    traj = np.zeros((N, 5), dtype=np.float64)
    x, y, th = x0, y0, th0
    for k in range(N):
        v = float(v_cmds[k])
        w = float(w_cmds[k])
        x  += v * np.cos(th) * dt
        y  += v * np.sin(th) * dt
        th  = _wrap(th + w * dt)
        traj[k] = [x, y, th, v, w]
    return traj


class LazyPhaseEnv3D(gym.Env):
    """
    PyBullet-backed Gymnasium environment for lazy-phase trajectory tracking.

    The episode task: follow a kinematic reference trajectory as closely as
    possible despite the 1st-order motor lag (and, on 3D terrain, slope-
    dependent lag variation).
    """

    metadata = {'render_modes': ['rgb_array']}

    def __init__(
        self,
        terrain_type: str = 'flat',     # 'flat' | 'hills' | 'ramps' | 'mixed'
        use_gui: bool = False,
        max_steps: int = 3000,
        traj_type: str = 'random',      # 'random' | 'fan' | 'circle' | 'figure8'
        terrain_size: float = 20.0,
        terrain_res: int = 128,
        slope_effect: bool = True,
        seed: int = 42,
        # fan-trajectory parameters (used when traj_type='fan')
        fan_v: float = 0.30,
        fan_omega: float = 1.0,
        fan_steps: int = 300,
    ):
        super().__init__()

        self.terrain_type = terrain_type
        self.use_gui = use_gui
        self.max_steps = max_steps
        self.traj_type = traj_type
        self.slope_effect = slope_effect
        self.fan_v = fan_v
        self.fan_omega = fan_omega
        self.fan_steps = fan_steps

        self._seed = seed
        self._rng  = np.random.RandomState(seed)

        # Terrain helper (heights generated on first reset / explicit call)
        self._terrain = TerrainGenerator(
            size=terrain_size, resolution=terrain_res, seed=seed
        )

        # ── Observation space ─────────────────────────────────────────────
        low  = np.array(
            [-12, -12, -12, -np.pi,
             -2.0, -6.0, -MAX_WHEEL_V, -MAX_WHEEL_V,
             -V_MAX, -W_MAX, -2.0, -6.0,
             -1.0, -1.0, -6.0, 0.0, 0.0],
            dtype=np.float32,
        )
        high = np.array(
            [ 12,  12,  12,  np.pi,
              2.0,  6.0,  MAX_WHEEL_V,  MAX_WHEEL_V,
              V_MAX,  W_MAX,  2.0,  6.0,
              1.0,  1.0,  6.0, 1.0, 1.0],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # ── Action space ──────────────────────────────────────────────────
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # Runtime state — populated by reset()
        self._cid        = None
        self._robot_id   = None
        self._terrain_id = None
        self._vR         = 0.0
        self._vL         = 0.0
        self._step_count = 0
        self._ref_traj   = None   # (N, 5) float64  [x, y, theta, v, omega]
        self._initialized = False

    # ─────────────────────────────── Gym API ──────────────────────────────────

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.RandomState(seed)
            self._seed = seed

        if not self._initialized:
            self._init_pybullet()

        # Generate reference trajectory for this episode
        self._ref_traj = self._make_reference()

        # Starting pose
        x0   = float(self._ref_traj[0, 0])
        y0   = float(self._ref_traj[0, 1])
        th0  = float(self._ref_traj[0, 2])
        z0   = self._terrain.get_height_at(x0, y0) + BASE_H + 0.02

        quat = p.getQuaternionFromEuler([0.0, 0.0, th0])
        p.resetBasePositionAndOrientation(
            self._robot_id, [x0, y0, z0], quat, physicsClientId=self._cid
        )
        p.resetBaseVelocity(
            self._robot_id, [0, 0, 0], [0, 0, 0], physicsClientId=self._cid
        )
        # Zero wheel motors (release so robot settles under gravity)
        p.setJointMotorControlArray(
            self._robot_id, [0, 1], p.VELOCITY_CONTROL,
            targetVelocities=[0.0, 0.0],
            forces=[0.0, 0.0],
            physicsClientId=self._cid,
        )

        self._vR = 0.0
        self._vL = 0.0
        self._step_count = 0

        # Let the robot settle (tiny warm-up)
        for _ in range(8):
            p.stepSimulation(physicsClientId=self._cid)

        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray):
        # ── 1. Decode action ──────────────────────────────────────────────
        v_cmd = float(np.clip(action[0], -1.0, 1.0)) * V_MAX
        w_cmd = float(np.clip(action[1], -1.0, 1.0)) * W_MAX

        vR_cmd = float(np.clip(v_cmd + w_cmd * HALF_TRACK, -MAX_WHEEL_V, MAX_WHEEL_V))
        vL_cmd = float(np.clip(v_cmd - w_cmd * HALF_TRACK, -MAX_WHEEL_V, MAX_WHEEL_V))

        # ── 2. Slope-dependent motor lag ──────────────────────────────────
        pos, quat = p.getBasePositionAndOrientation(
            self._robot_id, physicsClientId=self._cid
        )
        _, _, yaw = p.getEulerFromQuaternion(quat)

        if self.slope_effect:
            sx, sy = self._terrain.get_slope_at(pos[0], pos[1])
            cos_y, sin_y = np.cos(yaw), np.sin(yaw)
            slope_fwd = float(np.clip(sx * cos_y + sy * sin_y, -0.5, 0.5))
            tau_eff = MOTOR_TAU * (1.0 + K_SLOPE * max(0.0, slope_fwd))
        else:
            tau_eff = MOTOR_TAU

        pole = float(np.clip(1.0 - DT / tau_eff, 0.05, 0.99))

        # ── 3. Apply first-order lag  (the "lazy phase") ──────────────────
        self._vR = float(np.clip(pole * self._vR + (1.0 - pole) * vR_cmd,
                                 -MAX_WHEEL_V, MAX_WHEEL_V))
        self._vL = float(np.clip(pole * self._vL + (1.0 - pole) * vL_cmd,
                                 -MAX_WHEEL_V, MAX_WHEEL_V))

        # ── 4. Drive wheels in PyBullet ───────────────────────────────────
        wL = self._vL / WHEEL_R
        wR = self._vR / WHEEL_R
        p.setJointMotorControlArray(
            self._robot_id, [0, 1], p.VELOCITY_CONTROL,
            targetVelocities=[wL, wR],
            forces=[10.0, 10.0],
            physicsClientId=self._cid,
        )
        p.stepSimulation(physicsClientId=self._cid)
        self._step_count += 1

        # ── 5. Camera follow (GUI only) ───────────────────────────────────
        if self.use_gui:
            self._follow_camera()

        # ── 6. Observations, reward, termination ─────────────────────────
        obs     = self._get_obs()
        reward  = self._compute_reward()

        pos2, quat2 = p.getBasePositionAndOrientation(
            self._robot_id, physicsClientId=self._cid
        )
        roll, pitch, _ = p.getEulerFromQuaternion(quat2)

        tipped     = abs(roll) > 0.9 or abs(pitch) > 0.9   # ~52°
        out_bounds = (abs(pos2[0]) > self._terrain.size * 0.45
                      or abs(pos2[1]) > self._terrain.size * 0.45)

        terminated = bool(tipped or out_bounds)
        truncated  = self._step_count >= self.max_steps

        info = {
            'step': self._step_count,
            'tipped': tipped,
            'out_of_bounds': out_bounds,
        }
        return obs, reward, terminated, truncated, info

    def get_robot_state(self) -> dict:
        """Return current PyBullet state as a plain dict (used by evaluator)."""
        pos, quat = p.getBasePositionAndOrientation(
            self._robot_id, physicsClientId=self._cid
        )
        vel, ang_vel = p.getBaseVelocity(
            self._robot_id, physicsClientId=self._cid
        )
        _, _, yaw = p.getEulerFromQuaternion(quat)
        v_act  = vel[0] * np.cos(yaw) + vel[1] * np.sin(yaw)
        w_act  = ang_vel[2]
        return {
            'x': float(pos[0]), 'y': float(pos[1]), 'z': float(pos[2]),
            'theta': float(yaw),
            'v': float(v_act), 'omega': float(w_act),
            'vR': self._vR, 'vL': self._vL,
        }

    def render(self, mode='rgb_array'):
        try:
            pos, _ = p.getBasePositionAndOrientation(
                self._robot_id, physicsClientId=self._cid
            )
            eye = [pos[0] - 1.8, pos[1] - 1.8, pos[2] + 1.8]
            view_mat = p.computeViewMatrix(
                cameraEyePosition=eye,
                cameraTargetPosition=list(pos),
                cameraUpVector=[0, 0, 1],
                physicsClientId=self._cid,
            )
            proj_mat = p.computeProjectionMatrixFOV(
                fov=60, aspect=4.0 / 3.0, nearVal=0.1, farVal=100.0,
                physicsClientId=self._cid,
            )
            _, _, rgb, _, _ = p.getCameraImage(
                640, 480,
                viewMatrix=view_mat,
                projectionMatrix=proj_mat,
                physicsClientId=self._cid,
            )
            frame = np.array(rgb, dtype=np.uint8).reshape(480, 640, 4)
            return frame[:, :, :3]
        except Exception:
            return np.zeros((480, 640, 3), dtype=np.uint8)

    def close(self):
        if self._cid is not None:
            try:
                p.disconnect(physicsClientId=self._cid)
            except Exception:
                pass
            self._cid = None
            self._initialized = False

    # ─────────────────────── Internal helpers ─────────────────────────────────

    def _init_pybullet(self):
        """Connect to PyBullet, load terrain and robot URDF."""
        if self._cid is not None:
            try:
                p.disconnect(physicsClientId=self._cid)
            except Exception:
                pass

        mode = p.GUI if self.use_gui else p.DIRECT
        self._cid = p.connect(mode)
        p.setAdditionalSearchPath(
            os.path.dirname(__file__), physicsClientId=self._cid
        )
        p.setTimeStep(DT, physicsClientId=self._cid)
        p.setGravity(0.0, 0.0, -9.81, physicsClientId=self._cid)

        if self.use_gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=4.0, cameraYaw=45, cameraPitch=-30,
                cameraTargetPosition=[0.0, 0.0, 0.3],
                physicsClientId=self._cid,
            )

        # ── Terrain ───────────────────────────────────────────────────────
        self._terrain_id, _ = self._terrain.create(self._cid, self.terrain_type)

        # ── Robot URDF ────────────────────────────────────────────────────
        here = os.path.dirname(__file__)
        urdf = os.path.join(here, 'robot_3d.urdf')
        if not os.path.exists(urdf):
            urdf = os.path.join(here, 'robot.urdf')

        self._robot_id = p.loadURDF(
            urdf,
            basePosition=[0.0, 0.0, BASE_H + 0.05],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=self._cid,
        )

        # ── Friction tuning ───────────────────────────────────────────────
        # Drive wheels: good grip
        for ji in [0, 1]:
            p.changeDynamics(
                self._robot_id, ji,
                lateralFriction=0.85,
                spinningFriction=0.05,
                rollingFriction=0.01,
                physicsClientId=self._cid,
            )
        # Casters: near-zero friction (slide freely)
        n_joints = p.getNumJoints(self._robot_id, physicsClientId=self._cid)
        for ji in range(2, n_joints):
            p.changeDynamics(
                self._robot_id, ji,
                lateralFriction=0.005,
                spinningFriction=0.0,
                rollingFriction=0.0,
                physicsClientId=self._cid,
            )
        # Base body damping
        p.changeDynamics(
            self._robot_id, -1,
            linearDamping=0.05,
            angularDamping=0.05,
            physicsClientId=self._cid,
        )

        self._initialized = True

    def _get_obs(self) -> np.ndarray:
        pos, quat = p.getBasePositionAndOrientation(
            self._robot_id, physicsClientId=self._cid
        )
        vel, ang_vel = p.getBaseVelocity(
            self._robot_id, physicsClientId=self._cid
        )
        _, _, yaw = p.getEulerFromQuaternion(quat)
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)

        v_act  = vel[0] * cos_y + vel[1] * sin_y
        w_act  = ang_vel[2]

        ref_idx = min(self._step_count, len(self._ref_traj) - 1)
        xr, yr, thr, vr, wr = self._ref_traj[ref_idx]

        # Position error in world frame
        dxw = xr - pos[0]
        dyw = yr - pos[1]
        dist = np.sqrt(dxw**2 + dyw**2)

        # Rotate into robot frame
        dx_l =  dxw * cos_y + dyw * sin_y
        dy_l = -dxw * sin_y + dyw * cos_y

        dtheta = _wrap(thr - yaw)
        dv     = v_act - vr
        dw     = w_act - wr

        # Terrain slope in robot frame
        sx, sy = self._terrain.get_slope_at(pos[0], pos[1])
        s_fwd =  sx * cos_y + sy * sin_y
        s_lat = -sx * sin_y + sy * cos_y

        # Signed cross-track error
        cos_r, sin_r = np.cos(thr), np.sin(thr)
        cross_track = -(pos[0] - xr) * sin_r + (pos[1] - yr) * cos_r

        t_remain = 1.0 - self._step_count / self.max_steps
        t_ref    = ref_idx / max(1, len(self._ref_traj) - 1)

        obs = np.array([
            np.clip(dist,       -12,  12),
            np.clip(dx_l,       -12,  12),
            np.clip(dy_l,       -12,  12),
            np.clip(dtheta,  -np.pi, np.pi),
            np.clip(v_act,     -2.0,  2.0),
            np.clip(w_act,     -6.0,  6.0),
            np.clip(self._vR,  -MAX_WHEEL_V, MAX_WHEEL_V),
            np.clip(self._vL,  -MAX_WHEEL_V, MAX_WHEEL_V),
            np.clip(vr,        -V_MAX, V_MAX),
            np.clip(wr,        -W_MAX, W_MAX),
            np.clip(dv,        -2.0,  2.0),
            np.clip(dw,        -6.0,  6.0),
            np.clip(s_fwd,     -1.0,  1.0),
            np.clip(s_lat,     -1.0,  1.0),
            np.clip(cross_track, -6.0, 6.0),
            np.clip(t_remain,   0.0,  1.0),
            np.clip(t_ref,      0.0,  1.0),
        ], dtype=np.float32)
        return obs

    def _compute_reward(self) -> float:
        pos, quat = p.getBasePositionAndOrientation(
            self._robot_id, physicsClientId=self._cid
        )
        vel, ang_vel = p.getBaseVelocity(
            self._robot_id, physicsClientId=self._cid
        )
        _, _, yaw = p.getEulerFromQuaternion(quat)
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        v_act = vel[0] * cos_y + vel[1] * sin_y
        w_act = ang_vel[2]

        ref_idx = min(self._step_count, len(self._ref_traj) - 1)
        xr, yr, thr, vr, wr = self._ref_traj[ref_idx]

        # Positional tracking error
        dist = np.sqrt((pos[0] - xr)**2 + (pos[1] - yr)**2)

        # Heading error
        h_err = abs(_wrap(thr - yaw))

        # Cross-track error (lateral deviation from reference path)
        cos_r, sin_r = np.cos(thr), np.sin(thr)
        ct_err = abs(-(pos[0] - xr) * sin_r + (pos[1] - yr) * cos_r)

        # Velocity tracking — the core lazy-phase metric
        v_err = abs(v_act - vr)
        w_err = abs(w_act - wr)

        reward = (
            -2.00 * dist       # penalise Euclidean position error
            - 1.50 * ct_err    # penalise lateral deviation
            - 0.40 * h_err     # penalise heading deviation
            - 0.80 * v_err     # penalise linear velocity lag
            - 0.40 * w_err     # penalise angular velocity lag
            + 0.05             # alive bonus
        )
        return float(reward)

    def _make_reference(self) -> np.ndarray:
        """Return kinematic reference trajectory array (N, 5)."""
        if self.traj_type == 'fan':
            return self._ref_fan(self.fan_v, self.fan_omega, self.fan_steps)
        elif self.traj_type == 'circle':
            return self._ref_circle()
        elif self.traj_type == 'figure8':
            return self._ref_figure8()
        else:
            return self._ref_random()

    # ── Reference trajectory generators ──────────────────────────────────────

    def _ref_random(self) -> np.ndarray:
        """Multi-step random velocity commands, kinematically integrated."""
        N   = self.max_steps
        rng = self._rng
        vc  = np.zeros(N)
        wc  = np.zeros(N)
        k   = 0
        while k < N:
            hold = rng.randint(50, 200)
            v    = rng.uniform(-0.5, 0.8)
            w    = rng.uniform(-1.5, 1.5)
            end  = min(k + hold, N)
            vc[k:end] = v
            wc[k:end] = w
            k = end
        return kinematic_integrate(vc, wc)

    def _ref_fan(self, v: float, omega: float, n: int) -> np.ndarray:
        """Constant v, step omega at t=0 — replicates paper Figure 3 scenario."""
        vc = np.full(n, v)
        wc = np.full(n, omega)
        return kinematic_integrate(vc, wc)

    def _ref_circle(self) -> np.ndarray:
        N  = self.max_steps
        vc = np.full(N, 0.40)
        wc = np.full(N, 0.55)
        return kinematic_integrate(vc, wc)

    def _ref_figure8(self) -> np.ndarray:
        N  = self.max_steps
        vc = np.full(N, 0.40)
        wc = np.zeros(N)
        h  = N // 2
        wc[:h] =  0.80
        wc[h:] = -0.80
        return kinematic_integrate(vc, wc)

    # ── GUI helper ────────────────────────────────────────────────────────────

    def _follow_camera(self):
        try:
            pos, _ = p.getBasePositionAndOrientation(
                self._robot_id, physicsClientId=self._cid
            )
            p.resetDebugVisualizerCamera(
                cameraDistance=3.0, cameraYaw=50, cameraPitch=-28,
                cameraTargetPosition=[pos[0], pos[1], pos[2] + 0.1],
                physicsClientId=self._cid,
            )
        except Exception:
            pass
