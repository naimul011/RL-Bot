"""
Additional system identification models for comparison with ARX.

All models implement the same interface as MIMOARXModel:
  - fit(Y, U)
  - _predict_from_buffer(Y_buf, U_buf)   <- used by TrajectoryGenerator
  - predict(Y, U, mode)
  - score(Y_true, Y_hat)

Models:
  MIMOFIRModel   — Finite Impulse Response (feed-forward only, na=0)
  MIMOARMAXModel — ARMAX via iterative pseudo-linear regression (PLR)
  MIMOOEModel    — Output Error (simulation-consistent training)

Overflow safety
---------------
Both ARMAX and OE use iterative fitting where the simulated trajectory can
diverge in early iterations.  Three defences are applied:
  1. Phi sanity check before every solve — skip the step if Phi is not finite.
  2. Y_oe clipping to ±Y_CLIP after each OE simulation step.
  3. _predict_from_buffer clamps its output so NaN/Inf cannot propagate into
     the TrajectoryGenerator's rolling buffer.
"""

import numpy as np
from arx_model import MIMOARXModel

# Physical velocity bound used for clipping (m/s or rad/s, both ≤ this)
_Y_CLIP = 5.0


# ── FIR Model ─────────────────────────────────────────────────────────────────

class MIMOFIRModel(MIMOARXModel):
    """
    Finite Impulse Response model: y depends only on past inputs, no AR feedback.

    Equivalent to ARX with na_diag=na_cross=0.  Serves as a baseline showing
    what is lost when the model has no memory of its own past outputs.
    """

    def __init__(self, nb_diag: int = 9, nb_cross: int = 1, ridge_alpha: float = 1e-8):
        super().__init__(
            na_diag=0, na_cross=0,
            nb_diag=nb_diag, nb_cross=nb_cross,
            ridge_alpha=ridge_alpha,
        )

    def _predict_from_buffer(self, Y_buf: np.ndarray, U_buf: np.ndarray) -> np.ndarray:
        """FIR prediction: only past inputs matter, no AR terms."""
        phi_0 = np.concatenate([
            U_buf[-self.nb_diag:, 0][::-1],
            U_buf[-self.nb_cross:, 1][::-1],
        ])
        phi_1 = np.concatenate([
            U_buf[-self.nb_cross:, 0][::-1],
            U_buf[-self.nb_diag:, 1][::-1],
        ])
        return np.clip(
            np.nan_to_num(np.array([phi_0 @ self.theta_0_, phi_1 @ self.theta_1_])),
            -_Y_CLIP, _Y_CLIP,
        )


# ── ARMAX Model ───────────────────────────────────────────────────────────────

class MIMOARMAXModel(MIMOARXModel):
    """
    MIMO ARMAX model fitted by iterative pseudo-linear regression (PLR).

    Model:  A(q) y(k) = B(q) u(k) + C(q) e(k)

    The moving-average polynomial C captures correlated noise / disturbances.
    Training: alternate between estimating theta and updating the noise estimate
    e_hat(k) = y(k) - y_hat(k).  Converges in 2–4 iterations for most systems.

    In simulation (free-run) mode, MA terms are set to zero (no noise realisation
    is available), so the trajectory behaviour reverts to the ARX-like part.
    """

    def __init__(
        self,
        na_diag: int = 9, na_cross: int = 3,
        nb_diag: int = 9, nb_cross: int = 1,
        nc: int = 3,
        n_iter: int = 3,
        ridge_alpha: float = 1e-8,
    ):
        super().__init__(na_diag, na_cross, nb_diag, nb_cross, ridge_alpha)
        self.nc = nc
        self.n_iter = n_iter
        self.max_lag = max(na_diag, nb_diag, nc)

    def _build_armax_regressors(self, Y: np.ndarray, U: np.ndarray, Eps: np.ndarray):
        """
        Build augmented regressor: [phi_arx | phi_ma].
        phi_ma contains nc past residuals for each output channel independently.
        """
        N = len(Y)
        N_valid = N - self.max_lag
        assert N_valid > 0

        n0 = self.na_diag + self.na_cross + self.nb_diag + self.nb_cross + self.nc
        n1 = self.na_cross + self.na_diag + self.nb_cross + self.nb_diag + self.nc

        Phi_0 = np.zeros((N_valid, n0))
        Phi_1 = np.zeros((N_valid, n1))

        for i, k in enumerate(range(self.max_lag, N)):
            # ── channel 0 (v) ──
            c = 0
            if self.na_diag:
                Phi_0[i, c:c+self.na_diag]  = Y[k-self.na_diag:k, 0][::-1];  c += self.na_diag
            if self.na_cross:
                Phi_0[i, c:c+self.na_cross] = Y[k-self.na_cross:k, 1][::-1]; c += self.na_cross
            Phi_0[i, c:c+self.nb_diag]  = U[k-self.nb_diag:k, 0][::-1];  c += self.nb_diag
            if self.nb_cross:
                Phi_0[i, c:c+self.nb_cross] = U[k-self.nb_cross:k, 1][::-1]; c += self.nb_cross
            Phi_0[i, c:c+self.nc] = Eps[k-self.nc:k, 0][::-1]

            # ── channel 1 (ω) ──
            c = 0
            if self.na_cross:
                Phi_1[i, c:c+self.na_cross] = Y[k-self.na_cross:k, 0][::-1]; c += self.na_cross
            Phi_1[i, c:c+self.na_diag]  = Y[k-self.na_diag:k, 1][::-1];  c += self.na_diag
            if self.nb_cross:
                Phi_1[i, c:c+self.nb_cross] = U[k-self.nb_cross:k, 0][::-1]; c += self.nb_cross
            Phi_1[i, c:c+self.nb_diag]  = U[k-self.nb_diag:k, 1][::-1];  c += self.nb_diag
            Phi_1[i, c:c+self.nc] = Eps[k-self.nc:k, 1][::-1]

        return Phi_0, Phi_1, Y[self.max_lag:]

    @staticmethod
    def _ridge_solve(Phi, y, alpha):
        """Ridge LS.  Returns None if Phi contains non-finite values."""
        if not np.all(np.isfinite(Phi)):
            return None
        A = Phi.T @ Phi + alpha * np.eye(Phi.shape[1])
        return np.linalg.solve(A, Phi.T @ y)

    def fit(self, Y: np.ndarray, U: np.ndarray) -> 'MIMOARMAXModel':
        """Iterative PLR: alternates between LS fit and residual update."""
        Eps = np.zeros_like(Y)
        for _ in range(self.n_iter):
            P0, P1, Y_t = self._build_armax_regressors(Y, U, Eps)
            t0 = self._ridge_solve(P0, Y_t[:, 0], self.ridge_alpha)
            t1 = self._ridge_solve(P1, Y_t[:, 1], self.ridge_alpha)
            if t0 is None or t1 is None:
                break
            self.theta_0_, self.theta_1_ = t0, t1
            # Residual update — clip so Eps cannot blow up on a bad iteration
            raw_eps0 = Y_t[:, 0] - P0 @ self.theta_0_
            raw_eps1 = Y_t[:, 1] - P1 @ self.theta_1_
            Eps[self.max_lag:, 0] = np.clip(raw_eps0, -_Y_CLIP, _Y_CLIP)
            Eps[self.max_lag:, 1] = np.clip(raw_eps1, -_Y_CLIP, _Y_CLIP)

        # Fallback: plain ARX if PLR never succeeded
        if self.theta_0_ is None:
            P0, P1, Y_t = self._build_armax_regressors(Y, U, np.zeros_like(Y))
            self.theta_0_ = self._ridge_solve(P0, Y_t[:, 0], self.ridge_alpha) or np.zeros(P0.shape[1])
            self.theta_1_ = self._ridge_solve(P1, Y_t[:, 1], self.ridge_alpha) or np.zeros(P1.shape[1])

        self._is_fitted = True
        return self

    def _predict_from_buffer(self, Y_buf: np.ndarray, U_buf: np.ndarray) -> np.ndarray:
        """
        Simulation-mode prediction.  MA terms (residuals) are zero in free-run.
        Output is clamped to prevent NaN/Inf from entering the rolling buffer.
        """
        phi_0 = np.concatenate([
            Y_buf[-self.na_diag:, 0][::-1]  if self.na_diag  else np.empty(0),
            Y_buf[-self.na_cross:, 1][::-1] if self.na_cross else np.empty(0),
            U_buf[-self.nb_diag:, 0][::-1],
            U_buf[-self.nb_cross:, 1][::-1] if self.nb_cross else np.empty(0),
            np.zeros(self.nc),
        ])
        phi_1 = np.concatenate([
            Y_buf[-self.na_cross:, 0][::-1] if self.na_cross else np.empty(0),
            Y_buf[-self.na_diag:, 1][::-1],
            U_buf[-self.nb_cross:, 0][::-1] if self.nb_cross else np.empty(0),
            U_buf[-self.nb_diag:, 1][::-1],
            np.zeros(self.nc),
        ])
        raw = np.array([phi_0 @ self.theta_0_, phi_1 @ self.theta_1_])
        return np.clip(np.nan_to_num(raw), -_Y_CLIP, _Y_CLIP)

    def predict(self, Y: np.ndarray, U: np.ndarray, mode: str = 'one_step') -> np.ndarray:
        assert self._is_fitted
        N = len(Y)
        Y_hat = np.full((N, 2), np.nan)

        if mode == 'one_step':
            Eps = np.zeros_like(Y)
            for _ in range(2):
                P0, P1, _ = self._build_armax_regressors(Y, U, Eps)
                yh0 = P0 @ self.theta_0_
                yh1 = P1 @ self.theta_1_
                Eps[self.max_lag:, 0] = np.clip(Y[self.max_lag:, 0] - yh0, -_Y_CLIP, _Y_CLIP)
                Eps[self.max_lag:, 1] = np.clip(Y[self.max_lag:, 1] - yh1, -_Y_CLIP, _Y_CLIP)
            Y_hat[self.max_lag:, 0] = yh0
            Y_hat[self.max_lag:, 1] = yh1

        elif mode == 'simulation':
            Y_buf = Y[:self.max_lag].copy()
            U_buf = U[:self.max_lag].copy()
            for k in range(self.max_lag, N):
                y_hat_k = self._predict_from_buffer(Y_buf, U_buf)
                Y_hat[k] = y_hat_k
                Y_buf = np.roll(Y_buf, -1, axis=0); Y_buf[-1] = y_hat_k
                U_buf = np.roll(U_buf, -1, axis=0); U_buf[-1] = U[k]

        return Y_hat


# ── Output Error (OE) Model ───────────────────────────────────────────────────

class MIMOOEModel(MIMOARXModel):
    """
    Output Error model: the AR regressor is built from the model's own
    simulated outputs instead of the measured outputs.

    This aligns training with the simulation (free-run) use case and
    typically reduces accumulated error in long open-loop trajectories.

    Overflow defence: after each simulation step Y_oe is clipped to
    ±_Y_CLIP, and the ridge solve is skipped if Phi is non-finite.
    """

    def __init__(
        self,
        na_diag: int = 9, na_cross: int = 3,
        nb_diag: int = 9, nb_cross: int = 1,
        n_iter: int = 5,
        ridge_alpha: float = 1e-8,
    ):
        super().__init__(na_diag, na_cross, nb_diag, nb_cross, ridge_alpha)
        self.n_iter = n_iter

    @staticmethod
    def _ridge_solve(Phi, y, alpha):
        if not np.all(np.isfinite(Phi)):
            return None
        A = Phi.T @ Phi + alpha * np.eye(Phi.shape[1])
        return np.linalg.solve(A, Phi.T @ y)

    def fit(self, Y: np.ndarray, U: np.ndarray) -> 'MIMOOEModel':
        """
        Iterative OE: uses simulated outputs in the AR regressor,
        targets true measured Y.  Falls back to ARX if simulation diverges.
        """
        clip = max(float(np.abs(Y).max()) * 3.0, _Y_CLIP)
        Y_oe = Y.copy()

        for _ in range(self.n_iter):
            Phi_0, Phi_1, _ = self._build_regressors(Y_oe, U)
            Y_target = Y[self.max_lag:]

            t0 = self._ridge_solve(Phi_0, Y_target[:, 0], self.ridge_alpha)
            t1 = self._ridge_solve(Phi_1, Y_target[:, 1], self.ridge_alpha)

            if t0 is None or t1 is None:
                break   # Phi was non-finite — keep previous theta

            self.theta_0_, self.theta_1_ = t0, t1
            self._is_fitted = True

            # Re-simulate; sanitise and clip before next regressor build
            Y_sim = self.predict(Y, U, mode='simulation')
            finite = np.isfinite(Y_sim)
            Y_oe = np.where(finite, Y_sim, Y)
            Y_oe = np.clip(Y_oe, -clip, clip)

        # Absolute fallback: plain ARX on true Y
        if not self._is_fitted:
            Phi_0, Phi_1, _ = self._build_regressors(Y, U)
            Y_t = Y[self.max_lag:]
            self.theta_0_ = self._ridge_solve(Phi_0, Y_t[:, 0], self.ridge_alpha) \
                            or np.zeros(Phi_0.shape[1])
            self.theta_1_ = self._ridge_solve(Phi_1, Y_t[:, 1], self.ridge_alpha) \
                            or np.zeros(Phi_1.shape[1])
            self._is_fitted = True

        return self

    def _predict_from_buffer(self, Y_buf: np.ndarray, U_buf: np.ndarray) -> np.ndarray:
        """Clamp output so NaN/Inf cannot propagate into TrajectoryGenerator's buffer."""
        raw = super()._predict_from_buffer(Y_buf, U_buf)
        return np.clip(np.nan_to_num(raw), -_Y_CLIP, _Y_CLIP)
