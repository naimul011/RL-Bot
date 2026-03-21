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
"""

import numpy as np
from arx_model import MIMOARXModel


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
        # Channel 0 (v): [u_v(k-1..nb_diag), u_ω(k-1..nb_cross)]
        phi_0 = np.concatenate([
            U_buf[-self.nb_diag:, 0][::-1],
            U_buf[-self.nb_cross:, 1][::-1],
        ])
        # Channel 1 (ω): [u_v(k-1..nb_cross), u_ω(k-1..nb_diag)]
        phi_1 = np.concatenate([
            U_buf[-self.nb_cross:, 0][::-1],
            U_buf[-self.nb_diag:, 1][::-1],
        ])
        return np.array([phi_0 @ self.theta_0_, phi_1 @ self.theta_1_])


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

    # theta_0_ / theta_1_ will be longer than ARX (appended nc MA coeffs per channel)

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

    def fit(self, Y: np.ndarray, U: np.ndarray) -> 'MIMOARMAXModel':
        """Iterative PLR: alternates between LS fit and residual update."""
        Eps = np.zeros_like(Y)

        def ridge_lstsq(Phi, y):
            A = Phi.T @ Phi + self.ridge_alpha * np.eye(Phi.shape[1])
            return np.linalg.solve(A, Phi.T @ y)

        for _ in range(self.n_iter):
            P0, P1, Y_t = self._build_armax_regressors(Y, U, Eps)
            self.theta_0_ = ridge_lstsq(P0, Y_t[:, 0])
            self.theta_1_ = ridge_lstsq(P1, Y_t[:, 1])
            # Update residual estimate
            Eps[self.max_lag:, 0] = Y_t[:, 0] - P0 @ self.theta_0_
            Eps[self.max_lag:, 1] = Y_t[:, 1] - P1 @ self.theta_1_

        self._is_fitted = True
        return self

    def _predict_from_buffer(self, Y_buf: np.ndarray, U_buf: np.ndarray) -> np.ndarray:
        """
        Simulation-mode prediction.  MA terms (residuals) are set to zero
        because no noise realisation is available during free-run.
        """
        phi_0 = np.concatenate([
            Y_buf[-self.na_diag:, 0][::-1]  if self.na_diag  else np.empty(0),
            Y_buf[-self.na_cross:, 1][::-1] if self.na_cross else np.empty(0),
            U_buf[-self.nb_diag:, 0][::-1],
            U_buf[-self.nb_cross:, 1][::-1] if self.nb_cross else np.empty(0),
            np.zeros(self.nc),  # MA terms = 0 in simulation
        ])
        phi_1 = np.concatenate([
            Y_buf[-self.na_cross:, 0][::-1] if self.na_cross else np.empty(0),
            Y_buf[-self.na_diag:, 1][::-1],
            U_buf[-self.nb_cross:, 0][::-1] if self.nb_cross else np.empty(0),
            U_buf[-self.nb_diag:, 1][::-1],
            np.zeros(self.nc),
        ])
        return np.array([phi_0 @ self.theta_0_, phi_1 @ self.theta_1_])

    def predict(self, Y: np.ndarray, U: np.ndarray, mode: str = 'one_step') -> np.ndarray:
        """
        One-step mode uses the augmented (ARMAX) regressor with true residuals.
        Simulation mode uses the base ARX part (MA terms = 0).
        """
        assert self._is_fitted
        N = len(Y)
        Y_hat = np.full((N, 2), np.nan)

        if mode == 'one_step':
            Eps = np.zeros_like(Y)
            P0, P1, _ = self._build_armax_regressors(Y, U, Eps)
            Y_hat_raw_0 = P0 @ self.theta_0_
            Y_hat_raw_1 = P1 @ self.theta_1_
            # Iteratively refine residuals for one-step predictions
            for _ in range(2):
                Eps[self.max_lag:, 0] = Y[self.max_lag:, 0] - Y_hat_raw_0
                Eps[self.max_lag:, 1] = Y[self.max_lag:, 1] - Y_hat_raw_1
                P0, P1, _ = self._build_armax_regressors(Y, U, Eps)
                Y_hat_raw_0 = P0 @ self.theta_0_
                Y_hat_raw_1 = P1 @ self.theta_1_
            Y_hat[self.max_lag:, 0] = Y_hat_raw_0
            Y_hat[self.max_lag:, 1] = Y_hat_raw_1

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

    Training (iterative):
      1. Initialise Y_oe = Y (measured)
      2. Build regressor using Y_oe (model output) + U
      3. Fit LS against true Y target
      4. Re-simulate to get new Y_oe
      5. Repeat n_iter times
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

    def fit(self, Y: np.ndarray, U: np.ndarray) -> 'MIMOOEModel':
        """
        Iterative OE fitting: use simulated (model) outputs in the AR regressor
        while keeping true measured Y as the regression target.
        """
        def ridge_lstsq(Phi, y):
            A = Phi.T @ Phi + self.ridge_alpha * np.eye(Phi.shape[1])
            return np.linalg.solve(A, Phi.T @ y)

        Y_oe = Y.copy()   # starts with measured, becomes model output each iteration

        for _ in range(self.n_iter):
            # Build regressors using simulated Y_oe (not measured Y)
            Phi_0, Phi_1, _ = self._build_regressors(Y_oe, U)
            # But the regression TARGET is the true measured Y
            Y_target = Y[self.max_lag:]

            self.theta_0_ = ridge_lstsq(Phi_0, Y_target[:, 0])
            self.theta_1_ = ridge_lstsq(Phi_1, Y_target[:, 1])
            self._is_fitted = True

            # Re-simulate to get updated Y_oe for next iteration
            Y_sim = self.predict(Y, U, mode='simulation')
            # Warmup region: keep true Y; modelled region: use simulation
            Y_oe = np.where(np.isnan(Y_sim), Y, Y_sim)

        return self
