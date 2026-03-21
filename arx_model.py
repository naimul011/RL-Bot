"""
MIMO ARX (AutoRegressive with eXogenous input) model implementation.

Implements the paper's asymmetric per-channel ARX structure:
  - Diagonal channels (y11, y22, u11, u22): order 9
  - Cross channels (y12, y21): order 3
  - Cross exogenous (u12, u21): order 1

The MIMO ARX model is fit as two independent MISO (multi-input single-output)
least-squares problems, one per output channel.

MIMO ARX (eq. 6 in paper):
  y(k) = sum_{i=1}^{na} Ai * y(k-i)  +  sum_{i=1}^{nb} Bi * u(k-i)  + e(k)

Based on: Lee, Paulik, Krishnan, MWSCAS 2023
"""

import numpy as np


class MIMOARXModel:
    """
    MIMO ARX model with asymmetric per-channel polynomial orders.

    The model treats each output as a separate MISO least-squares problem:

    For v_out (channel 0):
      phi_0(k) = [v_out(k-1:k-na_diag),     # AR self-feedback
                  omega_out(k-1:k-na_cross),  # AR cross-coupling
                  v_in(k-1:k-nb_diag),        # exogenous self
                  omega_in(k-1:k-nb_cross)]   # exogenous cross

    For omega_out (channel 1):
      phi_1(k) = [v_out(k-1:k-na_cross),     # AR cross-coupling
                  omega_out(k-1:k-na_diag),   # AR self-feedback
                  v_in(k-1:k-nb_cross),        # exogenous cross
                  omega_in(k-1:k-nb_diag)]    # exogenous self
    """

    def __init__(
        self,
        na_diag: int = 9,    # AR order for self-channel (diagonal)
        na_cross: int = 3,   # AR order for cross-channel
        nb_diag: int = 9,    # Exogenous order for self-channel (diagonal)
        nb_cross: int = 1,   # Exogenous order for cross-channel
        ridge_alpha: float = 1e-8,  # Ridge regularization for numerical stability
    ):
        self.na_diag = na_diag
        self.na_cross = na_cross
        self.nb_diag = nb_diag
        self.nb_cross = nb_cross
        self.ridge_alpha = ridge_alpha

        self.max_lag = max(na_diag, nb_diag)

        # Fitted parameters (set by fit())
        self.theta_0_ = None   # parameters for v_out channel
        self.theta_1_ = None   # parameters for omega_out channel
        self._is_fitted = False

    def _build_regressors(self, Y: np.ndarray, U: np.ndarray) -> tuple:
        """
        Build regressor matrices for both output channels.

        Args:
            Y: shape (N, 2)  — [v_out, omega_out]
            U: shape (N, 2)  — [v_in, omega_in]

        Returns:
            (Phi_0, Phi_1, Y_target):
                Phi_0: shape (N_valid, n_regs_0) for v_out
                Phi_1: shape (N_valid, n_regs_1) for omega_out
                Y_target: shape (N_valid, 2)
        """
        N = len(Y)
        N_valid = N - self.max_lag
        assert N_valid > 0, "Not enough data for the specified model order."

        n_regs_0 = self.na_diag + self.na_cross + self.nb_diag + self.nb_cross
        n_regs_1 = self.na_cross + self.na_diag + self.nb_cross + self.nb_diag

        Phi_0 = np.zeros((N_valid, n_regs_0))
        Phi_1 = np.zeros((N_valid, n_regs_1))

        for i, k in enumerate(range(self.max_lag, N)):
            # Channel 0: v_out
            col = 0
            # AR: past v_out (self, diagonal)
            Phi_0[i, col:col + self.na_diag] = Y[k - self.na_diag:k, 0][::-1]
            col += self.na_diag
            # AR: past omega_out (cross)
            Phi_0[i, col:col + self.na_cross] = Y[k - self.na_cross:k, 1][::-1]
            col += self.na_cross
            # Exogenous: past v_in (self, diagonal)
            Phi_0[i, col:col + self.nb_diag] = U[k - self.nb_diag:k, 0][::-1]
            col += self.nb_diag
            # Exogenous: past omega_in (cross)
            Phi_0[i, col:col + self.nb_cross] = U[k - self.nb_cross:k, 1][::-1]

            # Channel 1: omega_out
            col = 0
            # AR: past v_out (cross)
            Phi_1[i, col:col + self.na_cross] = Y[k - self.na_cross:k, 0][::-1]
            col += self.na_cross
            # AR: past omega_out (self, diagonal)
            Phi_1[i, col:col + self.na_diag] = Y[k - self.na_diag:k, 1][::-1]
            col += self.na_diag
            # Exogenous: past v_in (cross)
            Phi_1[i, col:col + self.nb_cross] = U[k - self.nb_cross:k, 0][::-1]
            col += self.nb_cross
            # Exogenous: past omega_in (self, diagonal)
            Phi_1[i, col:col + self.nb_diag] = U[k - self.nb_diag:k, 1][::-1]

        Y_target = Y[self.max_lag:]

        return Phi_0, Phi_1, Y_target

    def fit(self, Y: np.ndarray, U: np.ndarray) -> 'MIMOARXModel':
        """
        Estimate ARX parameters via ridge-regularized least squares.

        Args:
            Y: shape (N, 2)  — ground-truth output [v_out, omega_out]
            U: shape (N, 2)  — input commands [v_in, omega_in]

        Returns:
            self (for method chaining)
        """
        Phi_0, Phi_1, Y_target = self._build_regressors(Y, U)

        # Ridge-regularized least squares: (Phi'Phi + alpha*I) theta = Phi'y
        def ridge_lstsq(Phi, y):
            A = Phi.T @ Phi
            A += self.ridge_alpha * np.eye(A.shape[0])
            b = Phi.T @ y
            return np.linalg.solve(A, b)

        self.theta_0_ = ridge_lstsq(Phi_0, Y_target[:, 0])
        self.theta_1_ = ridge_lstsq(Phi_1, Y_target[:, 1])
        self._is_fitted = True

        return self

    def _predict_from_buffer(
        self,
        Y_buf: np.ndarray,
        U_buf: np.ndarray,
    ) -> np.ndarray:
        """
        Predict y(k) from history buffers.

        Args:
            Y_buf: shape (max_lag, 2) — most recent outputs (oldest first)
            U_buf: shape (max_lag, 2) — most recent inputs (oldest first)

        Returns:
            y_hat: shape (2,)
        """
        assert self._is_fitted, "Model must be fitted before prediction."

        # Channel 0 (v_out)
        phi_0 = np.concatenate([
            Y_buf[-self.na_diag:, 0][::-1],
            Y_buf[-self.na_cross:, 1][::-1],
            U_buf[-self.nb_diag:, 0][::-1],
            U_buf[-self.nb_cross:, 1][::-1],
        ])
        v_hat = phi_0 @ self.theta_0_

        # Channel 1 (omega_out)
        phi_1 = np.concatenate([
            Y_buf[-self.na_cross:, 0][::-1],
            Y_buf[-self.na_diag:, 1][::-1],
            U_buf[-self.nb_cross:, 0][::-1],
            U_buf[-self.nb_diag:, 1][::-1],
        ])
        omega_hat = phi_1 @ self.theta_1_

        return np.array([v_hat, omega_hat])

    def predict(
        self,
        Y: np.ndarray,
        U: np.ndarray,
        mode: str = 'one_step',
    ) -> np.ndarray:
        """
        Predict over a full sequence.

        Args:
            Y:    shape (N, 2) — true outputs (used as history in one_step mode)
            U:    shape (N, 2) — input commands
            mode: 'one_step'   — use true past Y (for validation/fit%)
                  'simulation' — use own past predictions (free-run, for trajectories)

        Returns:
            Y_hat: shape (N, 2), NaN for the first max_lag warmup steps
        """
        assert self._is_fitted
        N = len(Y)
        Y_hat = np.full((N, 2), np.nan)

        if mode == 'one_step':
            # Use true outputs for lag terms — standard one-step-ahead prediction
            Phi_0, Phi_1, _ = self._build_regressors(Y, U)
            Y_hat[self.max_lag:, 0] = Phi_0 @ self.theta_0_
            Y_hat[self.max_lag:, 1] = Phi_1 @ self.theta_1_

        elif mode == 'simulation':
            # Free-run: use own predictions for lag terms
            Y_buf = Y[:self.max_lag].copy()    # warmup from true data
            U_buf = U[:self.max_lag].copy()

            for k in range(self.max_lag, N):
                y_hat_k = self._predict_from_buffer(Y_buf, U_buf)
                Y_hat[k] = y_hat_k

                # Shift buffers: drop oldest, append newest
                Y_buf = np.roll(Y_buf, -1, axis=0)
                Y_buf[-1] = y_hat_k
                U_buf = np.roll(U_buf, -1, axis=0)
                U_buf[-1] = U[k]

        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'one_step' or 'simulation'.")

        return Y_hat

    def score(self, Y_true: np.ndarray, Y_hat: np.ndarray) -> dict:
        """
        Compute goodness-of-fit metrics matching MATLAB SI Toolbox convention.

        Fit percentage:
            fit% = max(0, 100 * (1 - ||y - y_hat|| / ||y - mean(y)||))

        Only valid (non-NaN) predictions are included.

        Returns dict with fit_v, fit_omega, rmse_v, rmse_omega, r2_v, r2_omega
        """
        results = {}
        names = ['v', 'omega']

        for i, name in enumerate(names):
            y = Y_true[:, i]
            yh = Y_hat[:, i]

            # Filter valid predictions
            mask = ~np.isnan(yh)
            y = y[mask]
            yh = yh[mask]

            residuals = y - yh
            baseline = y - np.mean(y)

            fit_pct = max(0.0, 100.0 * (1.0 - np.linalg.norm(residuals) / np.linalg.norm(baseline)))
            rmse = np.sqrt(np.mean(residuals ** 2))
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum(baseline ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

            results[f'fit_{name}'] = fit_pct
            results[f'rmse_{name}'] = rmse
            results[f'r2_{name}'] = r2

        return results

    def check_stability(self) -> dict:
        """
        Check free-run stability by computing eigenvalues of the AR companion matrix.

        A model is stable if all AR eigenvalues lie strictly inside the unit circle.
        """
        assert self._is_fitted

        def _companion_eigvals(theta, na, offset=0):
            a_coeffs = theta[offset:offset + na]
            if na == 0:
                return np.array([])
            A = np.zeros((na, na))
            A[0, :] = a_coeffs
            if na > 1:
                A[1:, :-1] = np.eye(na - 1)
            return np.linalg.eigvals(A)

        eigs_v = _companion_eigvals(self.theta_0_, self.na_diag)
        eigs_omega = _companion_eigvals(self.theta_1_, self.na_diag, offset=self.na_cross)

        return {
            'stable_v': bool(np.all(np.abs(eigs_v) < 1.0)),
            'stable_omega': bool(np.all(np.abs(eigs_omega) < 1.0)),
            'max_eigval_v': float(np.max(np.abs(eigs_v))) if len(eigs_v) > 0 else 0.0,
            'max_eigval_omega': float(np.max(np.abs(eigs_omega))) if len(eigs_omega) > 0 else 0.0,
        }
