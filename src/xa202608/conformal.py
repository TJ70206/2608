from __future__ import annotations

import numpy as np


class SplitConformalRegressor:
    def __init__(self, alpha: float = 0.1) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = float(alpha)
        self.quantile_: float | None = None

    def fit(self, y_true_calib: np.ndarray, y_pred_calib: np.ndarray) -> "SplitConformalRegressor":
        y_true = np.asarray(y_true_calib, dtype=np.float64).reshape(-1)
        y_pred = np.asarray(y_pred_calib, dtype=np.float64).reshape(-1)
        if len(y_true) == 0:
            raise ValueError("calibration set is empty")
        scores = np.abs(y_true - y_pred)
        n = len(scores)
        q_level = np.ceil((n + 1) * (1.0 - self.alpha)) / n
        q_level = min(float(q_level), 1.0)
        self.quantile_ = float(np.quantile(scores, q_level, method="higher"))
        return self

    def predict_interval(self, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.quantile_ is None:
            raise RuntimeError("conformal regressor is not fitted")
        pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
        return pred - self.quantile_, pred + self.quantile_

    @staticmethod
    def coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
        y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
        lower = np.asarray(lower, dtype=np.float64).reshape(-1)
        upper = np.asarray(upper, dtype=np.float64).reshape(-1)
        return float(np.mean((y_true >= lower) & (y_true <= upper)))

    @staticmethod
    def average_width(lower: np.ndarray, upper: np.ndarray) -> float:
        lower = np.asarray(lower, dtype=np.float64).reshape(-1)
        upper = np.asarray(upper, dtype=np.float64).reshape(-1)
        return float(np.mean(upper - lower))
