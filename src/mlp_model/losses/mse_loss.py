import numpy as np

from ..losses.loss import Loss


class MSELoss(Loss):
    def __init__(self):
        super().__init__()

    def _modify_targets(self, y_pred, y_target):
        y_target = np.asarray(y_target)

        if y_pred.ndim == 2 and y_pred.shape[1] > 1:
            if y_target.ndim == 1 or (y_target.ndim == 2 and y_target.shape[1] == 1):
                labels = y_target.reshape(-1).astype(int)
                y_target = np.eye(y_pred.shape[1])[labels]

        return y_target

    def forward(self, y_pred: np.ndarray, y_target: np.ndarray) -> np.ndarray:
        y_target = self._modify_targets(y_pred, y_target)
        return np.mean((y_target - y_pred) ** 2)

    def backward(self, y_pred: np.ndarray, y_target: np.ndarray) -> np.ndarray:
        y_target = self._modify_targets(y_pred, y_target)
        return 2 * (y_pred - y_target) / y_pred.size
