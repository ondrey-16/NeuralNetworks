import numpy as np

from ..losses.loss import Loss


class MSELoss(Loss):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred: np.ndarray, y_target: np.ndarray) -> np.ndarray:
        return np.mean((y_target - y_pred) ** 2)

    def backward(self, y_pred: np.ndarray, y_target: np.ndarray) -> np.ndarray:
        return 2 * (y_pred - y_target) / y_pred.size
