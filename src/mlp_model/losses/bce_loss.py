import numpy as np
from ..losses.loss import Loss


class BCELoss(Loss):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred: np.ndarray, y_target: np.ndarray) -> np.ndarray:
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
        
        return -np.mean(y_target * np.log(y_pred) + (1 - y_target) * np.log(1 - y_pred))

    def backward(self, y_pred: np.ndarray, y_target: np.ndarray) -> np.ndarray:
        return (y_pred - y_target) / (y_pred * (1 - y_pred) * y_pred.size)