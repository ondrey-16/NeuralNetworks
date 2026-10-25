import numpy as np
from ..losses.loss import Loss


class CCELoss(Loss):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred: np.ndarray, y_target: np.ndarray) -> np.ndarray:
        return -np.sum(y_target * np.log(y_pred))

    def backward(self, y_pred: np.ndarray, y_target: np.ndarray) -> np.ndarray:
        return -y_target / y_pred