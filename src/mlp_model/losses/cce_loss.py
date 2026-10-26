import numpy as np
from ..losses.loss import Loss


class CCELoss(Loss):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred: np.ndarray, y_target: np.ndarray) -> np.ndarray:
        y_target = np.asarray(y_target).reshape(-1).astype(int)
        probs = np.clip(y_pred[np.arange(y_pred.shape[0]), y_target], 1e-12, 1.0)

        return -float(np.mean(np.log(probs)))

    def backward(self, y_pred: np.ndarray, y_target: np.ndarray) -> np.ndarray:
        y_target = np.asarray(y_target).reshape(-1).astype(int)
        grad = np.zeros_like(y_pred)

        probs = np.clip(y_pred[np.arange(y_pred.shape[0]), y_target], 1e-12, 1.0)
        grad[np.arange(y_pred.shape[0]), y_target] = -1.0 / (y_pred.shape[0] * probs)

        return grad