import numpy as np

from ..losses.loss import Loss

class HuberLoss(Loss):
    def __init__(self, delta: int):
        super().__init__()
        self.delta = delta

    def forward(self, y_pred: np.ndarray, y_target: np.ndarray) -> np.ndarray:
        abs_error = np.abs(y_pred - y_target)
        quadratic = np.minimum(abs_error, self.delta)
        linear = abs_error - quadratic
        loss = 0.5 * (quadratic ** 2) + self.delta * linear
        return float(np.mean(loss))


    def backward(self, y_pred: np.ndarray, y_target: np.ndarray) -> np.ndarray:
        error = y_pred - y_target
        abs_error = np.abs(error)
        is_small_error = abs_error <= self.delta

        grad = np.where(is_small_error, error, self.delta * np.sign(error))
        return grad / y_pred.size