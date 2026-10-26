import numpy as np

from ..activations.activation import Activation


class Swish(Activation):
    def __init__(self):
        super().__init__()

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x * self._sigmoid(x)

    def backward(self, x: np.ndarray, grad_out: np.ndarray) -> np.ndarray:
        sigm = self._sigmoid(x)
        grad_swish = sigm + x * sigm * (1.0 - sigm)
        return grad_out * grad_swish