import numpy as np

from ..activations.activation import Activation


class Tanh(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return np.tanh(x)

    def backward(self, x: np.ndarray, grad_out: np.ndarray) -> np.ndarray:
        tanh = self.forward(x)

        return grad_out * (1.0 - tanh ** 2)
