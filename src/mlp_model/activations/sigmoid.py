import numpy as np
from ..activations.activation import Activation


class Sigmoid(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def backward(self, x: np.ndarray, grad_out: np.ndarray) -> np.ndarray:
        sigm = self.forward(x)

        return grad_out * sigm * (1.0 - sigm)
