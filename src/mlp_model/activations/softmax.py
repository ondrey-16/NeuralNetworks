import numpy as np
from ..activations.activation import Activation


class Softmax(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)

    def backward(self, x: np.ndarray, grad_out: np.ndarray) -> np.ndarray:
        sm = self.forward(x)
        return sm * (grad_out - np.sum(grad_out * sm))
