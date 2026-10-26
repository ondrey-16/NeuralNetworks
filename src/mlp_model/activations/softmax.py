import numpy as np
from ..activations.activation import Activation


class Softmax(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted)

        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def backward(self, x: np.ndarray, grad_out: np.ndarray) -> np.ndarray:
        sm = self.forward(x)
        dot = np.sum(grad_out * sm, axis=1, keepdims=True)
        
        return sm * (grad_out - dot)
