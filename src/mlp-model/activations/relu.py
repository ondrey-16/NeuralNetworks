from activations.activation import Activation
import numpy as np

class ReLU(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return np.maximum(0, x)

    def backward(self, x: np.ndarray, grad_out: np.ndarray) -> np.ndarray:
        return grad_out * (x > 0)