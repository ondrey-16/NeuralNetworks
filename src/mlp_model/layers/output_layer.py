import numpy as np

from ..layers.layer import Layer


class OutputLayer(Layer) :
    def __init__(self, input_size=32, output_size=32):
        super().__init__(input_size, output_size)

    def forward(self, x, train=True):
        y = x @ self.weights + self.biases
        if (train):
            self.in_cache = x
            self.out_cache = y

        return y

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        self.grad_weights = self.in_cache.T @ grad_out
        self.grad_biases = np.sum(grad_out, axis=0)

        return grad_out @ self.weights.T
