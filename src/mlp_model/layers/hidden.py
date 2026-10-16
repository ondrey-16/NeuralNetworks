import numpy as np
from ..activations.relu import ReLU
from ..layers.layer import Layer


class Hidden(Layer) :
    def __init__(self, input_size=32, output_size=32, activation=None):
        super().__init__(input_size, output_size)
        self.activation = ReLU() if activation is None else activation

    def forward(self, x, train=True):
        y = x @ self.weights + self.biases
        if (train):
            self.in_cache = x
            self.out_cache = y

        return self.activation.forward(y)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        grad_h = self.activation.backward(self.out_cache, grad_out)

        self.grad_weights = self.in_cache.T @ grad_h
        self.grad_biases = np.sum(grad_h, axis=0)

        return grad_h @ self.weights.T
