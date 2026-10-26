import numpy as np
from ..activations.activation import Activation
from ..layers.layer import Layer


class OutputLayer(Layer) :
    def __init__(self, input_size=32, output_size=32, activation: Activation=None, if_bias=True):
        super().__init__(input_size, output_size, if_bias)
        self.activation = activation

    def forward(self, x, train=True):
        y = x @ self.weights
        if self.if_bias:
            y += self.biases
        if train:
            self.in_cache = x
            self.out_cache = y

        return y if self.activation is None else self.activation.forward(y)

    def backward(self, grad_out):
        grad_h = (
            grad_out if self.activation is None
            else self.activation.backward(self.out_cache, grad_out)
        )

        self.grad_weights = self.in_cache.T @ grad_h
        if self.if_bias:
            self.grad_biases = np.sum(grad_h, axis=0)
        else:
            self.grad_biases.fill(0)

        return grad_h @ self.weights.T
