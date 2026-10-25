from abc import ABC, abstractmethod

import numpy as np

from ..initialize_weights.weights_initializer import WeightsInitializer


class Layer(ABC) :
    def __init__(self, input_size=32, output_size=32):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.biases = np.zeros((output_size,))
        self.weights = np.empty((input_size, output_size))

        self.grad_weights = np.zeros_like(self.weights)
        self.grad_biases = np.zeros_like(self.biases)

        self.in_cache = None
        self.out_cache = None

    @abstractmethod
    def forward(self, x: np.ndarray, train=True) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        pass

    def initialize_weights(self, initializer: WeightsInitializer):
        initializer.initialize_weights(self.weights, self.biases)
