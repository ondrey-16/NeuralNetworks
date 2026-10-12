import numpy as np
from layer import Layer

class OutputLayer(Layer) :
    def __init__(self, input_size=32, output_size=32):
        super().__init__(input_size, output_size)
        self.biases = np.zeros((output_size,))
        self.weights = np.empty((input_size, output_size))

        self.in_cache = None
        self.out_cache = None

    def forward(self, x, train=True):
        y = x @ self.weights + self.biases
        if (train):
            self.in_cache = x
            self.out_cache = y

        return y
