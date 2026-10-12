import numpy as np
from layer import Layer
from ..activations.relu import ReLU

class Hidden(Layer) :
    def __init__(self, input_size=32, output_size=32, activation=ReLU()):
        super().__init__(input_size, output_size)
        self.activation = activation
        self.biases = np.zeros((output_size,))
        self.weights = np.empty((input_size, output_size))

        self.in_cache = None
        self.out_cache = None

    def forward(self, x, train=True):
        y = x @ self.weights + self.biases
        if (train):
            self.in_cache = x
            self.out_cache = y

        return self.activation.forward(y)
