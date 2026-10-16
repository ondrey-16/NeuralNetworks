import numpy as np
from ..initialize_weights.weights_initializer import WeightsInitializer


class XavierInitializer(WeightsInitializer):
    def __init__(self):
        super().__init__()

    def initialize_weights(self, weights: np.ndarray, biases: np.ndarray):
        biases.fill(0)

        n_in, n_out = weights.shape
        x_limit = np.sqrt(6.0 / (n_in + n_out))

        weights[:] = np.random.uniform(-x_limit, x_limit, weights.shape)
