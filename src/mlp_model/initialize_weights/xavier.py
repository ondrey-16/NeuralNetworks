import numpy as np

from ..initialize_weights.weights_initializer import WeightsInitializer


class XavierInitializer(WeightsInitializer):
    def __init__(self, random_state=42):
        super().__init__()
        self.rng = np.random.default_rng(random_state)

    def initialize_weights(self, weights: np.ndarray, biases: np.ndarray):
        biases.fill(0)

        n_in, n_out = weights.shape
        x_limit = np.sqrt(6.0 / (n_in + n_out))

        weights[:] = self.rng.uniform(-x_limit, x_limit, weights.shape)
