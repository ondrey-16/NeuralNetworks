from abc import ABC, abstractmethod

import numpy as np


class WeightsInitializer(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def initialize_weights(self, weights: np.ndarray, biases: np.ndarray) -> None:
        pass
