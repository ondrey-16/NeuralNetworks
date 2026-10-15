from abc import ABC, abstractmethod

import numpy as np


class Activation(ABC) :
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, x: np.ndarray, grad_out: np.ndarray) -> np.ndarray:
        pass
