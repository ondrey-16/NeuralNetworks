from abc import ABC, abstractmethod

import numpy as np


class Loss(ABC) :
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, y_pred: np.ndarray, y_target: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, y_pred: np.ndarray, y_target: np.ndarray) -> np.ndarray:
        pass
