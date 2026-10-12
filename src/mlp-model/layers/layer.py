import numpy as np
from abc import ABC, abstractmethod

class Layer(ABC) :
    def __init__(self, input_size=32, output_size=32):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

    @abstractmethod
    def forward(self, x: np.ndarray, train=True) -> np.ndarray:
        pass