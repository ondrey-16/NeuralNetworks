import numpy as np
from layers.layer import Layer

class MLP:
    def __init__(self, layers: list[Layer]):
        for i in range(len(layers) - 1):
            if (layers[i].output_size != layers[i].input_size):
                raise ValueError("Błąd architektury. Niespójność warstw!")
        
        self.layers = layers

    def forward(self, x: np.ndarray):
        for layer in self.layers:
            x = layer.forward(x)

        return x