import numpy as np
from layers.layer import Layer
from initialize_weights.weights_initializer import WeightsInitializer

class MLP:
    def __init__(self, layers: list[Layer]):
        for i in range(len(layers) - 1):
            if (layers[i].output_size != layers[i + 1].input_size):
                raise ValueError("Błąd architektury. Niespójność warstw!")
        
        self.layers = layers

    def initialize_weights(self, initializer: WeightsInitializer):
        for layer in self.layers:
            layer.initialize_weights(initializer)

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            grad = layer.backward(grad_out=grad)

        return grad