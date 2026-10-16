import copy

import numpy as np
from .initialize_weights.weights_initializer import WeightsInitializer
from .layers.layer import Layer


class MLP:
    def __init__(self, layers: list[Layer]):
        for i in range(len(layers) - 1):
            if (layers[i].output_size != layers[i + 1].input_size):
                raise ValueError("Błąd architektury. Niespójność warstw!")

        self.layers = layers
        self.initializer = None

    def initialize_weights(self, initializer: WeightsInitializer):
        self.initializer = copy.deepcopy(initializer)
        for layer in self.layers:
            layer.initialize_weights(initializer)

    def forward(self, x: np.ndarray, train=False) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x, train)

        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            grad = layer.backward(grad_out=grad)

        return grad

    def clone(self) -> MLP:
        cloned_mlp = copy.deepcopy(self)
        cloned_mlp.initialize_weights(cloned_mlp.initializer)

        for layer in cloned_mlp.layers:
            layer.grad_weights.fill(0)
            layer.grad_biases.fill(0)
            layer.in_cache = None
            layer.out_cache = None

        return cloned_mlp
