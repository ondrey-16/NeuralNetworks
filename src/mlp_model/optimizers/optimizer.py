from abc import ABC, abstractmethod

from mlp import MLP

class Optimizer(ABC):
    def __init__(self, model: MLP):
        super().__init__()
        self.model = model

    @abstractmethod
    def step(self) -> None:
        pass

    @abstractmethod
    def clone(self, model: MLP) -> Optimizer:
        pass

    def zero_grad(self) -> None:
        for layer in self.model.layers:
            layer.grad_weights.fill(0)
            layer.grad_biases.fill(0)
