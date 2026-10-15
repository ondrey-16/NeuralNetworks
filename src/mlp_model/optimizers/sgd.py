from mlp import MLP
from optimizers.optimizer import Optimizer


class SGDOptimizer(Optimizer):
    def __init__(self, model: MLP, lr=0.01):
        super().__init__(model)
        self.lr = lr

    def step(self):
        for layer in self.model.layers:
            layer.weights -= (self.lr * layer.grad_weights)
            layer.biases -= (self.lr * layer.grad_biases)

    def clone(self, model: MLP) -> Optimizer:
        return SGDOptimizer(model, self.lr)
