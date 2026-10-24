import numpy as np

from ..mlp import MLP
from ..optimizers.optimizer import Optimizer


class SGDMomentumOptimizer(Optimizer):
    def __init__(self, model: MLP, lr=0.01, momentum=0.9):
        super().__init__(model)
        self.lr = lr
        self.m_weights = [np.zeros_like(layer.weights) for layer in model.layers]
        self.m_biases = [np.zeros_like(layer.biases) for layer in model.layers]
        self.momentum = momentum

    def step(self):
        i = 0
        for layer in self.model.layers:
            self.m_weights[i] = (self.momentum * self.m_weights[i] + layer.grad_weights)
            self.m_biases[i] = (self.momentum * self.m_biases[i] + layer.grad_biases)

            layer.weights -= (self.lr * self.m_weights[i])
            layer.biases -= (self.lr * self.m_biases[i])

            i += 1

    def clone(self, model: MLP) -> Optimizer:
        return SGDMomentumOptimizer(model, self.lr, self.momentum)
