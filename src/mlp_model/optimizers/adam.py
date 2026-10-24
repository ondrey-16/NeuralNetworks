import numpy as np

from ..mlp import MLP
from ..optimizers.optimizer import Optimizer


class AdamOptimizer(Optimizer):
    def __init__(self, model: MLP, lr=0.01, beta1=0.9, beta2=0.99, eps=1e-8):
            super().__init__(model)
            self.lr = lr
            self.beta1 = beta1
            self.beta2 = beta2
            self.eps = eps

            self.t = 0

            self.m_weights = [np.zeros_like(layer.weights) for layer in model.layers]
            self.v_weights = [np.zeros_like(layer.weights) for layer in model.layers]

            self.m_biases = [np.zeros_like(layer.biases) for layer in model.layers]
            self.v_biases = [np.zeros_like(layer.biases) for layer in model.layers]

    def step(self):
        self.t += 1

        i = 0
        for layer in self.model.layers:
            dw = layer.grad_weights
            db = layer.grad_biases

            self.m_weights[i] = (self.beta1 * self.m_weights[i] + (1 - self.beta1) * dw)
            self.m_biases[i] = (self.beta1 * self.m_biases[i] + (1 - self.beta1) * db)

            self.v_weights[i] = (self.beta2 * self.v_weights[i] + (1 - self.beta2) * (dw ** 2))
            self.v_biases[i] = (self.beta2 * self.v_biases[i] + (1 - self.beta2) * (db ** 2))\

            mw = self.m_weights[i] / (1 - self.beta1 ** self.t)
            mb = self.m_biases[i] / (1 - self.beta1 ** self.t)

            vw = self.v_weights[i] / (1 - self.beta2 ** self.t)
            vb = self.v_biases[i] / (1 - self.beta2 ** self.t)

            layer.weights -= (self.lr * mw / (np.sqrt(vw) + self.eps))
            layer.biases -= (self.lr * mb / (np.sqrt(vb) + self.eps))

            i += 1

    def clone(self, model: MLP) -> Optimizer:
        return AdamOptimizer(model, self.lr, self.beta1, self.beta2, self.eps)
