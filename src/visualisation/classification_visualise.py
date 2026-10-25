import numpy as np
import matplotlib.pyplot as plt

from ..trainer import MLPTrainer

def plot_decision_boundary(trainer: MLPTrainer, X: np.ndarray, y: np.ndarray):
    y = y.reshape(-1)
    
    x_min = X[:, 0].min() - 0.2
    x_max = X[:, 0].max() + 0.2
    y_min = X[:, 1].min() - 0.2
    y_max = X[:, 1].max() + 0.2

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), 
                         np.linspace(y_min, y_max, 250))

    grid = np.c_[xx.ravel(), yy.ravel()]

    preds = trainer.predict(grid)

    if preds.ndim == 2 and preds.shape[1] > 1:
        predicted_classes = np.argmax(preds, axis=1)
    else:
        predicted_classes = (preds.reshape(-1) >= 0.5).astype(int)

    Z = predicted_classes.reshape(xx.shape)
    classes = np.sort(np.unique(y))

    plt.figure(figsize=(8, 6))

    plt.contourf(xx, yy, Z, levels=np.arange(len(classes) + 1) - 0.5,
                 cmap="tab10", alpha=0.25)

    for cls in classes:
        mask = y == cls

        plt.scatter(X[mask, 0], X[mask, 1], label=f"Class {cls}", s=20, 
        edgecolors="black", linewidths=0.2)

    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("Decision boundary")
    plt.legend()
    plt.tight_layout()
    plt.show()