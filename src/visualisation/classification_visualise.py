import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from trainer.trainer import MLPTrainer

def plot_decision_boundary(trainer: MLPTrainer, X: np.ndarray, y: np.ndarray, type=""):
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
    plt.title(f"{type} - Decision boundary")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_loss(trainings, metric):
    plt.figure(figsize=(9, 5))

    for name, runs in trainings.items():
        losses = np.array([run["history"][metric] for run in runs])

        mean = np.mean(losses, axis=0)
        std = np.std(losses, axis=0)

        epochs = np.arange(1, len(mean) + 1)

        line, = plt.plot(epochs, mean, label=name, linewidth=2)

        plt.fill_between(epochs, mean-std, mean + std, 
                         color=line.get_color(), alpha=0.2)

    plt.xlabel("Epochs")
    plt.ylabel("Loss function value")
    plt.title(f"Plot of {metric} function")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_weights(history, layer_idx=0, n_weights=16, type=""):
    weights = np.array([epoch_weights[layer_idx].flatten() \
                          for epoch_weights in history["weights"]])

    epochs = np.arange(weights.shape[0])

    plt.figure(figsize=(10, 8))

    for i in range(n_weights):
        plt.plot(epochs, weights[:, i], label=f"Weight {i + 1}")

    plt.xlabel("Epoch")
    plt.ylabel("Weight value")
    plt.title(f"{type} - Weights evolution")
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=4
    )
    plt.tight_layout()
    plt.grid(alpha=0.3)
    plt.show()

def plot_confussion_matrix(y_val, y_pred):
    ConfusionMatrixDisplay.from_predictions(
        y_val, y_pred,
        cmap="Blues"
    )

    plt.show()