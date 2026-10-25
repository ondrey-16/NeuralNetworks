import pathlib
import pickle

import matplotlib.pyplot as plt
import numpy as np


def plot_experiment(file_name: str, layer_idx: int = 0, max_weights: int = 8):
    filepath = pathlib.Path(__file__).parent.parent.resolve()
    filepath = filepath.joinpath("history_data", file_name).with_suffix(".pkl")
    with open(filepath, "rb") as f:
        history = pickle.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].plot(history["train_loss"], label="Train Loss", color="royalblue", linewidth=2)
    if history.get("val_loss"):
        axes[0].plot(history["val_loss"], label="Val Loss", color="darkorange", linestyle="--", linewidth=2)
    axes[0].set_xlabel("Epoka")
    axes[0].set_ylabel("Wartość funkcji straty")
    axes[0].set_title("Błąd w kolejnych iteracjach uczenia")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    layer_weights = [epoch_w[layer_idx].flatten() for epoch_w in history["weights"]]
    weights_matrix = np.array(layer_weights)
    num_to_plot = min(max_weights, weights_matrix.shape[1])

    for i in range(num_to_plot):
        axes[1].plot(weights_matrix[:, i], label=f"Waga {i + 1}")

    axes[1].set_xlabel("Epoka")
    axes[1].set_ylabel("Wartość wagi")
    axes[1].set_title(f"Wartości wag (warstwa {layer_idx + 1})")
    axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()


import sys

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_name = sys.argv[1]
    else:
        file_name = "results/model_run.pkl"

    plot_experiment(file_name, layer_idx=0)