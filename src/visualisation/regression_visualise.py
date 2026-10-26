import pathlib
import matplotlib.pyplot as plt
import numpy as np


def _resolve_save_path(file_name: str, suffix: str, default_ext: str = ".png") -> pathlib.Path:
    raw_path = pathlib.Path(file_name)
    parts = [part for part in raw_path.parts if part not in (raw_path.anchor, "/", "\\")]
    relative_path = pathlib.Path(*parts)

    base_dir = pathlib.Path(__file__).resolve().parent.parent.parent
    target_dir = base_dir.joinpath("visualize_plot")

    full_path = target_dir.joinpath(relative_path)
    extension = full_path.suffix if full_path.suffix else default_ext
    new_filename = f"{full_path.stem}{suffix}{extension}"
    save_path = full_path.parent.joinpath(new_filename)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    return save_path


def _save_or_show(save_path: pathlib.Path | None):
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_learning_curve(history: dict, file_name: str | None = None):
    epochs = range(1, len(history["train_loss"]) + 1)
    modes = [
        (False, "_loss_lin", "Loss"),
        (True, "_loss_log", "Loss (log scale)")
    ]

    for log_scale, suffix, ylabel in modes:
        plt.figure(figsize=(9, 5))
        plt.plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)

        if history.get("val_loss"):
            plt.plot(epochs, history["val_loss"], label="Validation Loss", linewidth=2, linestyle="--")

        if log_scale:
            plt.yscale("log")

        plt.title("Przebieg funkcji straty w kolejnych epokach")
        plt.xlabel("Epoka")
        plt.ylabel(ylabel)
        plt.grid(True, which="both", linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()

        save_path = None
        if file_name is not None:
            save_path = _resolve_save_path(file_name, suffix=suffix, default_ext=".png")

        _save_or_show(save_path)


def plot_regression_fit(history: dict, model, X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray = None, y_test: np.ndarray = None,
                        file_name: str | None = None):
    best_weights = history["weights"][-1]
    for layer, w in zip(model.layers, best_weights):
        layer.weights = w.copy()

    plt.figure(figsize=(9, 5))
    plt.scatter(X_train, y_train, color="gray", alpha=0.6, label="Zbiór uczący")

    if X_test is not None and y_test is not None:
        plt.scatter(X_test, y_test, color="orange", alpha=0.5, label="Zbiór testowy")
        x_min = min(float(np.min(X_train)), float(np.min(X_test)))
        x_max = max(float(np.max(X_train)), float(np.max(X_test)))
    else:
        x_min = float(np.min(X_train))
        x_max = float(np.max(X_train))

    margin = (x_max - x_min) * 0.05
    x_dense = np.linspace(x_min - margin, x_max + margin, 500).reshape(-1, 1)
    y_dense_pred = model.forward(x_dense, train=False)

    plt.plot(x_dense, y_dense_pred, color="red", linewidth=2, label="Predykcja modelu")
    plt.title("Wyniki uczenia sieci dla problemu regresji (R -> R)")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    save_path = _resolve_save_path(file_name, suffix="_fit", default_ext=".png") if file_name else None
    _save_or_show(save_path)


def plot_weights_evolution(history: dict, layer_idx: int = 0, max_weights: int = 16,
                           file_name: str | None = None):
    weights_history = history["weights"]

    layer_weights = np.array([w[layer_idx] for w in weights_history])
    n_epochs_plus_init = layer_weights.shape[0]
    total_weights = layer_weights.shape[1] * layer_weights.shape[2]

    flattened = layer_weights.reshape(n_epochs_plus_init, total_weights)

    plt.figure(figsize=(10, 5))

    if total_weights <= max_weights:
        for i in range(total_weights):
            plt.plot(range(n_epochs_plus_init), flattened[:, i], label=f"Waga {i}")
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    else:
        indices = np.linspace(0, total_weights - 1, max_weights, dtype=int)
        for i in indices:
            plt.plot(range(n_epochs_plus_init), flattened[:, i], label=f"Waga {i}")
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    plt.title(f"Ewolucja wartości wag (Warstwa {layer_idx})")
    plt.xlabel("Krok uczenia (Epoka)")
    plt.ylabel("Wartość wag")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    save_path = _resolve_save_path(file_name, suffix=f"_weights_l{layer_idx}", default_ext=".png") if file_name else None
    _save_or_show(save_path)


def print_cv_summary(cv_results: dict, file_name: str | None = None, print_to_console: bool = True):
    folds = cv_results["folds"]
    n_folds = len(folds)

    lines = []
    for res in folds:
        fold_num = res["fold"]
        val_loss = res["val_loss"]
        metrics = res["val_metrics"]
        lines.append(f"Fold {fold_num}: loss = {val_loss:.6f} | metrics {metrics}")

    mean_loss = cv_results["mean_loss"]
    std_loss = cv_results["std_loss"]
    mean_metrics = cv_results["mean_metrics"]
    std_metrics = cv_results["std_metrics"]

    separator = "-" * 63
    lines.append(separator)
    lines.append(f"Cross validation with {n_folds} folds:")
    lines.append(f"Mean loss: {mean_loss:.6f} (std: {std_loss:.6f})")

    if mean_metrics:
        lines.append("Mean metrics:")
        for metric_name, mean_val in mean_metrics.items():
            std_val = std_metrics.get(metric_name, 0.0)
            lines.append(f"  {metric_name.upper()}: {mean_val:.6f} (std: {std_val:.6f})")

    lines.append(separator)
    summary_text = "\n".join(lines)

    if print_to_console:
        print(summary_text)

    if file_name is not None:
        save_path = _resolve_save_path(file_name, suffix="_cv_summary", default_ext=".txt")
        save_path.write_text(summary_text, encoding="utf-8")

print(_resolve_save_path(file_name="helo",suffix="_helo", default_ext=".png" ))