import pandas as pd
import numpy as np

from mlp_model.mlp import MLP
from mlp_model.layers.hidden import Hidden
from mlp_model.layers.output_layer import OutputLayer
from mlp_model.activations.sigmoid import Sigmoid
from mlp_model.initialize_weights.xavier import XavierInitializer
from mlp_model.activations.softmax import Softmax
from mlp_model.optimizers.sgd import SGDOptimizer

from trainer.trainer import MLPTrainer

def create_mlp_model(input_size, classes_count, 
                     hidden_sizes=(16,), activation_hidden=Sigmoid(), random_state=42, if_bias: bool = True):
    layers = []
    curr_size = input_size

    for size in hidden_sizes:
        layers.append(Hidden(curr_size, size, activation_hidden, if_bias=if_bias))
        curr_size = size

    if classes_count == 2:
        layers.append(OutputLayer(curr_size, 1, Sigmoid(), if_bias=if_bias))
    else:
        layers.append(OutputLayer(curr_size, classes_count, Softmax(), if_bias=if_bias))

    model = MLP(layers)
    model.initialize_weights(XavierInitializer(random_state))

    return model


def run_training(activation_hidden, loss_cls, X_train, y_train, X_val, y_val, hidden_sizes=(16,), classes_count=2, 
                 seeds=(42, 43, 44, 45, 46), epochs=100, batch_size=32, learning_rate=0.01, if_bias: bool = True):
    runs = []

    for seed in seeds:
        np.random.seed(seed)

        model = create_mlp_model(X_train.shape[1], classes_count, hidden_sizes, activation_hidden, random_state=seed, if_bias=if_bias)
        optimizer = SGDOptimizer(model, learning_rate)

        trainer = MLPTrainer(model, optimizer, loss_cls, epochs, batch_size, 42)

        history = trainer.fit(X_train, y_train, X_val, y_val)

        result = {
            "seed": seed,
            "loss": history["val_loss"][-1],
            "metrics": history["val_metrics"][-1],
            "history": history,
            "trainer": trainer
        }

        runs.append(result)
    # i = 0
    # for layer in model.layers:
    #     print(f"warstwa: {i}")
    #     i += 1
    #     print(layer.biases)

    return runs


def summarize_results(trainings):
    summary = []

    for name, runs in trainings.items():
        row = {"config": name}

        losses = [run["loss"] for run in runs]

        row["loss_mean"] = np.mean(losses)
        row["loss_std"] = np.std(losses)

        metrics_names = runs[0]["metrics"].keys()

        for metric in metrics_names:
            values = [run["metrics"][metric] for run in runs]

            row[f"{metric}_mean"] = np.mean(values)
            row[f"{metric}_std"] = np.std(values)

        summary.append(row)

    return pd.DataFrame(summary)

def evaluate_runs(runs, X_test, y_test):
    results = []

    for run in runs:
        trainer = run["trainer"]

        test_loss, test_metrics = trainer.evaluate(
            X_test, y_test
        )

        results.append({"seed": run["seed"], "test_loss": test_loss, **test_metrics})

    return pd.DataFrame(results)