import copy

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, f1_score

from mlp_model.losses.loss import Loss
from mlp_model.mlp import MLP
from mlp_model.optimizers.optimizer import Optimizer


class MLPTrainer:
    def __init__(self, model: MLP, optimizer: Optimizer, loss: Loss, epochs=100,
                 batch_size=32, shuffle=True, random_state=42):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state

        self.history = self._init_history()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.forward(np.asarray(X), train=False)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        y = np.asarray(y)

        loss = float(self.loss.forward(preds, y))
        metrics = self._get_metrics(preds, y)

        return loss, metrics

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray) -> dict:
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)

        if X_val is not None:
            X_val = np.asarray(X_val)
            y_val = np.asarray(y_val)

        self.history = self._init_history()

        self.history["weights"].append(self._get_weights())

        rng = np.random.default_rng(self.random_state)
        n_samples = len(X_train)

        for _ in range(self.epochs):
            indices = np.arange(n_samples)

            if self.shuffle:
                rng.shuffle(indices)

            epoch_grad_norms = []

            for start in range(0, n_samples, self.batch_size):
                batch_idx = indices[start:start + self.batch_size]

                X_batch = X_train[batch_idx]
                y_batch = y_train[batch_idx]

                self.optimizer.zero_grad()

                preds = self.model.forward(X_batch, train=True)

                batch_loss = self.loss.forward(preds, y_batch)
                self.history["batch_loss"].append(float(batch_loss))

                grad = self.loss.backward(preds, y_batch)
                self.model.backward(grad=grad)

                epoch_grad_norms.append(self._get_gradient_norms())

                self.optimizer.step()

            train_loss, metrics = self.evaluate(X_train, y_train)
            self.history["train_loss"].append(train_loss)

            self.history["train_accuracy"].append(metrics["accuracy"])
            self.history["train_precision"].append(metrics["precision"])
            self.history["train_f1_score"].append(metrics["f1_score"])

            if X_val is not None:

                val_loss, metrics = self.evaluate(X_val, y_val)
                self.history["val_loss"].append(val_loss)

                self.history["val_accuracy"].append(metrics["accuracy"])
                self.history["val_precision"].append(metrics["precision"])
                self.history["val_f1_score"].append(metrics["f1_score"])

            self.history["grad_norms"].append(np.mean(epoch_grad_norms, axis=0).tolist())
            self.history["weights"].append(self._get_weights())

        return self.history

    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv=5) -> dict:
        X = np.asarray(X)
        y = np.asarray(y)

        kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)

        fold_results = []

        fold = 1
        for (train_idx, val_idx) in kfold.split(X, y):
            model = self.model.clone()
            optimizer = self.optimizer.clone(model)

            fold_trainer = MLPTrainer(model, optimizer, copy.deepcopy(self.loss), self.epochs,
                                   self.batch_size, self.shuffle, self.random_state)

            history = fold_trainer.fit(X[train_idx], y[train_idx],
                                       X[val_idx], y[val_idx])

            score = history["val_loss"][-1]
            fold_results.append({
                "fold": fold,
                "val_loss": score,
                "val_accuracy": history["val_accuracy"][-1],
                "val_precision": history["val_precision"][-1],
                "val_f1_score": history["val_f1_score"][-1],
                "history": history
            })

            print(f"Fold {fold}: loss = {score:.6f}")
            fold += 1

        scores = np.array([result["val_loss"] for result in fold_results])
        accuracies = np.array([result["val_accuracy"] for result in fold_results])
        precisions = np.array([result["val_precision"] for result in fold_results])
        f1_scores = np.array([result["val_f1_score"] for result in fold_results])

        print("---------------------------------------------------------------\n")
        print(f"Cross validation with {cv} folds results:")
        print(f"Mean loss: {float(np.mean(scores)):.6f}")
        print(f"Std loss: {float(np.std(scores)):.6f}")
        print("---------------------------------------------------------------")
        print(f"Mean accuracy: {float(np.mean(accuracies)):.6f}")
        print(f"Std accuracy: {float(np.std(accuracies)):.6f}")
        print("---------------------------------------------------------------")
        print(f"Mean precision: {float(np.mean(precisions)):.6f}")
        print(f"Std precision: {float(np.std(precisions)):.6f}")
        print("---------------------------------------------------------------")
        print(f"Mean f1_score: {float(np.mean(f1_scores)):.6f}")
        print(f"Std f1_score: {float(np.std(f1_scores)):.6f}")

        return {
            "folds": fold_results,
            "mean_loss": float(np.mean(scores)),
            "std_loss": float(np.std(scores)),
            "mean_accuracy": float(np.mean(accuracies)),
            "std_accuracy": float(np.std(accuracies)),
            "mean_precision": float(np.mean(precisions)),
            "std_precision": float(np.std(precisions)),
            "mean_f1_score": float(np.mean(f1_scores)),
            "std_f1_score": float(np.std(f1_scores)),
        }

    def _init_history(self):
        return {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
            "train_precision": [],
            "val_precision": [],
            "train_f1_score": [],
            "val_f1_score": [],
            "batch_loss": [],
            "weights": [],
            "grad_norms": []
        }

    def _get_weights(self):
        return [
            layer.weights.copy() for layer in self.model.layers
        ]

    def _get_gradient_norms(self):
        return [
            float(np.linalg.norm(layer.grad_weights)) for layer in self.model.layers
        ]

    def _get_metrics(self, preds: np.ndarray, y: np.ndarray):
        if preds.ndim == 2 and preds.shape[1] > 1:
            y_pred = np.argmax(preds, axis=1)
        else:
            y_pred = (preds.reshape(-1) >= 0.5).astype(int)

        y_true = np.asarray(y).reshape(-1)

        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "f1_score": f1_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
        }