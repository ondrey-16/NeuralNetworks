import copy
import numpy as np
from sklearn.model_selection import KFold

from mlp_model.mlp import MLP
from mlp_model.optimizers.optimizer import Optimizer
from mlp_model.losses.loss import Loss

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

        self.history = {
            "train_loss": [],
            "val_loss": []
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.forward(np.asarray(X), train=False)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        y = np.asarray(y)

        return float(self.loss.forward(preds, y))

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: np.ndarray, y_val: np.ndarray) -> dict:
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)

        if X_val is not None:
            X_val = np.asarray(X_val)
            y_val = np.asarray(y_val)

        self.history = {
            "train_loss": [],
            "val_loss": []
        }

        rng = np.random.default_rng(self.random_state)
        n_samples = len(X_train)

        for _ in range(self.epochs):
            indices = np.arange(n_samples)

            if self.shuffle:
                rng.shuffle(indices)

            for start in range(0, n_samples, self.batch_size):
                batch_idx = indices[start:start + self.batch_size]

                X_batch = X_train[batch_idx]
                y_batch = y_train[batch_idx]

                self.optimizer.zero_grad()

                preds = self.model.forward(X_batch, train=True)

                grad = self.loss.backward(preds, y_batch)

                self.model.backward(grad=grad)
                self.optimizer.step()

            train_loss = self.evaluate(X_train, y_train)
            self.history["train_loss"].append(train_loss)

            if X_val is not None:
                val_loss = self.evaluate(X_val, y_val)
                self.history["val_loss"].append(val_loss)

        return self.history

    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv=5) -> dict:
        X = np.asarray(X)
        y = np.asarray(y)

        kfold = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)

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
                "history": history
            })
            
            print(f"Fold {fold}: loss = {score:.6f}")
            fold += 1

        scores = np.array([result["val_loss"] for result in fold_results])

        print("---------------------------------------------------------------\n")
        print(f"Cross validation with {cv} folds results:")
        print(f"Mean loss: {float(np.mean(scores))}")
        print(f"Std loss: {float(np.std(scores))}")

        return {
            "folds": fold_results,
            "mean_loss": float(np.mean(scores)),
            "std_loss": float(np.std(scores))
        }