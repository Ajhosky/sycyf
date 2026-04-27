"""
Manual implementation of a lightweight MLP classifier.

The code intentionally avoids high-level machine-learning libraries. This makes it easier
to inspect intermediate values and later compare them with HDL/Verilog simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from config import NUM_CLASSES


@dataclass
class Standardizer:
    """Feature standardization: X_normalized = (X - mean) / std."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> "Standardizer":
        return cls(mean=X.mean(axis=0), std=X.std(axis=0) + 1e-9)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    @classmethod
    def fit_transform(cls, X: np.ndarray) -> tuple["Standardizer", np.ndarray]:
        standardizer = cls.fit(X)
        return standardizer, standardizer.transform(X)


@dataclass
class MLPClassifier:
    """
    Lightweight classifier:
        input features -> Dense -> ReLU -> Dense -> class logits
    """

    input_dim: int
    hidden_dim: int = 12
    output_dim: int = NUM_CLASSES
    learning_rate: float = 0.08
    seed: int = 17
    W1: np.ndarray = field(init=False)
    b1: np.ndarray = field(init=False)
    W2: np.ndarray = field(init=False)
    b2: np.ndarray = field(init=False)
    history: List[Dict[str, float]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.W1 = rng.normal(loc=0.0, scale=0.2, size=(self.input_dim, self.hidden_dim))
        self.b1 = np.zeros(self.hidden_dim, dtype=np.float64)
        self.W2 = rng.normal(loc=0.0, scale=0.2, size=(self.hidden_dim, self.output_dim))
        self.b2 = np.zeros(self.output_dim, dtype=np.float64)

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0.0)

    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        return (x > 0.0).astype(np.float64)

    @staticmethod
    def softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_values = np.exp(shifted)
        return exp_values / exp_values.sum(axis=1, keepdims=True)

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run forward pass.

        Returns:
            z1      - hidden pre-activation
            hidden  - hidden activation after ReLU
            logits  - output layer values before softmax
        """
        z1 = X @ self.W1 + self.b1
        hidden = self.relu(z1)
        logits = hidden @ self.W2 + self.b2
        return z1, hidden, logits

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities."""
        _, _, logits = self.forward(X)
        return self.softmax(logits)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class indices."""
        return self.predict_proba(X).argmax(axis=1)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000, verbose: bool = True) -> None:
        """Train the MLP using manual backpropagation."""
        if X.ndim != 2:
            raise ValueError("X must be a 2D feature matrix")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")

        num_samples = X.shape[0]
        y_one_hot = np.zeros((num_samples, self.output_dim), dtype=np.float64)
        y_one_hot[np.arange(num_samples), y] = 1.0

        for epoch in range(epochs):
            z1, hidden, logits = self.forward(X)
            probabilities = self.softmax(logits)

            loss = -np.mean(np.log(probabilities[np.arange(num_samples), y] + 1e-12))
            predictions = probabilities.argmax(axis=1)
            accuracy = float((predictions == y).mean())

            d_logits = (probabilities - y_one_hot) / num_samples

            dW2 = hidden.T @ d_logits
            db2 = d_logits.sum(axis=0)

            d_hidden = d_logits @ self.W2.T
            d_z1 = d_hidden * self.relu_derivative(z1)

            dW1 = X.T @ d_z1
            db1 = d_z1.sum(axis=0)

            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1
            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2

            self.history.append({"epoch": epoch, "loss": float(loss), "accuracy": accuracy})

            if verbose and (epoch % 200 == 0 or epoch == epochs - 1):
                print(
                    f"epoch={epoch:4d} | loss={loss:.4f} | "
                    f"train_accuracy={accuracy * 100.0:.2f}%"
                )
