"""
Evaluation utilities for frame-level and sequence-level tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from config import LABEL_NAMES, NUM_CLASSES, SEQUENCE_LENGTH
from decision_buffer import majority_vote
from feature_extractor import GeometricFeatureExtractor
from frame_ops import generate_noisy_frame, shift_frame
from mlp_classifier import MLPClassifier, Standardizer
from patterns import BASE_PATTERNS


@dataclass
class Evaluator:
    """Runs tests and prints metrics for the reference model."""

    model: MLPClassifier
    standardizer: Standardizer
    extractor: GeometricFeatureExtractor

    def evaluate_frame_dataset(self, name: str, X: np.ndarray, y: np.ndarray) -> float:
        X_norm = self.standardizer.transform(X)
        y_pred = self.model.predict(X_norm)
        accuracy = float((y_pred == y).mean())
        print(f"{name:36s} | frame accuracy = {accuracy * 100.0:6.2f}%")
        return accuracy

    def evaluate_clean_patterns(self) -> float:
        features = []
        labels = []

        for label in range(NUM_CLASSES):
            features.append(self.extractor.extract(BASE_PATTERNS[label]))
            labels.append(label)

        return self.evaluate_frame_dataset(
            name="T1: clean base signs",
            X=np.vstack(features),
            y=np.array(labels, dtype=np.int64),
        )

    def evaluate_shift_test(self, max_shift: int = 1) -> float:
        features = []
        labels = []

        for label in range(NUM_CLASSES):
            for dx in range(-max_shift, max_shift + 1):
                for dy in range(-max_shift, max_shift + 1):
                    frame = shift_frame(BASE_PATTERNS[label], dx=dx, dy=dy)
                    features.append(self.extractor.extract(frame))
                    labels.append(label)

        return self.evaluate_frame_dataset(
            name="T2: shifted clean signs",
            X=np.vstack(features),
            y=np.array(labels, dtype=np.int64),
        )

    def evaluate_noisy_frames(
        self,
        name: str,
        samples_per_class: int,
        flip_probability: float,
        max_shift: int,
        seed: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        features = []
        labels = []

        for label in range(NUM_CLASSES):
            for _ in range(samples_per_class):
                frame = generate_noisy_frame(
                    label=label,
                    rng=rng,
                    max_shift=max_shift,
                    flip_probability=flip_probability,
                )
                features.append(self.extractor.extract(frame))
                labels.append(label)

        X = np.vstack(features)
        y = np.array(labels, dtype=np.int64)
        y_pred = self.model.predict(self.standardizer.transform(X))
        accuracy = float((y_pred == y).mean())
        print(f"{name:36s} | frame accuracy = {accuracy * 100.0:6.2f}%")
        return X, y, y_pred

    def evaluate_sequences(
        self,
        name: str,
        sequence_count: int,
        flip_probability: float,
        max_shift: int,
        seed: int,
        sequence_length: int = SEQUENCE_LENGTH,
    ) -> Tuple[float, float]:
        rng = np.random.default_rng(seed)

        correct_frames = 0
        total_frames = 0
        correct_sequences = 0
        previous_decision = None

        for _ in range(sequence_count):
            label = int(rng.integers(0, NUM_CLASSES))

            frames = [
                generate_noisy_frame(
                    label=label,
                    rng=rng,
                    max_shift=max_shift,
                    flip_probability=flip_probability,
                )
                for _ in range(sequence_length)
            ]

            X = self.extractor.extract_batch(frames)
            probabilities = self.model.predict_proba(self.standardizer.transform(X))
            frame_predictions = probabilities.argmax(axis=1)

            correct_frames += int((frame_predictions == label).sum())
            total_frames += sequence_length

            final_decision = majority_vote(
                predicted_classes=frame_predictions,
                probabilities=probabilities,
                previous_decision=previous_decision,
            )
            previous_decision = final_decision
            correct_sequences += int(final_decision == label)

        frame_accuracy = correct_frames / total_frames
        sequence_accuracy = correct_sequences / sequence_count

        print(
            f"{name:36s} | frame accuracy = {frame_accuracy * 100.0:6.2f}% | "
            f"voting accuracy = {sequence_accuracy * 100.0:6.2f}%"
        )

        return float(frame_accuracy), float(sequence_accuracy)

    @staticmethod
    def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        for true_label, predicted_label in zip(y_true, y_pred):
            matrix[true_label, predicted_label] += 1
        return matrix

    @staticmethod
    def print_confusion_matrix(matrix: np.ndarray) -> None:
        header = "true \\ pred".ljust(12)
        for label in range(NUM_CLASSES):
            header += LABEL_NAMES[label].rjust(10)
        print(header)

        for true_label in range(NUM_CLASSES):
            row = LABEL_NAMES[true_label].ljust(12)
            for predicted_label in range(NUM_CLASSES):
                row += str(matrix[true_label, predicted_label]).rjust(10)
            print(row)
