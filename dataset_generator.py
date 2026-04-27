"""
Dataset generation for training and testing the Define-stage reference model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from config import NUM_CLASSES, SEQUENCE_LENGTH
from feature_extractor import GeometricFeatureExtractor
from frame_ops import generate_noisy_frame


@dataclass
class DatasetGenerator:
    """
    Generates noisy/shifted frames and converts them to feature vectors.
    """

    extractor: GeometricFeatureExtractor
    seed: int = 17
    max_shift: int = 1
    flip_probability: float = 0.05

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def generate_frame(self, label: int) -> np.ndarray:
        """Generate one noisy frame for a given label."""
        return generate_noisy_frame(
            label=label,
            rng=self.rng,
            max_shift=self.max_shift,
            flip_probability=self.flip_probability,
        )

    def generate_sequence(self, label: int, length: int = SEQUENCE_LENGTH) -> List[np.ndarray]:
        """Generate a sequence of noisy frames for one label."""
        return [self.generate_frame(label) for _ in range(length)]

    def make_dataset(
        self,
        samples_per_class: int,
        labels: Sequence[int] = tuple(range(NUM_CLASSES)),
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a frame-level dataset.

        Returns:
            X - feature matrix, shape [N, feature_count]
            y - label vector, shape [N]
        """
        features = []
        targets = []

        for label in labels:
            for _ in range(samples_per_class):
                frame = self.generate_frame(label)
                features.append(self.extractor.extract(frame))
                targets.append(label)

        return np.vstack(features), np.array(targets, dtype=np.int64)

    def make_sequence_dataset(
        self,
        sequences_per_class: int,
        sequence_length: int = SEQUENCE_LENGTH,
        labels: Sequence[int] = tuple(range(NUM_CLASSES)),
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Create a sequence-level dataset.

        Returns:
            X_sequences - list of arrays, each array has shape [sequence_length, feature_count]
            y           - label for each sequence
        """
        sequences = []
        targets = []

        for label in labels:
            for _ in range(sequences_per_class):
                frames = self.generate_sequence(label, length=sequence_length)
                sequences.append(self.extractor.extract_batch(frames))
                targets.append(label)

        return sequences, np.array(targets, dtype=np.int64)
