"""
Geometric-moment feature extraction for 8x8 binary frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from config import FRAME_SIZE
from frame_ops import validate_frame


@dataclass
class GeometricFeatureExtractor:
    """
    Extracts a compact vector of geometric features from a binary 8x8 image.

    Feature vector:
        f0 = M00 / 64
        f1 = x_centroid / 7
        f2 = y_centroid / 7
        f3 = mu20 normalized
        f4 = mu02 normalized
        f5 = mu11 normalized
        f6 = mu30 normalized
        f7 = mu03 normalized

    Raw moments provide basic position and mass information.
    Central moments describe shape around the centroid and are less sensitive to translation.
    """

    frame_size: int = FRAME_SIZE
    feature_names: List[str] = field(
        default_factory=lambda: [
            "M00_norm",
            "x_centroid_norm",
            "y_centroid_norm",
            "mu20_norm",
            "mu02_norm",
            "mu11_norm",
            "mu30_norm",
            "mu03_norm",
        ]
    )

    def __post_init__(self) -> None:
        self.y_grid, self.x_grid = np.mgrid[0:self.frame_size, 0:self.frame_size]

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)

    def extract(self, frame: np.ndarray) -> np.ndarray:
        """Extract feature vector from one 8x8 frame."""
        validate_frame(frame)
        image = frame.astype(np.float64)

        m00 = image.sum()
        if m00 == 0:
            return np.zeros(self.feature_count, dtype=np.float64)

        m10 = (self.x_grid * image).sum()
        m01 = (self.y_grid * image).sum()

        x_centroid = m10 / m00
        y_centroid = m01 / m00

        x = self.x_grid - x_centroid
        y = self.y_grid - y_centroid

        mu20 = ((x ** 2) * image).sum()
        mu02 = ((y ** 2) * image).sum()
        mu11 = ((x * y) * image).sum()
        mu30 = ((x ** 3) * image).sum()
        mu03 = ((y ** 3) * image).sum()

        max_coord = self.frame_size - 1
        scale_2 = m00 * (max_coord ** 2)
        scale_3 = m00 * (max_coord ** 3)

        return np.array(
            [
                m00 / float(self.frame_size * self.frame_size),
                x_centroid / float(max_coord),
                y_centroid / float(max_coord),
                mu20 / scale_2,
                mu02 / scale_2,
                mu11 / scale_2,
                mu30 / scale_3,
                mu03 / scale_3,
            ],
            dtype=np.float64,
        )

    def extract_batch(self, frames: List[np.ndarray] | np.ndarray) -> np.ndarray:
        """Extract feature vectors from many frames."""
        return np.vstack([self.extract(frame) for frame in frames])
