"""
Base 8x8 binary navigation signs used by the reference model.

Conventions:
    # - active pixel / bright sign pixel
    . - background pixel
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from config import BACK, FORWARD, FRAME_SIZE, LABEL_NAMES, LEFT, RIGHT


def ascii_to_frame(pattern: str) -> np.ndarray:
    """Convert an ASCII 8x8 pattern into a binary NumPy array."""
    lines = [line.strip() for line in pattern.strip().splitlines()]
    frame = np.array(
        [[1 if char == "#" else 0 for char in line] for line in lines],
        dtype=np.uint8,
    )

    if frame.shape != (FRAME_SIZE, FRAME_SIZE):
        raise ValueError(f"Pattern must be {FRAME_SIZE}x{FRAME_SIZE}, got {frame.shape}")

    return frame


BASE_PATTERNS: Dict[int, np.ndarray] = {
    FORWARD: ascii_to_frame(
        """
        ...##...
        ..####..
        .######.
        ...##...
        ...##...
        ...##...
        ...##...
        ...##...
        """
    ),
    LEFT: ascii_to_frame(
        """
        ........
        ...#....
        ..##....
        .#######
        .#######
        ..##....
        ...#....
        ........
        """
    ),
    RIGHT: ascii_to_frame(
        """
        ........
        ....#...
        ....##..
        #######.
        #######.
        ....##..
        ....#...
        ........
        """
    ),
    BACK: ascii_to_frame(
        """
        ...##...
        ...##...
        ...##...
        ...##...
        ...##...
        .######.
        ..####..
        ...##...
        """
    ),
}


def frame_to_ascii(frame: np.ndarray) -> str:
    """Convert a binary frame to a readable ASCII representation."""
    return "\n".join("".join("#" if value else "." for value in row) for row in frame)


def print_frame(frame: np.ndarray) -> None:
    """Print a binary frame in ASCII form."""
    print(frame_to_ascii(frame))


def print_base_patterns() -> None:
    """Print all base patterns with labels."""
    for label, frame in BASE_PATTERNS.items():
        print(f"{LABEL_NAMES[label]}:")
        print_frame(frame)
        print()
