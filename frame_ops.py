"""
Operations on 8x8 frames: shifting and impulse-noise generation.
"""

from __future__ import annotations

import numpy as np

from config import FRAME_SIZE
from patterns import BASE_PATTERNS


def validate_frame(frame: np.ndarray) -> None:
    """Validate that the input is an 8x8 frame."""
    if frame.shape != (FRAME_SIZE, FRAME_SIZE):
        raise ValueError(f"Expected frame shape {(FRAME_SIZE, FRAME_SIZE)}, got {frame.shape}")


def shift_frame(frame: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """
    Shift frame by dx and dy.

    dx > 0 means shift right.
    dx < 0 means shift left.
    dy > 0 means shift down.
    dy < 0 means shift up.

    Pixels shifted outside the matrix are discarded. Empty places are filled with zeros.
    """
    validate_frame(frame)

    output = np.zeros_like(frame)
    height, width = frame.shape

    src_x0 = max(0, -dx)
    src_x1 = min(width, width - dx)
    dst_x0 = max(0, dx)
    dst_x1 = min(width, width + dx)

    src_y0 = max(0, -dy)
    src_y1 = min(height, height - dy)
    dst_y0 = max(0, dy)
    dst_y1 = min(height, height + dy)

    if src_x1 > src_x0 and src_y1 > src_y0:
        output[dst_y0:dst_y1, dst_x0:dst_x1] = frame[src_y0:src_y1, src_x0:src_x1]

    return output


def add_impulse_noise(
    frame: np.ndarray,
    rng: np.random.Generator,
    flip_probability: float,
) -> np.ndarray:
    """
    Add impulse noise by randomly flipping pixels.

    Each pixel is flipped independently with probability `flip_probability`.
    """
    validate_frame(frame)

    if not 0.0 <= flip_probability <= 1.0:
        raise ValueError("flip_probability must be in range [0.0, 1.0]")

    output = frame.copy()
    noise_mask = rng.random(frame.shape) < flip_probability
    output[noise_mask] = 1 - output[noise_mask]

    return output.astype(np.uint8)


def generate_noisy_frame(
    label: int,
    rng: np.random.Generator,
    max_shift: int = 1,
    flip_probability: float = 0.05,
) -> np.ndarray:
    """
    Generate one noisy and shifted 8x8 frame for a selected class.
    """
    if label not in BASE_PATTERNS:
        raise ValueError(f"Unknown label: {label}")

    base_frame = BASE_PATTERNS[label]
    dx = int(rng.integers(-max_shift, max_shift + 1))
    dy = int(rng.integers(-max_shift, max_shift + 1))

    shifted = shift_frame(base_frame, dx=dx, dy=dy)
    noisy = add_impulse_noise(shifted, rng=rng, flip_probability=flip_probability)

    return noisy
