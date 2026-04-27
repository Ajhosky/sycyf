"""
Fixed-point export helpers for later HDL/Verilog migration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from mlp_classifier import MLPClassifier, Standardizer


def quantize_to_fixed_point(array: np.ndarray, fractional_bits: int = 8) -> np.ndarray:
    """
    Convert floating-point values to signed fixed-point integers.

    Example:
        fractional_bits = 8 means value_fixed = round(value_float * 2^8)
    """
    scale = 2 ** fractional_bits
    return np.round(array * scale).astype(np.int32)


def dequantize_from_fixed_point(array: np.ndarray, fractional_bits: int = 8) -> np.ndarray:
    """Convert signed fixed-point integers back to floating-point values."""
    scale = 2 ** fractional_bits
    return array.astype(np.float64) / scale


def export_quantized_model_npz(
    model: MLPClassifier,
    standardizer: Standardizer,
    filename: str | Path,
    fractional_bits: int = 8,
) -> None:
    """Save float and fixed-point model parameters into an NPZ file."""
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        filename,
        W1_float=model.W1,
        b1_float=model.b1,
        W2_float=model.W2,
        b2_float=model.b2,
        mean_float=standardizer.mean,
        std_float=standardizer.std,
        W1_fixed=quantize_to_fixed_point(model.W1, fractional_bits),
        b1_fixed=quantize_to_fixed_point(model.b1, fractional_bits),
        W2_fixed=quantize_to_fixed_point(model.W2, fractional_bits),
        b2_fixed=quantize_to_fixed_point(model.b2, fractional_bits),
        mean_fixed=quantize_to_fixed_point(standardizer.mean, fractional_bits),
        std_fixed=quantize_to_fixed_point(standardizer.std, fractional_bits),
        fractional_bits=fractional_bits,
    )


def _flatten_values(array: np.ndarray) -> Iterable[int]:
    return array.astype(np.int32).flatten(order="C")


def export_verilog_header(
    model: MLPClassifier,
    filename: str | Path,
    fractional_bits: int = 8,
) -> None:
    """
    Export MLP weights as simple Verilog localparam arrays.

    This is a helper for the next project stage. The generated file is not a complete HDL
    implementation; it only contains quantized constants.
    """
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    arrays = {
        "W1": quantize_to_fixed_point(model.W1, fractional_bits),
        "B1": quantize_to_fixed_point(model.b1, fractional_bits),
        "W2": quantize_to_fixed_point(model.W2, fractional_bits),
        "B2": quantize_to_fixed_point(model.b2, fractional_bits),
    }

    with filename.open("w", encoding="utf-8") as file:
        file.write("// Auto-generated fixed-point MLP parameters\n")
        file.write(f"// Fractional bits: {fractional_bits}\n\n")

        for name, array in arrays.items():
            values = ", ".join(str(value) for value in _flatten_values(array))
            file.write(f"localparam integer {name}_SIZE = {array.size};\n")
            file.write(f"localparam signed [31:0] {name} [0:{array.size - 1}] = '{{{values}}};\n\n")
