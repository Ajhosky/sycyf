"""
Main experiment for the Define-stage reference model.

Run from this directory:
    python main.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from config import DEFAULT_SEED, FORWARD, LABEL_NAMES, LEFT, RIGHT
from dataset_generator import DatasetGenerator
from decision_buffer import majority_vote
from evaluator import Evaluator
from feature_extractor import GeometricFeatureExtractor
from fixed_point import export_quantized_model_npz, export_verilog_header
from mlp_classifier import MLPClassifier, Standardizer
from patterns import print_base_patterns


def main() -> None:
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    print("Base 8x8 patterns:\n")
    print_base_patterns()

    extractor = GeometricFeatureExtractor()

    train_generator = DatasetGenerator(
        extractor=extractor,
        seed=DEFAULT_SEED,
        max_shift=1,
        flip_probability=0.05,
    )

    print("Generating training dataset...")
    X_train, y_train = train_generator.make_dataset(samples_per_class=2000)

    standardizer = Standardizer.fit(X_train)
    X_train_norm = standardizer.transform(X_train)

    print("\nTraining lightweight MLP:")
    model = MLPClassifier(
        input_dim=extractor.feature_count,
        hidden_dim=12,
        output_dim=4,
        learning_rate=0.08,
        seed=DEFAULT_SEED,
    )
    model.fit(X_train_norm, y_train, epochs=1000, verbose=True)

    evaluator = Evaluator(model=model, standardizer=standardizer, extractor=extractor)

    print("\nFrame-level tests:")
    evaluator.evaluate_clean_patterns()
    evaluator.evaluate_shift_test(max_shift=1)

    _, _, _ = evaluator.evaluate_noisy_frames(
        name="T3: light noise, shifted signs",
        samples_per_class=250,
        flip_probability=0.05,
        max_shift=1,
        seed=101,
    )

    X_cm, y_cm, y_pred_cm = evaluator.evaluate_noisy_frames(
        name="T4: strong noise, shifted signs",
        samples_per_class=250,
        flip_probability=0.10,
        max_shift=1,
        seed=102,
    )

    _, _, _ = evaluator.evaluate_noisy_frames(
        name="T5: very strong noise",
        samples_per_class=250,
        flip_probability=0.15,
        max_shift=1,
        seed=103,
    )

    print("\nSequence-level tests with majority voting:")
    evaluator.evaluate_sequences(
        name="T6: sequences, light noise",
        sequence_count=500,
        flip_probability=0.05,
        max_shift=1,
        seed=201,
    )
    evaluator.evaluate_sequences(
        name="T7: sequences, strong noise",
        sequence_count=500,
        flip_probability=0.10,
        max_shift=1,
        seed=202,
    )
    evaluator.evaluate_sequences(
        name="T8: sequences, very strong noise",
        sequence_count=500,
        flip_probability=0.15,
        max_shift=1,
        seed=203,
    )

    print("\nTie-breaking test:")
    example_predictions = np.array([LEFT, LEFT, LEFT, RIGHT, RIGHT, RIGHT])
    example_probabilities = np.array(
        [
            [0.05, 0.70, 0.20, 0.05],
            [0.05, 0.65, 0.25, 0.05],
            [0.05, 0.60, 0.30, 0.05],
            [0.05, 0.20, 0.70, 0.05],
            [0.05, 0.25, 0.65, 0.05],
            [0.05, 0.30, 0.60, 0.05],
        ]
    )
    tie_decision = majority_vote(
        predicted_classes=example_predictions,
        probabilities=example_probabilities,
        previous_decision=FORWARD,
    )
    print("Predictions:", [LABEL_NAMES[int(x)] for x in example_predictions])
    print("Final decision after tie-breaking:", LABEL_NAMES[tie_decision])

    print("\nConfusion matrix for T4:")
    matrix = evaluator.confusion_matrix(y_cm, y_pred_cm)
    evaluator.print_confusion_matrix(matrix)

    npz_path = output_dir / "quantized_reference_model.npz"
    vh_path = output_dir / "mlp_params.vh"

    export_quantized_model_npz(
        model=model,
        standardizer=standardizer,
        filename=npz_path,
        fractional_bits=8,
    )
    export_verilog_header(model=model, filename=vh_path, fractional_bits=8)

    print(f"\nSaved fixed-point NPZ model: {npz_path}")
    print(f"Saved Verilog parameter header: {vh_path}")


if __name__ == "__main__":
    main()
