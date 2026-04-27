"""
Main experiment for the Define-stage reference model.

Run from this directory:
    python main.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from config import FORWARD, LABEL_NAMES, LEFT, RIGHT
from dataset_generator import DatasetGenerator
from decision_buffer import majority_vote
from evaluator import Evaluator
from experiment_config import load_experiment_config
from feature_extractor import GeometricFeatureExtractor
from fixed_point import export_quantized_model_npz, export_verilog_header
from mlp_classifier import MLPClassifier, Standardizer
from patterns import print_base_patterns


def main() -> None:
    cfg = load_experiment_config()

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    print("Base 8x8 patterns:\n")
    print_base_patterns()

    extractor = GeometricFeatureExtractor()

    train_generator = DatasetGenerator(
        extractor=extractor,
        seed=cfg.seed,
        max_shift=cfg.train_max_shift,
        flip_probability=cfg.train_flip_probability,
    )

    print("Generating training dataset...")
    X_train, y_train = train_generator.make_dataset(samples_per_class=cfg.train_samples_per_class)

    standardizer = Standardizer.fit(X_train)
    X_train_norm = standardizer.transform(X_train)

    print("\nTraining lightweight MLP:")
    model = MLPClassifier(
        input_dim=extractor.feature_count,
        hidden_dim=cfg.hidden_dim,
        output_dim=4,
        learning_rate=cfg.learning_rate,
        seed=cfg.seed,
    )
    model.fit(X_train_norm, y_train, epochs=cfg.train_epochs, verbose=cfg.verbose_training)

    if cfg.extra_training_rounds > 0:
        print("\nExtra hardening rounds:")
        for round_idx in range(cfg.extra_training_rounds):
            round_seed = cfg.seed + 1000 + round_idx
            extra_generator = DatasetGenerator(
                extractor=extractor,
                seed=round_seed,
                max_shift=cfg.extra_max_shift,
                flip_probability=cfg.extra_flip_probability,
            )
            X_extra, y_extra = extra_generator.make_dataset(
                samples_per_class=cfg.extra_samples_per_class
            )
            model.fit(
                standardizer.transform(X_extra),
                y_extra,
                epochs=cfg.extra_epochs_per_round,
                verbose=cfg.verbose_training,
            )
            print(
                f"  round {round_idx + 1}/{cfg.extra_training_rounds}: "
                f"samples/class={cfg.extra_samples_per_class}, "
                f"noise={cfg.extra_flip_probability}, epochs={cfg.extra_epochs_per_round}"
            )

    evaluator = Evaluator(model=model, standardizer=standardizer, extractor=extractor)

    print("\nFrame-level tests:")
    evaluator.evaluate_clean_patterns()
    evaluator.evaluate_shift_test(max_shift=cfg.train_max_shift)

    _, _, _ = evaluator.evaluate_noisy_frames(
        name="T3: light noise, shifted signs",
        samples_per_class=cfg.eval_samples_per_class,
        flip_probability=0.05,
        max_shift=cfg.train_max_shift,
        seed=101,
    )

    X_cm, y_cm, y_pred_cm = evaluator.evaluate_noisy_frames(
        name="T4: strong noise, shifted signs",
        samples_per_class=cfg.eval_samples_per_class,
        flip_probability=0.10,
        max_shift=cfg.train_max_shift,
        seed=102,
    )

    _, _, _ = evaluator.evaluate_noisy_frames(
        name="T5: very strong noise",
        samples_per_class=cfg.eval_samples_per_class,
        flip_probability=0.15,
        max_shift=cfg.train_max_shift,
        seed=103,
    )

    print("\nSequence-level tests with majority voting:")
    evaluator.evaluate_sequences(
        name="T6: sequences, light noise",
        sequence_count=cfg.eval_sequence_count,
        flip_probability=0.05,
        max_shift=cfg.train_max_shift,
        seed=201,
    )
    evaluator.evaluate_sequences(
        name="T7: sequences, strong noise",
        sequence_count=cfg.eval_sequence_count,
        flip_probability=0.10,
        max_shift=cfg.train_max_shift,
        seed=202,
    )
    evaluator.evaluate_sequences(
        name="T8: sequences, very strong noise",
        sequence_count=cfg.eval_sequence_count,
        flip_probability=0.15,
        max_shift=cfg.train_max_shift,
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
        fractional_bits=cfg.fractional_bits,
    )
    export_verilog_header(model=model, filename=vh_path, fractional_bits=cfg.fractional_bits)

    print(f"\nSaved fixed-point NPZ model: {npz_path}")
    print(f"Saved Verilog parameter header: {vh_path}")


if __name__ == "__main__":
    main()
