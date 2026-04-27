"""Runtime experiment configuration loaded from environment variables and .env file."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from config import DEFAULT_SEED


@dataclass(frozen=True)
class ExperimentConfig:
    train_samples_per_class: int = 2000
    train_flip_probability: float = 0.05
    train_max_shift: int = 1
    hidden_dim: int = 12
    learning_rate: float = 0.08
    train_epochs: int = 1000

    extra_training_rounds: int = 0
    extra_samples_per_class: int = 1000
    extra_flip_probability: float = 0.12
    extra_max_shift: int = 1
    extra_epochs_per_round: int = 250

    seed: int = DEFAULT_SEED
    eval_samples_per_class: int = 250
    eval_sequence_count: int = 500
    fractional_bits: int = 8
    verbose_training: bool = True



def _to_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _load_dotenv(dotenv_path: Path = Path(".env")) -> None:
    """Load .env key/value pairs into os.environ without overriding exported env vars."""
    if not dotenv_path.exists():
        return

    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue

        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip()

        if value and len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]

        os.environ.setdefault(key, value)


def load_experiment_config() -> ExperimentConfig:
    _load_dotenv()

    def env_int(name: str, default: int) -> int:
        return int(os.getenv(name, default))

    def env_float(name: str, default: float) -> float:
        return float(os.getenv(name, default))

    def env_bool(name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return _to_bool(raw)

    return ExperimentConfig(
        train_samples_per_class=env_int("TRAIN_SAMPLES_PER_CLASS", 2000),
        train_flip_probability=env_float("TRAIN_FLIP_PROBABILITY", 0.05),
        train_max_shift=env_int("TRAIN_MAX_SHIFT", 1),
        hidden_dim=env_int("MLP_HIDDEN_DIM", 12),
        learning_rate=env_float("MLP_LEARNING_RATE", 0.08),
        train_epochs=env_int("TRAIN_EPOCHS", 1000),
        extra_training_rounds=env_int("EXTRA_TRAINING_ROUNDS", 0),
        extra_samples_per_class=env_int("EXTRA_SAMPLES_PER_CLASS", 1000),
        extra_flip_probability=env_float("EXTRA_FLIP_PROBABILITY", 0.12),
        extra_max_shift=env_int("EXTRA_MAX_SHIFT", 1),
        extra_epochs_per_round=env_int("EXTRA_EPOCHS_PER_ROUND", 250),
        seed=env_int("EXPERIMENT_SEED", DEFAULT_SEED),
        eval_samples_per_class=env_int("EVAL_SAMPLES_PER_CLASS", 250),
        eval_sequence_count=env_int("EVAL_SEQUENCE_COUNT", 500),
        fractional_bits=env_int("FRACTIONAL_BITS", 8),
        verbose_training=env_bool("VERBOSE_TRAINING", True),
    )
