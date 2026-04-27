"""
Decision buffer and majority-voting logic for the final movement command.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

import numpy as np

from config import BACK, NUM_CLASSES, SEQUENCE_LENGTH


def majority_vote(
    predicted_classes: np.ndarray,
    probabilities: Optional[np.ndarray] = None,
    previous_decision: Optional[int] = None,
    fallback_decision: int = BACK,
) -> int:
    """
    Majority voting with deterministic tie-breaking.

    Tie-breaking policy:
        1. Choose the class with the highest number of votes.
        2. If there is a tie and probabilities are available, choose the class with
           the highest summed probability.
        3. If there is still a tie, keep previous_decision.
        4. If previous_decision is unavailable, use fallback_decision.
    """
    if len(predicted_classes) == 0:
        return int(previous_decision if previous_decision is not None else fallback_decision)

    counts = np.bincount(predicted_classes.astype(np.int64), minlength=NUM_CLASSES)
    max_count = counts.max()
    candidates = np.where(counts == max_count)[0]

    if len(candidates) == 1:
        return int(candidates[0])

    if probabilities is not None:
        probability_sums = probabilities.sum(axis=0)
        candidate_scores = probability_sums[candidates]
        best_score = candidate_scores.max()
        best_candidates = candidates[np.isclose(candidate_scores, best_score)]

        if len(best_candidates) == 1:
            return int(best_candidates[0])

    if previous_decision is not None:
        return int(previous_decision)

    return int(fallback_decision)


@dataclass
class DecisionBuffer:
    """Stores the last N frame decisions and returns a stabilized decision."""

    length: int = SEQUENCE_LENGTH
    fallback_decision: int = BACK
    previous_decision: Optional[int] = None
    decisions: Deque[int] = field(default_factory=deque, init=False)
    probabilities: Deque[np.ndarray] = field(default_factory=deque, init=False)

    def __post_init__(self) -> None:
        self.decisions = deque(maxlen=self.length)
        self.probabilities = deque(maxlen=self.length)

    def clear(self) -> None:
        self.decisions.clear()
        self.probabilities.clear()
        self.previous_decision = None

    def push(self, decision: int, probability: Optional[np.ndarray] = None) -> None:
        self.decisions.append(int(decision))
        if probability is not None:
            self.probabilities.append(np.asarray(probability, dtype=np.float64))

    def is_full(self) -> bool:
        return len(self.decisions) == self.length

    def vote(self) -> int:
        if len(self.probabilities) == len(self.decisions) and len(self.probabilities) > 0:
            probs = np.vstack(list(self.probabilities))
        else:
            probs = None

        decision = majority_vote(
            predicted_classes=np.array(self.decisions, dtype=np.int64),
            probabilities=probs,
            previous_decision=self.previous_decision,
            fallback_decision=self.fallback_decision,
        )
        self.previous_decision = decision
        return decision
