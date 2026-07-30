"""Shared prediction engine.

Two building blocks every game module can lean on:

* ``monte_carlo`` — estimate the probability of an event by sampling a trial
  function many times. Use it whenever exact combinatorics is impractical.
* probability/EV helpers — small utilities for working with discrete
  distributions returned by the exact game models.

The engine is pure-Python and deterministic when you pass a ``seed``, which
keeps tests reproducible.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Hashable, Mapping

# A discrete distribution maps each outcome to its probability.
Distribution = Mapping[Hashable, float]


@dataclass(frozen=True)
class MonteCarloResult:
    """Outcome of a Monte-Carlo estimate.

    Attributes:
        probability: fraction of trials in which the event occurred.
        trials: number of trials run.
        hits: number of trials in which the event occurred.
        std_error: standard error of ``probability`` (binomial approximation).
    """

    probability: float
    trials: int
    hits: int
    std_error: float

    def confidence_interval(self, z: float = 1.96) -> tuple[float, float]:
        """Return an approximate (lo, hi) interval; default z is ~95%."""
        margin = z * self.std_error
        return (max(0.0, self.probability - margin),
                min(1.0, self.probability + margin))


def monte_carlo(
    trial: Callable[[random.Random], bool],
    trials: int = 100_000,
    seed: int | None = None,
) -> MonteCarloResult:
    """Estimate P(event) by running ``trial`` ``trials`` times.

    Args:
        trial: a function that takes a ``random.Random`` and returns True when
            the event of interest occurred in that single simulated trial.
        trials: number of independent trials to run.
        seed: optional seed for reproducibility.

    Returns:
        A :class:`MonteCarloResult` with the estimated probability and its
        standard error.
    """
    if trials <= 0:
        raise ValueError("trials must be positive")

    rng = random.Random(seed)
    hits = sum(1 for _ in range(trials) if trial(rng))
    p = hits / trials
    std_error = math.sqrt(p * (1.0 - p) / trials)
    return MonteCarloResult(probability=p, trials=trials, hits=hits,
                            std_error=std_error)


def normalize(counts: Mapping[Hashable, float]) -> dict[Hashable, float]:
    """Turn raw counts/weights into a probability distribution summing to 1."""
    total = sum(counts.values())
    if total <= 0:
        raise ValueError("counts must sum to a positive number")
    return {outcome: value / total for outcome, value in counts.items()}


def distribution_from_samples(samples) -> dict[Hashable, float]:
    """Build an empirical distribution from an iterable of observed outcomes."""
    return normalize(Counter(samples))


def expected_value(dist: Distribution) -> float:
    """Expected value of a distribution whose outcomes are numeric."""
    return sum(float(outcome) * prob for outcome, prob in dist.items())


def variance(dist: Distribution) -> float:
    """Variance of a distribution whose outcomes are numeric."""
    mean = expected_value(dist)
    return sum(prob * (float(outcome) - mean) ** 2 for outcome, prob in dist.items())
