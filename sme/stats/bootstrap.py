"""Paired bootstrap confidence intervals (Efron & Tibshirani 1993).

Standard non-parametric CI for per-question paired differences.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BootstrapCIResult:
    """Result of a paired bootstrap CI computation."""

    mean_diff: float
    ci_lower: float
    ci_upper: float
    n_bootstrap: int
    confidence_level: float
    p_value_approx: float  # fraction of bootstrap diffs crossing zero


def paired_bootstrap_ci(
    scores_a: list[float],
    scores_b: list[float],
    *,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> BootstrapCIResult:
    """Paired bootstrap CI on per-item score differences.

    Args:
        scores_a: Per-item scores for condition A
        scores_b: Per-item scores for condition B (same length, paired)
        n_bootstrap: Number of bootstrap resamples
        confidence: Confidence level (default 0.95 for 95% CI)
        seed: RNG seed for reproducibility

    Returns:
        BootstrapCIResult with mean difference and CI bounds
    """
    assert len(scores_a) == len(scores_b), "Paired scores must be same length"
    rng = np.random.RandomState(seed)
    a = np.array(scores_a, dtype=float)
    b = np.array(scores_b, dtype=float)
    diffs = a - b
    observed_mean = float(np.mean(diffs))
    n = len(diffs)

    # Vectorized resample in chunks. Drawing all (n_bootstrap, n) indices
    # at once would peak at ~1.6 GB for n_bootstrap=n=10000; chunking caps
    # the live allocation at O(chunk_size * n). Legacy RandomState.randint
    # consumes the same byte stream regardless of how the draws are
    # chunked, so the chunked path is bit-identical to a single draw for
    # any fixed seed.
    chunk_size = 1000
    boot_means = np.empty(n_bootstrap)
    for i in range(0, n_bootstrap, chunk_size):
        current_chunk = min(chunk_size, n_bootstrap - i)
        indices = rng.randint(0, n, size=(current_chunk, n))
        boot_means[i : i + current_chunk] = diffs[indices].mean(axis=1)

    alpha = 1.0 - confidence
    ci_lower = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    if observed_mean >= 0:
        p_approx = float(np.mean(boot_means < 0))
    else:
        p_approx = float(np.mean(boot_means > 0))
    p_approx = min(2 * p_approx, 1.0)

    return BootstrapCIResult(
        mean_diff=observed_mean,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence,
        p_value_approx=p_approx,
    )
