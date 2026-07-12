"""Statistical significance tests for method comparisons."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["bonferroni_correction", "wilcoxon_signed_rank"]


def wilcoxon_signed_rank(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
) -> float:
    """Compute the Wilcoxon signed-rank two-sided p-value.

    Falls back to a simple permutation test when SciPy is unavailable
    or when the sample size is too small for the asymptotic
    approximation.

    Args:
        scores_a: Per-seed scores for method A.
        scores_b: Per-seed scores for method B.

    Returns:
        The p-value in [0, 1].
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(
            f"wilcoxon_signed_rank: lengths must match; got {len(scores_a)} vs {len(scores_b)}"
        )
    diffs = np.asarray(scores_a, dtype=np.float64) - np.asarray(scores_b, dtype=np.float64)
    diffs = diffs[diffs != 0]
    if diffs.size == 0:
        return 1.0
    try:
        from scipy.stats import wilcoxon  # type: ignore[import-not-found]

        _, p = wilcoxon(diffs, zero_method="wilcox", correction=False, alternative="two-sided")
        return float(p)
    except (ImportError, ValueError):
        return _permutation_pvalue(diffs)


def _permutation_pvalue(diffs: np.ndarray, n_resamples: int = 10_000, seed: int = 0) -> float:
    """A permutation-test fallback for the Wilcoxon signed-rank p-value."""
    rng = np.random.default_rng(seed)
    observed = float(np.abs(diffs).sum())
    count = 0
    for _ in range(n_resamples):
        signs = rng.choice([-1.0, 1.0], size=diffs.shape[0])
        permuted = float(np.abs(diffs * signs).sum())
        if permuted >= observed:
            count += 1
    return float((count + 1) / (n_resamples + 1))


def bonferroni_correction(p_values: Sequence[float]) -> list[float]:
    """Apply the Bonferroni correction to a list of p-values.

    Args:
        p_values: The unadjusted p-values.

    Returns:
        The adjusted p-values, capped at 1.0.
    """
    if not p_values:
        return []
    m = len(p_values)
    return [min(1.0, p * m) for p in p_values]
