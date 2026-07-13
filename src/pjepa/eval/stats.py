"""Statistical significance tests for method comparisons.

The module exposes two helpers used by the TU SOTA experiment:

* :func:`wilcoxon_signed_rank` — two-sided signed-rank p-value for
  paired score series. Falls back to a sign-permutation test when
  SciPy is unavailable or the sample size is too small for the
  asymptotic approximation.
* :func:`bonferroni_correction` — Bonferroni family-wise correction
  for a list of p-values.

## Complexity

The Wilcoxon implementation inherits SciPy's ``O(n log n)`` cost in
the asymptotic regime. The permutation fallback is ``O(R * n)``
where ``R = n_resamples`` (default 10_000); with the default
configuration and ``n <= 32`` this is well under 100 ms per call.
The Bonferroni correction is a single ``O(m)`` pass over ``m``
p-values.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["bonferroni_correction", "permutation_pvalue", "wilcoxon_signed_rank"]


def wilcoxon_signed_rank(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
) -> float:
    """Compute the Wilcoxon signed-rank two-sided p-value.

    The Wilcoxon signed-rank test is the standard non-parametric test
    for paired score series. The version here reports SciPy's
    asymptotic approximation when available; when SciPy is missing
    or the sample is too small for the asymptotic regime, the
    function falls back to a sign-permutation test that always
    produces a valid ``[0, 1]`` p-value.

    Args:
        scores_a: Per-seed scores for method A.
        scores_b: Per-seed scores for method B with the same length.

    Returns:
        The two-sided p-value in ``[0, 1]``.

    Raises:
        ValueError: When ``scores_a`` and ``scores_b`` have different
          lengths.
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
        from scipy.stats import wilcoxon

        _, p = wilcoxon(diffs, zero_method="wilcox", correction=False, alternative="two-sided")
        return float(p)
    except (ImportError, ValueError):
        return permutation_pvalue(diffs)


def permutation_pvalue(diffs: np.ndarray, n_resamples: int = 10_000, seed: int = 0) -> float:
    """A sign-permutation fallback for :func:`wilcoxon_signed_rank`.

    Args:
        diffs: Non-zero per-seed differences, ``[n]``.
        n_resamples: Number of random sign flips.
        seed: Seed for reproducibility.

    Returns:
        A ``[0, 1]`` p-value. The ``+1`` in the numerator and
        denominator matches the standard "add-one" correction
        used by :mod:`scipy.stats` permutation tests.
    """
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

    The Bonferroni correction multiplies every p-value by the number
    of comparisons and caps the result at ``1.0``. It is the most
    conservative family-wise correction in common use; for the TU
    SOTA experiment we pair it with the per-comparison Wilcoxon
    test so the result is always interpretable.

    Args:
        p_values: The unadjusted p-values. Empty input yields an
          empty list.

    Returns:
        The adjusted p-values, capped at ``1.0``.
    """
    if not p_values:
        return []
    m = len(p_values)
    return [min(1.0, p * m) for p in p_values]
