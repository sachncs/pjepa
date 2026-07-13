"""Bootstrap confidence intervals for paired score comparisons.

The standard error on a mean difference is hard to estimate from
short score series (TU SOTA scores are typically aggregated over
3-10 seeds). The classical percentile bootstrap sidesteps the
closed-form problem by resampling the per-seed score differences
with replacement, recomputing the mean of each resample, and reading
off the empirical quantile intervals.

The implementation here is the **percentile** bootstrap: the
``(α/2, 1 - α/2)`` quantiles of the bootstrap distribution form the
confidence interval, and the two-sided p-value is the fraction of
resamples that fall on the opposite side of zero from the observed
mean. This is *not* the full BCa (bias-corrected and accelerated)
bootstrap of Efron & Tibshirani (1993); it omits the bias and
acceleration corrections. The trade-off is a smaller library surface
(no ``scipy`` dependency, no jackknife pass) at the cost of slightly
wider intervals when the underlying distribution is skewed. The
appendix of the paper draft recommends the percentile variant for
``n_seeds <= 10`` where BCa cannot be estimated reliably anyway.

## Complexity

Let ``n = len(scores_a)`` (equal to ``len(scores_b)`` by contract) and
``R`` the configured ``n_resamples``. The procedure is
``O(R + n)``: one ``O(n)`` array for the per-pair differences, one
``O(R * n)`` resample-and-mean matrix, and ``O(R log R)`` to obtain
the quantiles. Memory is ``O(R * n)`` floats; with the default
``R = 10_000`` and ``n <= 100`` that is well under 1 MiB per call.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["BootstrapCI", "paired_bootstrap_ci"]


@dataclass(frozen=True)
class BootstrapCI:
    """Result of a paired bootstrap computation.

    Attributes:
        mean_diff: The mean difference (``mean(scores_a) - mean(scores_b)``).
        ci_low: Lower bound of the confidence interval at the configured
          ``1 - alpha`` level.
        ci_high: Upper bound of the confidence interval at the configured
          ``1 - alpha`` level.
        p_value: Two-sided p-value: fraction of bootstrap-resampled means
          that fall on the opposite side of zero from ``mean_diff``.
    """

    mean_diff: float
    ci_low: float
    ci_high: float
    p_value: float


def paired_bootstrap_ci(
    scores_a: list[float],
    scores_b: list[float],
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> BootstrapCI:
    """Compute a paired percentile bootstrap CI on the mean difference.

    The function enforces the paired structure of the inputs: the two
    arrays must share length and the per-index difference is computed
    elementwise before resampling. Resampling is performed by drawing
    ``n_resamples`` index arrays of shape ``[R, n]`` from a
    :class:`numpy.random.Generator`, then materialising the
    bootstrap distribution of the mean of the differences. This
    matches the canonical resampling-with-replacement recipe.

    Args:
        scores_a: Per-seed scores for method A, e.g. ``[0.82, 0.81, ...]``.
        scores_b: Per-seed scores for method B with the same length.
        n_resamples: Number of bootstrap resamples. ``10_000`` is
          ample for ``n_seeds <= 100``.
        alpha: Significance level; the returned interval has nominal
          coverage ``1 - alpha``.
        seed: Seed forwarded to :class:`numpy.random.default_rng` so
          the CI is reproducible across calls.

    Returns:
        A populated :class:`BootstrapCI`. ``ci_low`` / ``ci_high``
        are the ``alpha/2`` and ``1 - alpha/2`` percentiles of the
        bootstrap distribution of ``mean(a - b)``.

    Raises:
        ValueError: When ``scores_a`` and ``scores_b`` have
          mismatching length or are both empty.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(
            f"paired_bootstrap_ci: lengths must match; got {len(scores_a)} vs {len(scores_b)}"
        )
    if not scores_a:
        raise ValueError("paired_bootstrap_ci: empty inputs")
    if n_resamples <= 0:
        raise ValueError(f"paired_bootstrap_ci: n_resamples must be positive; got {n_resamples}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"paired_bootstrap_ci: alpha must be in (0, 1); got {alpha}")
    rng = np.random.default_rng(seed)
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    diff = a - b
    mean_diff = float(diff.mean())
    idx = rng.integers(0, len(diff), size=(n_resamples, len(diff)))
    samples = diff[idx].mean(axis=1)
    ci_low = float(np.quantile(samples, alpha / 2))
    ci_high = float(np.quantile(samples, 1.0 - alpha / 2))
    # Two-sided p-value: fraction of bootstrap samples on the opposite side of zero
    # from the observed mean. This is the standard "sign-flip" two-sided test
    # used by Efron & Tibshirani (1993, §13.4) for the percentile bootstrap.
    if mean_diff >= 0:
        p_value = float((samples < 0).mean())
    else:
        p_value = float((samples > 0).mean())
    return BootstrapCI(
        mean_diff=mean_diff,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
    )
