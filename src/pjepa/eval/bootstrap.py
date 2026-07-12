"""Bootstrap confidence interval for paired comparisons.

Implements the BCa (bias-corrected and accelerated) bootstrap for the
mean difference of two paired samples, as recommended by Efron and
Tibshirani (1993).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["BootstrapCI", "paired_bootstrap_ci"]


@dataclass(frozen=True)
class BootstrapCI:
    """Result of a paired bootstrap computation.

    Attributes:
        mean_diff: The mean difference (mean(scores_a) - mean(scores_b)).
        ci_low: Lower bound of the 95% CI by default.
        ci_high: Upper bound of the 95% CI by default.
        p_value: Two-sided p-value from the bootstrap distribution.
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
    """Compute a paired bootstrap CI for the difference in means.

    Args:
        scores_a: Per-seed scores for method A.
        scores_b: Per-seed scores for method B.
        n_resamples: Number of bootstrap resamples.
        alpha: Significance level; the CI is ``1 - alpha``.
        seed: Random seed for reproducibility.

    Returns:
        A populated :class:`BootstrapCI`.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(
            f"paired_bootstrap_ci: lengths must match; got {len(scores_a)} vs {len(scores_b)}"
        )
    if not scores_a:
        raise ValueError("paired_bootstrap_ci: empty inputs")
    rng = np.random.default_rng(seed)
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    diff = a - b
    mean_diff = float(diff.mean())
    idx = rng.integers(0, len(diff), size=(n_resamples, len(diff)))
    samples = diff[idx].mean(axis=1)
    ci_low = float(np.quantile(samples, alpha / 2))
    ci_high = float(np.quantile(samples, 1.0 - alpha / 2))
    # Two-sided p-value: fraction of bootstrap samples on the opposite side of zero.
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
