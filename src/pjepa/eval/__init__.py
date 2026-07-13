"""Evaluation utilities: metrics, bootstrap CI, statistical tests, plotting.

The subpackage is organised into five complementary modules:

* :mod:`pjepa.eval.metrics` — classification and continual-learning
  metrics (accuracy, mean per-class accuracy, forgetting, BWT, FWT).
* :mod:`pjepa.eval.bootstrap` — paired percentile bootstrap CI for
  paired score comparisons.
* :mod:`pjepa.eval.stats` — Wilcoxon signed-rank significance test
  (with sign-permutation fallback) and Bonferroni correction.
* :mod:`pjepa.eval.plots` — radar, heatmap, and SVG-fallback plotting
  helpers used by the TU SOTA experiment.
* :mod:`pjepa.eval.style` — publication-quality rcParams presets.
* :mod:`pjepa.eval.aggregate` — the Phase 12 result aggregator that
  flattens per-experiment outputs into a release-quality table.
"""

from __future__ import annotations

from pjepa.eval.aggregate import (
    AggregatedRow,
    AggregationResult,
    aggregate_all,
    build_rows_from_ablation,
    build_rows_from_cl,
    build_rows_from_decoupling,
    build_rows_from_ogb,
    build_rows_from_sensitivity,
    build_rows_from_tu,
    build_summary_rows,
    coerce_float,
    coerce_int,
    content_hash,
    default_source_paths,
    first_existing_path,
    format_metric_float,
    merge_rows,
    read_csv_rows,
    render_summary_markdown,
    write_artifacts,
)
from pjepa.eval.bootstrap import BootstrapCI, paired_bootstrap_ci
from pjepa.eval.metrics import (
    accuracy,
    backward_transfer,
    forgetting_rate,
    forward_transfer,
    mean_per_class_accuracy,
)
from pjepa.eval.plots import plot_heatmap, plot_radar, render_svg_fallback
from pjepa.eval.stats import bonferroni_correction, wilcoxon_signed_rank
from pjepa.eval.style import (
    PUBLICATION_COLOR_PALETTE,
    PUBLICATION_DPI,
    PUBLICATION_FIGSIZE,
    color_for,
    set_publication_style,
)

__all__ = [
    "PUBLICATION_COLOR_PALETTE",
    "PUBLICATION_DPI",
    "PUBLICATION_FIGSIZE",
    "AggregatedRow",
    "AggregationResult",
    "BootstrapCI",
    "accuracy",
    "aggregate_all",
    "backward_transfer",
    "bonferroni_correction",
    "build_rows_from_ablation",
    "build_rows_from_cl",
    "build_rows_from_decoupling",
    "build_rows_from_ogb",
    "build_rows_from_sensitivity",
    "build_rows_from_tu",
    "build_summary_rows",
    "coerce_float",
    "coerce_int",
    "content_hash",
    "default_source_paths",
    "first_existing_path",
    "forgetting_rate",
    "format_metric_float",
    "forward_transfer",
    "mean_per_class_accuracy",
    "merge_rows",
    "paired_bootstrap_ci",
    "plot_heatmap",
    "plot_radar",
    "read_csv_rows",
    "render_summary_markdown",
    "render_svg_fallback",
    "set_publication_style",
    "wilcoxon_signed_rank",
    "write_artifacts",
]
