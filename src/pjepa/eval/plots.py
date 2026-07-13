"""Plotting helpers for the TU SOTA experiment.

These produce plan-compliant radar and heatmap charts using
``matplotlib``. The functions are deliberately small so they can
be exercised in unit tests with an ``Agg`` backend (no display
required).

If ``matplotlib`` is unavailable, :func:`render_svg_fallback`
produces a minimal SVG by hand so the pipeline still completes on
minimal installs; callers can detect this by checking the file
extension.

## Architecture

* :func:`plot_radar` / :func:`plot_heatmap` delegate to the
  ``Agg`` backend (``matplotlib.use("Agg")`` is set the first
  time :func:`import_matplotlib` runs).
* :func:`plot_radar_bar` is a fallback grouped-bar chart used
  when fewer than three datasets are available — the radar chart
  needs at least three axes for the polygon shape to be readable.
* :func:`render_svg_fallback` does not import matplotlib at
  all; it writes the SVG by hand.

## Complexity

The plot helpers are dominated by I/O (one ``fig.savefig``
call per function). The radar plot materialises an
``[N, D]`` ``cosine / sin`` lookup; the heatmap materialises an
``[R, C]`` masked array.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pjepa.exceptions import ConfigError

__all__ = [
    "ensure_parent_directory",
    "import_matplotlib",
    "normalise_radar_values",
    "plot_heatmap",
    "plot_radar",
    "plot_radar_bar",
    "render_svg_fallback",
]


def ensure_parent_directory(path: Path) -> None:
    """Create ``path.parent`` if it does not yet exist.

    Args:
        path: The destination path whose parent should exist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)


def import_matplotlib() -> tuple[Any, Any]:
    """Lazily import matplotlib and select the ``Agg`` backend.

    Returns:
        A ``(matplotlib, pyplot)`` tuple.

    Raises:
        ConfigError: When ``matplotlib`` is not installed.
    """
    try:
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.use("Agg")
        return matplotlib, plt
    except ImportError as exc:
        raise ConfigError(
            "plot_radar/plot_heatmap: matplotlib is required; install with `pip install matplotlib`"
        ) from exc


def normalise_radar_values(values: Sequence[float]) -> list[float]:
    """Normalise ``values`` to ``[0, 1]`` for a radar plot.

    The normalisation rescales the per-method values into ``[0, 1]``
    independently of their absolute range so multiple methods can be
    plotted on the same axes. When the min and max are within
    ``1e-12`` of each other the function returns ``[0.5]`` so the
    plot has a visible point rather than collapsing to zero.

    Args:
        values: Per-axis values.

    Returns:
        A list of ``[0, 1]`` floats with the same length.
    """
    lo = min(values)
    hi = max(values)
    if hi - lo < 1e-12:
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def plot_radar_bar(
    method_means: Mapping[str, Sequence[float]],
    datasets: Sequence[str],
    path: Path,
    title: str,
    plt: Any,
) -> Path:
    """Fallback grouped-bar chart when fewer than 3 datasets are available.

    Args:
        method_means: A mapping ``{method_name: [mean_per_dataset]}``.
        datasets: Dataset names (one per bar group).
        path: Destination file path.
        title: Plot title.
        plt: The matplotlib pyplot module (already imported via
          :func:`import_matplotlib`).

    Returns:
        The destination :class:`Path`.
    """
    import numpy as np

    methods = list(method_means.keys())
    fig, ax = plt.subplots(figsize=(max(4.0, 0.6 * len(datasets) + 2), 4.0))
    x = np.arange(len(datasets))
    width = 0.8 / max(len(methods), 1)
    for i, method in enumerate(methods):
        values = [
            float(v) if not (isinstance(v, float) and math.isnan(v)) else 0.0
            for v in method_means[method]
        ]
        ax.bar(x + i * width, values, width=width, label=method)
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(list(datasets))
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_radar(
    method_means: Mapping[str, Sequence[float]],
    datasets: Sequence[str],
    output_path: str | Path,
    title: str = "TU SOTA radar",
) -> Path:
    """Plot a radar chart of mean per-class accuracy across datasets.

    Args:
        method_means: A mapping ``{method_name: [mean_per_dataset]}``.
          The order of the inner list must match ``datasets``.
        datasets: Dataset names (one per axis). Must contain three
          or more entries; the helper falls back to
          :func:`plot_radar_bar` for shorter lists.
        output_path: Destination file path (``.png`` or ``.svg``).
        title: Plot title.

    Returns:
        The path of the rendered file.

    Raises:
        ConfigError: If ``method_means`` is empty or its inner
          vectors have a different length than ``datasets``.
    """
    if not method_means:
        raise ConfigError("plot_radar: at least one method is required")
    for name, values in method_means.items():
        if len(values) != len(datasets):
            raise ConfigError(
                f"plot_radar: method {name!r} has {len(values)} values, expected {len(datasets)}"
            )
    path = Path(output_path)
    ensure_parent_directory(path)
    _, plt = import_matplotlib()
    if len(datasets) < 3:
        return plot_radar_bar(method_means, datasets, path, title=title, plt=plt)
    n_axes = len(datasets)
    angles = [n * 2.0 * math.pi / n_axes for n in range(n_axes)]
    angles += angles[:1]
    fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(6.0, 6.0))
    for name, values in method_means.items():
        normalised = normalise_radar_values(list(values))
        series = list(normalised) + [normalised[0]]
        ax.plot(angles, series, label=name)
        ax.fill(angles, series, alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(list(datasets))
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"])
    ax.set_title(title)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_heatmap(
    matrix: Sequence[Sequence[float]],
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    output_path: str | Path,
    title: str = "TU SOTA heatmap",
) -> Path:
    """Plot a heatmap of mean accuracy.

    Args:
        matrix: A 2-D matrix ``[n_rows][n_cols]`` of values.
          ``NaN`` entries are rendered as transparent cells.
        row_labels: Labels for the rows (datasets).
        col_labels: Labels for the columns (methods).
        output_path: Destination file path.
        title: Plot title.

    Returns:
        The path of the rendered file.
    """
    path = Path(output_path)
    ensure_parent_directory(path)
    _, plt = import_matplotlib()
    import numpy as np

    arr = np.array(matrix, dtype=float)
    fig, ax = plt.subplots(
        figsize=(max(4.0, 0.6 * len(col_labels) + 2), max(3.0, 0.5 * len(row_labels) + 2))
    )
    masked = np.ma.masked_invalid(arr)
    base_cmap = plt.get_cmap("viridis")
    cmap = (
        base_cmap.with_extremes(bad="white")
        if hasattr(base_cmap, "with_extremes")
        else base_cmap.copy()
    )
    if not hasattr(base_cmap, "with_extremes"):
        cmap.set_bad("white")
    im = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(list(col_labels), rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(list(row_labels))
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            value = arr[i, j]
            if math.isnan(value):
                continue
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white", fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="mean accuracy")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def render_svg_fallback(
    width: int,
    height: int,
    output_path: str | Path,
    background: str = "white",
) -> Path:
    """Render a minimal SVG of the given size; used when matplotlib is unavailable.

    Args:
        width: Pixel width.
        height: Pixel height.
        output_path: Destination ``.svg`` path.
        background: Fill colour for the background rectangle.

    Returns:
        The path of the rendered file.

    Raises:
        ConfigError: When ``output_path`` does not end with ``.svg``.
    """
    path = Path(output_path)
    ensure_parent_directory(path)
    if path.suffix.lower() != ".svg":
        raise ConfigError(f"render_svg_fallback: expected .svg path; got {path}")
    svg = (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' "
        f"viewBox='0 0 {width} {height}'>"
        f"<rect width='100%' height='100%' fill='{background}'/>"
        f"<text x='50%' y='50%' text-anchor='middle' font-family='sans-serif' "
        f"font-size='14' fill='black'>matplotlib unavailable</text>"
        f"</svg>"
    )
    path.write_text(svg, encoding="utf-8")
    return path
