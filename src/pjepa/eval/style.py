"""Publication-quality plot styling for pjepa figures.

Every figure in the codebase goes through :func:`set_publication_style`
to guarantee consistent typography, dpi, and colour treatment across
the three Phase-4 validation experiments and downstream reports.

## Colour palette

The deterministic six-colour palette :data:`_COLOR_PALETTE` is shared
across every publication figure; pick a colour by calling
:func:`color_for` with a sequential index so individual plots stay
comparable across reports.

## Style application

:func:`set_publication_style` calls
:func:`matplotlib.pyplot.style.use` to install the ``ggplot`` base
style and then layers the project-specific rcParams on top. The
``ggplot`` style install can raise when the user's matplotlib
configuration is missing the bundled style sheet (rare but observed
on minimal Linux containers); we handle the specific
``OSError`` and ``ImportError`` that the install can produce.
"""

from __future__ import annotations

import logging
from typing import Final

import matplotlib as mpl
import matplotlib.pyplot as plt

from pjepa.logging_setup import get_logger

__all__ = [
    "PUBLICATION_COLOR_PALETTE",
    "PUBLICATION_DPI",
    "PUBLICATION_FIGSIZE",
    "color_for",
    "set_publication_style",
]

PUBLICATION_DPI: Final[int] = 300
"""Default DPI for publication-quality figures."""

PUBLICATION_FIGSIZE: Final[tuple[float, float]] = (6.0, 4.0)
"""Default figure size for publication-quality figures (inches)."""

PUBLICATION_COLOR_PALETTE: Final[tuple[str, ...]] = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
)
"""Six-colour qualitative palette used by every pjepa publication figure.

Exposed as a public constant so individual plotting helpers can
``PUBLICATION_COLOR_PALETTE[i]`` without going through the
wrapping :func:`color_for` helper.
"""


def set_publication_style() -> None:
    """Apply the pjepa publication-quality style to matplotlib.

    The function calls ``matplotlib.pyplot.style.use('ggplot')``
    (with a narrow exception handler for the rare ``OSError`` /
    ``ImportError`` on minimal installs) and then layers project
    specific rcParams on top:

    * ``figure.dpi`` = 100 (preview) and ``savefig.dpi`` =
      :data:`PUBLICATION_DPI` (publication export).
    * ``savefig.bbox`` = ``'tight'`` with 0.1" padding so the saved
      file never clips axis labels.
    * ``font.family`` = ``'DejaVu Sans'`` (bundled with every
      matplotlib install) at 10pt base with 12pt bold titles.
    * ``axes.spines.top`` / ``axes.spines.right`` turned off so the
      default look matches ggplot's "minimal" border.
    * ``axes.grid`` enabled with a soft 30% transparent grid.
    * ``legend.frameon`` turned off so legends blend into the figure.
    * ``axes.prop_cycle`` set to :data:`_COLOR_PALETTE` so the
      first six series auto-assigned by matplotlib match the
      project palette.

    The function is idempotent: subsequent calls overwrite the
    previously configured rcParams without raising.

    Returns:
        None.
    """
    log = get_logger(__name__)
    try:
        plt.style.use("ggplot")
    except (OSError, ImportError, KeyError) as exc:
        log.debug(
            "ggplot style unavailable; rcParams only",
            extra={"event": "style.use_failed", "error": str(exc)},
        )
    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": PUBLICATION_DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.labelweight": "regular",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 1.5,
            "lines.markersize": 5,
            "figure.figsize": PUBLICATION_FIGSIZE,
            "figure.autolayout": True,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "errorbar.capsize": 3.0,
            "axes.prop_cycle": mpl.cycler(color=list(PUBLICATION_COLOR_PALETTE)),
        }
    )


def color_for(index: int) -> str:
    """Return the publication-palette colour at position ``index``.

    Args:
        index: Position into the publication colour palette; values
          outside the range wrap around modulo the palette length.

    Returns:
        A matplotlib-compatible hex colour string.

    Example:
        >>> color_for(0)
        '#1f77b4'
        >>> color_for(1)
        '#ff7f0e'
    """
    if not PUBLICATION_COLOR_PALETTE:
        return "#1f77b4"
    return PUBLICATION_COLOR_PALETTE[int(index) % len(PUBLICATION_COLOR_PALETTE)]


# Silence linters that flag the ``logging`` import as unused; it is
# kept for parity with the other module-level configure-style helpers.
_ = logging
