"""Persistent-JEPA (pjepa).

A production-grade implementation of the Persistent Graph World Model for
continual developmental learning, described in the paper draft at
``docs/paper/paper.md``.

The package is organised into the following subpackages:

* ``graphs`` — typed attributed graphs, persistent state, working graph.
* ``encoders`` — Euclidean MPNN, hyperbolic, dual-geometric, JEPA predictor.
* ``retrieval`` — submodular greedy working-graph retrieval.
* ``rewriting`` — hyperedge-replacement grammar, bisimulation metric,
  four-conditions acceptance criterion, DPO rewriting.
* ``scheduler`` — PPO scheduler, replay buffer, sleep-cycle cadence.
* ``objectives`` — the unified free-energy functional and its components.
* ``dynamics`` — the evolution operator and its fixed-point/contraction analysis.
* ``utils`` — cross-cutting utilities (seeding, logging helpers, monitoring).
* ``perf`` — performance infrastructure (compile, autocast, EMA, fused ops).
* ``cli`` — the Typer-based command-line application.
* ``training`` — pretraining, supervised training, evaluation, ablations.
* ``data`` — dataset loaders (TUDataset, OGB) and continual-learning splits.
* ``baselines`` — re-implementations of published baselines for comparison.
* ``eval`` — metrics, bootstrap confidence intervals, significance tests,
  table and figure generation.
* ``augmentations`` — graph augmentations used in self-supervised training.

The public API is what is re-exported here. Internal helpers are kept
module-private and not listed in ``__all__``.
"""

from __future__ import annotations

__version__ = "0.0.1"

__all__ = ["__version__"]