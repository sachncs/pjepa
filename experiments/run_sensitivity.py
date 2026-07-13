"""Sensitivity sweep for the working-graph budget ``B`` (Phase 11).

Implements the plan's section 2.3: train Persistent-JEPA on PROTEINS
with each ``B ∈ {8, 16, 32, 64, 128, 256}`` for 3 seeds and plot
accuracy vs ``B``.

The path exercises the same machinery as the headline TU SOTA
experiment (DualGeometricEncoder + JEPAPredictor + TargetEncoder +
GreedyRetrieval + FacilityLocationUtility + FourConditions), so the
sensitivity sweep doubles as an end-to-end smoke test of every Phase
1-8 component in a single, configurable setting.

The bootstrap CI for each budget is the standard finite-sample CI
of the mean: ``paired_bootstrap_ci(scores, zeros)`` returns the CI
of the mean directly. The previous implementation bootstrapped the
*deviation from the mean* and re-shifted the CI, which gave an
identical numeric result but obscured the semantics. The new code
makes the CI an actual interval around the mean.

Outputs (plan-compliant):

* ``<output_dir>/tables/sensitivity_B.csv`` — per-(B, seed) rows.
* ``<output_dir>/tables/sensitivity_B_summary.csv`` — per-B summary.
* ``<output_dir>/plots/sensitivity_B.png`` — accuracy vs B with
  log-x axis and bootstrap 95% CI error bars.

Legacy paths retained for compatibility:

* ``<output_dir>/sensitivity_B.csv``.
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch

from pjepa.augmentations.base import AugmentationPipeline, PipelineMode
from pjepa.augmentations.feature import DropFeature
from pjepa.augmentations.structural import DropEdge, DropNode
from pjepa.data.tu import load_tu_dataset
from pjepa.encoders import DualGeometricEncoder, JEPAPredictor, TargetEncoder
from pjepa.eval import (
    color_for,
    mean_per_class_accuracy,
    paired_bootstrap_ci,
    set_publication_style,
)
from pjepa.graphs import PersistentState, TypedAttributedGraph
from pjepa.logging_setup import LogFormat, configure_logging, get_logger
from pjepa.retrieval import FacilityLocationUtility, GreedyRetrieval
from pjepa.rewriting import HRG, accept_candidate
from pjepa.utils.seeding import set_global_seed

__all__ = [
    "DEFAULT_BUDGETS",
    "SensitivityConfig",
    "aggregate_per_budget",
    "build_jepa_augmentation_pipeline",
    "default_smoke_config",
    "encode_and_mean_pool",
    "kfold_indices",
    "pretrain_jepa_one_epoch",
    "render_sensitivity_figure",
    "run_sensitivity",
    "train_one_run",
    "write_table_csv",
]


DEFAULT_BUDGETS: tuple[int, ...] = (8, 16, 32, 64, 128, 256)


@dataclass(frozen=True)
class SensitivityConfig:
    """Configuration for the budget sensitivity sweep.

    Attributes:
        dataset: The TU dataset to sweep over. Default per Phase 11
          plan: ``"PROTEINS"``.
        budgets: Working-graph budgets ``B`` to sweep.
        n_seeds: Seeds per budget.
        n_folds: Cross-validation folds per seed.
        epochs: Training epochs per run.
        learning_rate: Optimiser learning rate.
        batch_size: Mini-batch size.
        hidden_dim: Encoder hidden width.
        num_layers: Encoder depth.
        output_dir: Output directory; ``tables/`` and ``plots/``
          sub-directories are created underneath.
        smoke: When ``True``, the experiment runs the fast smoke
          configuration used by the unit tests.
        bootstrap_resamples: Bootstrap CI resample count.
    """

    dataset: str = "PROTEINS"
    budgets: tuple[int, ...] = DEFAULT_BUDGETS
    n_seeds: int = 3
    n_folds: int = 2
    epochs: int = 30
    learning_rate: float = 1e-2
    batch_size: int = 16
    hidden_dim: int = 64
    num_layers: int = 3
    output_dir: str = "results"
    smoke: bool = False
    bootstrap_resamples: int = 1000


def default_smoke_config(output_dir: str = "results/sensitivity_smoke") -> SensitivityConfig:
    """A fast smoke configuration used by the unit tests.

    Args:
        output_dir: Output directory; defaults to the standard
          ``results/sensitivity_smoke`` location.

    Returns:
        A smoke-tuned :class:`SensitivityConfig`.
    """
    return SensitivityConfig(
        dataset="MUTAG",
        budgets=(4, 8),
        n_seeds=1,
        n_folds=2,
        epochs=2,
        learning_rate=1e-2,
        batch_size=4,
        hidden_dim=32,
        num_layers=2,
        output_dir=output_dir,
        smoke=True,
        bootstrap_resamples=200,
    )


def build_jepa_augmentation_pipeline() -> AugmentationPipeline:
    """Composite graph augmentation pipeline for the JEPA pretraining step.

    Returns:
        A configured :class:`AugmentationPipeline` (``RANDOM_SAMPLE_ONE``
        over ``DropEdge`` / ``DropNode`` / ``DropFeature``).
    """
    return AugmentationPipeline(
        [
            DropEdge(strength=0.2),
            DropNode(strength=0.2),
            DropFeature(strength=0.2),
        ],
        mode=PipelineMode.RANDOM_SAMPLE_ONE,
    )


def encode_and_mean_pool(
    encoder: DualGeometricEncoder, graph: TypedAttributedGraph
) -> torch.Tensor:
    """Mean-pool the per-vertex encoder output into a 1-D tensor.

    The dual-geometric encoder returns a ``(e, h)`` tuple; the
    Euclidean and hyperbolic components are concatenated before
    pooling so the hyperbolic branch contributes to the classifier
    input (the previous version dropped ``h`` and only consumed
    ``e``).

    Args:
        encoder: The dual-geometric encoder.
        graph: The input graph.

    Returns:
        A 1-D ``float`` tensor.
    """
    out = encoder(graph)
    if isinstance(out, tuple):
        eu, hyp = out
        out = torch.cat([eu, hyp], dim=-1)
    if out.ndim == 2:
        out = out.mean(dim=0)
    return out


def pretrain_jepa_one_epoch(
    encoder: DualGeometricEncoder,
    predictor: JEPAPredictor,
    target: TargetEncoder,
    train_pairs: list[tuple[TypedAttributedGraph, int]],
    config: SensitivityConfig,
) -> None:
    """Run a single JEPA pretraining epoch over ``train_pairs``.

    The online encoder is invoked **outside** any
    ``torch.no_grad`` block so its parameters receive gradients
    through the predictor; the target encoder is invoked inside a
    ``torch.no_grad`` block (it provides a stable EMA target).
    After every mini-batch the target encoder is updated as an EMA
    of the online encoder.

    Args:
        encoder: The online encoder.
        predictor: The JEPA predictor head.
        target: The EMA target encoder wrapper.
        train_pairs: The training (graph, label) pairs.
        config: The experiment configuration.
    """
    aug = build_jepa_augmentation_pipeline()
    params = list(encoder.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=1e-4)
    for g, _ in train_pairs:
        aug_g = aug(g)
        ctx_feat = encode_and_mean_pool(encoder, aug_g)
        with torch.no_grad():
            tgt_feat = encode_and_mean_pool(target.shadow, g)
        predicted = predictor(ctx_feat.unsqueeze(0))
        target_t = tgt_feat.unsqueeze(0)
        loss = torch.nn.functional.smooth_l1_loss(predicted, target_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        target.update()


def kfold_indices(n: int, k: int, seed_split: int) -> list[tuple[list[int], list[int]]]:
    """Yield ``(train_indices, test_indices)`` for ``k`` cross-validation folds.

    Args:
        n: Total number of items.
        k: Number of folds (must be positive).
        seed_split: Seed for the index-shuffling generator.

    Returns:
        A list of ``(train_indices, test_indices)`` tuples.

    Raises:
        ValueError: If ``k`` is not positive.
    """
    if k <= 0:
        raise ValueError(f"kfold_indices: k must be positive; got {k}")
    g = torch.Generator().manual_seed(int(seed_split))
    indices = torch.randperm(n, generator=g).tolist()
    fold_size = (n + k - 1) // k
    folds: list[tuple[list[int], list[int]]] = []
    for fold_idx in range(k):
        start = fold_idx * fold_size
        end = min(start + fold_size, n)
        train = indices[:start] + indices[end:]
        test = indices[start:end]
        folds.append((train, test))
    return folds


def train_one_run(
    train_pairs: list[tuple[TypedAttributedGraph, int]],
    test_pairs: list[tuple[TypedAttributedGraph, int]],
    num_classes: int,
    budget: int,
    config: SensitivityConfig,
    seed: int,
) -> float:
    """Train a Persistent-JEPA model with working-graph budget ``budget``.

    Args:
        train_pairs: The training (graph, label) pairs.
        test_pairs: The test (graph, label) pairs.
        num_classes: Number of output classes.
        budget: Working-graph budget ``B``.
        config: The experiment configuration.
        seed: Per-run seed.

    Returns:
        Mean per-class test accuracy in [0, 1].
    """
    if not train_pairs or not test_pairs:
        return 0.0
    input_dim = train_pairs[0][0].vertex_features.shape[1]
    encoder = DualGeometricEncoder(
        input_dim=input_dim,
        euclidean_dim=config.hidden_dim,
        hyperbolic_dim=max(8, config.hidden_dim // 4),
        num_layers=config.num_layers,
    )
    predictor = JEPAPredictor(
        input_dim=config.hidden_dim + max(8, config.hidden_dim // 4),
        hidden_dim=max(64, config.hidden_dim * 2),
        output_dim=config.hidden_dim + max(8, config.hidden_dim // 4),
    )
    target = TargetEncoder(encoder, momentum=0.99)
    classifier = torch.nn.Sequential(
        torch.nn.Linear(
            config.hidden_dim + max(8, config.hidden_dim // 4), max(16, config.hidden_dim // 2)
        ),
        torch.nn.ReLU(),
        torch.nn.Linear(max(16, config.hidden_dim // 2), num_classes),
    )

    pretrain_jepa_one_epoch(encoder, predictor, target, train_pairs, config)

    params = list(encoder.parameters()) + list(classifier.parameters())
    optimizer = torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.epochs))
    loss_fn = torch.nn.CrossEntropyLoss()

    persistent: PersistentState | None = None
    grammar = HRG(nonterminals=("S",), terminals=("a",), productions=(), start="S")
    retriever = GreedyRetrieval(budget=max(1, int(budget)))

    bs = max(1, min(config.batch_size, len(train_pairs)))
    for _epoch in range(config.epochs):
        perm = torch.randperm(len(train_pairs))
        for start in range(0, len(train_pairs), bs):
            idx = perm[start : start + bs].tolist()
            batch = [train_pairs[i] for i in idx]
            optimizer.zero_grad()
            logits_list: list[torch.Tensor] = []
            targets_list: list[int] = []
            for g, lbl in batch:
                obs = (
                    g.vertex_features.mean(dim=0, keepdim=True)
                    if g.num_vertices() > 0
                    else torch.zeros((1, input_dim))
                )
                if persistent is not None and persistent.num_vertices() > 0:
                    utility = FacilityLocationUtility(
                        vertex_features=persistent.graph.vertex_features
                    )
                    result = retriever.select(persistent.graph, obs, utility=utility)
                    working = result.working
                else:
                    utility = FacilityLocationUtility(vertex_features=g.vertex_features)
                    result = retriever.select(g, obs, utility=utility)
                    working = result.working
                if working.num_vertices() > 0:
                    candidate = working.graph
                    if persistent is not None and persistent.num_vertices() > 0:
                        accepted, _ = accept_candidate(
                            candidate=candidate,
                            current=persistent.graph,
                            observation=obs,
                            grammar=grammar,
                        )
                        if accepted:
                            persistent = persistent.commit(
                                candidate=candidate,
                                cost=float(
                                    abs(candidate.num_vertices() - persistent.num_vertices())
                                ),
                                timestamp=float(_epoch),
                                delta_j=-1e-3,
                            )
                    else:
                        persistent = PersistentState(graph=candidate)
                    feats = encode_and_mean_pool(encoder, candidate)
                else:
                    feats = encode_and_mean_pool(encoder, g)
                logits_list.append(classifier(feats.unsqueeze(0)))
                targets_list.append(lbl)
            logits = torch.cat(logits_list, dim=0)
            tgt = torch.tensor(targets_list, dtype=torch.long)
            loss = loss_fn(logits, tgt)
            loss.backward()
            optimizer.step()
            target.update()
        scheduler.step()

    encoder.eval()
    classifier.eval()
    preds: list[int] = []
    labels_eval: list[int] = []
    with torch.no_grad():
        for g, lbl in test_pairs:
            obs = (
                g.vertex_features.mean(dim=0, keepdim=True)
                if g.num_vertices() > 0
                else torch.zeros((1, input_dim))
            )
            utility = FacilityLocationUtility(vertex_features=g.vertex_features)
            result = retriever.select(g, obs, utility=utility)
            target_graph = result.working.graph if result.working.num_vertices() > 0 else g
            feats = encode_and_mean_pool(encoder, target_graph)
            preds.append(int(classifier(feats.unsqueeze(0)).argmax(dim=-1).item()))
            labels_eval.append(lbl)
    return mean_per_class_accuracy(preds, labels_eval)


def aggregate_per_budget(
    rows: list[dict[str, object]], n_resamples: int, seed: int = 0
) -> list[dict[str, object]]:
    """Aggregate per-(B, seed) rows into plan-compliant per-B summary rows.

    The bootstrap CI for each budget is the standard finite-sample CI
    of the mean: ``paired_bootstrap_ci(scores, [0.0] * n)`` returns
    the CI of the mean directly.

    Args:
        rows: The per-(B, seed, fold) raw rows.
        n_resamples: Bootstrap resample count.
        seed: Random seed for the bootstrap resampler.

    Returns:
        A list of per-budget summary rows.
    """
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        grouped[int(row["budget"])].append(float(row["accuracy"]))
    summary: list[dict[str, object]] = []
    for b in sorted(grouped.keys()):
        scores = list(grouped[b])
        n = len(scores)
        if n == 0:
            continue
        mean = sum(scores) / n
        var = sum((s - mean) ** 2 for s in scores) / max(n - 1, 1)
        std = var**0.5
        if n >= 2:
            # Standard finite-score bootstrap CI of the mean.
            ci = paired_bootstrap_ci(scores, [0.0] * n, n_resamples=n_resamples, seed=seed)
            ci_low = ci.ci_low
            ci_high = ci.ci_high
        else:
            ci_low = mean
            ci_high = mean
        summary.append(
            {
                "budget": int(b),
                "n_runs": n,
                "mean_accuracy": float(mean),
                "std_accuracy": float(std),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
            }
        )
    return summary


def render_sensitivity_figure(summary: list[dict[str, object]], png_path: Path) -> None:
    """Render the plan-compliant accuracy-vs-B sensitivity plot (log-x axis).

    Args:
        summary: The per-budget summary produced by
          :func:`aggregate_per_budget`.
        png_path: Destination file path for the PNG figure.
    """
    set_publication_style()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    budgets = [int(s["budget"]) for s in summary]
    means = [float(s["mean_accuracy"]) for s in summary]
    ci_lows = [float(s["ci_low"]) for s in summary]
    ci_highs = [float(s["ci_high"]) for s in summary]
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.errorbar(
        budgets,
        means,
        yerr=[
            [max(0.0, means[i] - ci_lows[i]) for i in range(len(means))],
            [max(0.0, ci_highs[i] - means[i]) for i in range(len(means))],
        ],
        marker="o",
        color=color_for(0),
        capsize=3.0,
        label="Persistent-JEPA",
    )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("working-graph budget B (log scale)")
    ax.set_ylabel("mean accuracy (bootstrap 95% CI)")
    ax.set_title("Sensitivity of Persistent-JEPA to budget B")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path)
    plt.close(fig)


def write_table_csv(rows: list[dict[str, object]], path: Path) -> None:
    """Write ``rows`` to ``path`` as a CSV with the union of keys as fieldnames.

    Args:
        rows: The row dicts to write.
        path: Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_sensitivity(config: SensitivityConfig) -> dict[str, object]:
    """Run the budget sensitivity sweep.

    The function iterates over ``(budget, seed, fold)``, trains one
    Persistent-JEPA model per combination, and aggregates the
    per-(B, seed) accuracies into a per-B summary with bootstrap
    CIs.

    Args:
        config: Experiment configuration. Use
          :func:`default_smoke_config` for fast tests.

    Returns:
        A dictionary with ``raw_rows``, ``summary``, and the output
        paths ``csv``, ``summary_csv``, ``legacy_csv``, and ``png``.
    """
    log = get_logger(__name__)
    log.info(
        "sensitivity starting",
        extra={
            "event": "sensitivity.start",
            "dataset": config.dataset,
            "budgets": list(config.budgets),
            "n_seeds": config.n_seeds,
        },
    )
    graphs, num_classes = load_tu_dataset(config.dataset)
    pairs = [(g.graph, g.label) for g in graphs]
    raw_rows: list[dict[str, object]] = []
    for budget in config.budgets:
        for seed in range(config.n_seeds):
            fold_splits = kfold_indices(len(pairs), config.n_folds, seed * 1000 + int(budget))
            for fold_idx, (train_idx, test_idx) in enumerate(fold_splits):
                train_pairs = [pairs[i] for i in train_idx]
                test_pairs = [pairs[i] for i in test_idx]
                if not train_pairs or not test_pairs:
                    continue
                set_global_seed(seed * 1000 + int(budget) + fold_idx)
                start_t = time.time()
                accuracy = train_one_run(
                    train_pairs=train_pairs,
                    test_pairs=test_pairs,
                    num_classes=num_classes,
                    budget=int(budget),
                    config=config,
                    seed=seed * 1000 + int(budget) + fold_idx,
                )
                elapsed = time.time() - start_t
                raw_rows.append(
                    {
                        "budget": int(budget),
                        "seed": seed,
                        "fold": fold_idx,
                        "accuracy": accuracy,
                        "elapsed_seconds": elapsed,
                    }
                )
                log.info(
                    "sensitivity run complete",
                    extra={
                        "event": "sensitivity.run_complete",
                        "budget": int(budget),
                        "seed": seed,
                        "fold": fold_idx,
                        "accuracy": accuracy,
                    },
                )

    summary = aggregate_per_budget(raw_rows, n_resamples=config.bootstrap_resamples)
    out_root = Path(config.output_dir)
    tables_dir = out_root / "tables"
    plots_dir = out_root / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "sensitivity_B.csv"
    summary_csv = tables_dir / "sensitivity_B_summary.csv"
    png_path = plots_dir / "sensitivity_B.png"
    legacy_csv = out_root / "sensitivity_B.csv"

    raw_with_summary_cols: list[dict[str, object]] = []
    summary_index = {int(s["budget"]): s for s in summary}
    for row in raw_rows:
        s = summary_index.get(int(row["budget"]), {})
        merged = dict(row)
        merged["mean_accuracy"] = s.get("mean_accuracy", float("nan"))
        merged["std_accuracy"] = s.get("std_accuracy", float("nan"))
        merged["ci_low"] = s.get("ci_low", float("nan"))
        merged["ci_high"] = s.get("ci_high", float("nan"))
        raw_with_summary_cols.append(merged)

    write_table_csv(raw_with_summary_cols, csv_path)
    write_table_csv(summary, summary_csv)
    write_table_csv(raw_rows, legacy_csv)
    render_sensitivity_figure(summary, png_path)

    log.info(
        "sensitivity complete",
        extra={
            "event": "sensitivity.complete",
            "n_rows": len(raw_rows),
            "n_summary_rows": len(summary),
            "csv": str(csv_path),
            "summary_csv": str(summary_csv),
            "png": str(png_path),
        },
    )
    return {
        "raw_rows": raw_rows,
        "summary": summary,
        "csv": str(csv_path),
        "summary_csv": str(summary_csv),
        "legacy_csv": str(legacy_csv),
        "png": str(png_path),
    }


def main() -> int:
    """CLI entry point for the sensitivity sweep.

    Returns:
        ``0`` on a successful run.
    """
    parser = argparse.ArgumentParser(
        description="Run the Persistent-JEPA budget sensitivity sweep."
    )
    parser.add_argument("--dataset", default=SensitivityConfig.dataset)
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="*",
        default=list(DEFAULT_BUDGETS),
        help="Budget B values to sweep.",
    )
    parser.add_argument("--seeds", type=int, default=SensitivityConfig.n_seeds)
    parser.add_argument("--folds", type=int, default=SensitivityConfig.n_folds)
    parser.add_argument("--epochs", type=int, default=SensitivityConfig.epochs)
    parser.add_argument("--output-dir", default=SensitivityConfig.output_dir)
    parser.add_argument("--smoke", action="store_true", help="Run the fast smoke configuration.")
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=SensitivityConfig.bootstrap_resamples,
    )
    args = parser.parse_args()
    configure_logging(level="INFO", fmt=LogFormat.JSON)
    if args.smoke:
        cfg = default_smoke_config(output_dir=args.output_dir)
    else:
        cfg = SensitivityConfig(
            dataset=args.dataset,
            budgets=tuple(args.budgets),
            n_seeds=args.seeds,
            n_folds=args.folds,
            epochs=args.epochs,
            output_dir=args.output_dir,
            bootstrap_resamples=args.bootstrap_resamples,
        )
    run_sensitivity(cfg)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
