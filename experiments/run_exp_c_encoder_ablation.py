"""Experiment C — Encoder ablation (Phase 4).

Validates Proposition 3 (Hierarchical Consistency) by training three
encoder variants — Euclidean-only MPNN, Hyperbolic-only MPNN, and the
Dual-Geometric encoder — on a held-out collection of synthetic AST-like
graphs and comparing their ability to predict per-vertex depth.

The three encoders and what they actually emit at ``forward()``:

* ``EuclideanMPNN`` — a three-layer GIN-style MPNN; output is a
  ``[N, hidden_dim]`` Euclidean per-vertex embedding.
* ``HyperbolicMPNN`` — the same MPNN followed by a
  :class:`HyperbolicProjection`; output is a ``[N, hyperbolic_dim]``
  hyperbolic per-vertex embedding with norms below 1.
* ``DualGeometricEncoder`` — emits the **tuple**
  ``(euclidean, hyperbolic)`` and the predictor concatenates both
  components so that the dual-geometric nature of the encoder is
  actually exercised by the loss (a Euclidean-only extractor would
  not be a faithful ablation of the encoder class).

Held-out evaluation uses disjoint train/test graphs from distinct
seeds so that the encoder must generalise across structurally distinct
trees; a random-chance reference line is drawn per depth as
``1 / (D + 1)``.

Outputs:
    ``<output_dir>/encoder_ablation.csv``
    ``<output_dir>/encoder_ablation.png``
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn

from pjepa.encoders import (
    DualGeometricEncoder,
    EuclideanMPNN,
    HyperbolicProjection,
)
from pjepa.eval import color_for, set_publication_style
from pjepa.graphs import TypedAttributedGraph
from pjepa.logging_setup import LogFormat, configure_logging, get_logger
from pjepa.utils.seeding import set_global_seed

__all__ = [
    "DEFAULT_DEPTHS",
    "DEFAULT_EPOCHS",
    "DEFAULT_N_GRAPHS",
    "DEFAULT_N_SEEDS",
    "HIDDEN_DIM",
    "EncoderAblationConfig",
    "HyperbolicMPNN",
    "build_ast_like_graph",
    "build_encoder",
    "evaluate_accuracy",
    "plot_accuracy",
    "predict_depth_loss",
    "run_encoder_ablation",
    "train_one_encoder",
]


DEFAULT_DEPTHS: tuple[int, ...] = (5, 10)
DEFAULT_N_GRAPHS: int = 6
DEFAULT_N_SEEDS: int = 2
DEFAULT_EPOCHS: int = 30
HIDDEN_DIM: int = 32


class EncoderAblationConfig:
    """Configuration for the encoder ablation experiment.

    Attributes:
        depths: AST depths to sweep over.
        n_graphs: Number of AST-like graphs per depth (split
          train/test).
        n_seeds: Number of random seeds per (depth, encoder) cell.
        epochs: Number of training epochs per run.
        output_dir: Output directory for CSV and PNG.
        test_fraction: Fraction of graphs held out for evaluation.
        seed: Base seed for the graph-construction generator.
    """

    def __init__(
        self,
        depths: tuple[int, ...] = DEFAULT_DEPTHS,
        n_graphs: int = DEFAULT_N_GRAPHS,
        n_seeds: int = DEFAULT_N_SEEDS,
        epochs: int = DEFAULT_EPOCHS,
        output_dir: str = "results",
        test_fraction: float = 0.25,
        seed: int = 0,
    ) -> None:
        """Store the experiment parameters.

        Args:
            depths: AST depths to sweep over.
            n_graphs: Number of AST-like graphs per depth.
            n_seeds: Number of random seeds per (depth, encoder) cell.
            epochs: Number of training epochs per run.
            output_dir: Output directory for CSV and PNG.
            test_fraction: Fraction of graphs held out for evaluation.
            seed: Base seed for the graph-construction generator.
        """
        self.depths = tuple(int(d) for d in depths if int(d) > 0)
        self.n_graphs = int(n_graphs)
        self.n_seeds = int(n_seeds)
        self.epochs = int(epochs)
        self.output_dir = str(output_dir)
        self.test_fraction = float(test_fraction)
        self.seed = int(seed)


def build_ast_like_graph(depth: int, branching: int, seed: int) -> TypedAttributedGraph:
    """Build a synthetic b-ary tree of given depth with one-hot depth features.

    Vertex features are one-hot vectors of size ``depth + 1`` indicating
    the depth of each vertex; this provides a structured input signal
    so the encoder ablation measures topology-aware representation
    learning rather than feature memorisation. Different ``seed`` values
    produce structurally identical trees (the construction is
    deterministic), but the seed is preserved in the signature so
    callers can route it through :func:`set_global_seed` to vary
    surrounding randomness.

    Args:
        depth: Tree depth (root has depth 0).
        branching: Branching factor per non-leaf node.
        seed: Random seed (currently unused, kept for API consistency).

    Returns:
        A :class:`TypedAttributedGraph` with one-hot depth features.

    Raises:
        ValueError: If ``branching < 2`` or ``depth < 0``.
    """
    del seed
    if branching < 2:
        raise ValueError(f"build_ast_like_graph: branching must be >= 2; got {branching}")
    if depth < 0:
        raise ValueError(f"build_ast_like_graph: depth must be non-negative; got {depth}")
    n_vertices = sum(int(branching) ** k for k in range(int(depth) + 1))
    if n_vertices == 0:
        return TypedAttributedGraph(
            vertex_features=torch.zeros((0, 0)),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
        )
    depth_labels = torch.zeros(n_vertices, dtype=torch.long)
    running = 0
    for k in range(int(depth) + 1):
        level_size = int(branching) ** k
        for v in range(running, running + level_size):
            depth_labels[v] = k
        running += level_size
    edges: list[tuple[int, int]] = []
    for level in range(int(depth)):
        start = sum(int(branching) ** k for k in range(level))
        next_start = sum(int(branching) ** k for k in range(level + 1))
        for parent in range(start, start + int(branching) ** level):
            for child_offset in range(int(branching)):
                child = next_start + (parent - start) * int(branching) + child_offset
                edges.append((parent, child))
    edge_index = (
        torch.tensor(edges, dtype=torch.long).T if edges else torch.zeros((2, 0), dtype=torch.long)
    )
    return TypedAttributedGraph(
        vertex_features=torch.nn.functional.one_hot(
            depth_labels, num_classes=int(depth) + 1
        ).float(),
        edge_index=edge_index,
        edge_features=torch.zeros((edge_index.shape[1], 1)),
        vertex_labels=depth_labels,
    )


class HyperbolicMPNN(nn.Module):
    """Hyperbolic-only encoder: EuclideanMPNN followed by HyperbolicProjection.

    The forward pass returns a ``[N, output_dim]`` hyperbolic
    embedding whose norms lie strictly below the projection's
    ``max_norm``.

    Attributes:
        mpnn: The underlying Euclidean MPNN trunk.
        proj: The hyperbolic projection head.
        output_dim: Output dimension (matches ``proj.output_dim``).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
    ) -> None:
        """Initialise the hyperbolic encoder.

        Args:
            input_dim: Vertex feature dimensionality.
            hidden_dim: Width of the MPNN layers.
            num_layers: Number of MPNN layers.
            output_dim: Output dimension of the hyperbolic projection.
        """
        super().__init__()
        self.mpnn = EuclideanMPNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=hidden_dim,
        )
        self.proj = HyperbolicProjection(
            input_dim=hidden_dim,
            output_dim=output_dim,
        )
        self.output_dim = int(output_dim)

    def forward(self, graph: TypedAttributedGraph) -> torch.Tensor:
        """Encode the graph into a hyperbolic per-vertex embedding.

        Args:
            graph: The input graph.

        Returns:
            A ``[N, output_dim]`` tensor of hyperbolic features.
        """
        x = self.mpnn(graph)
        return self.proj(x)


def predict_depth_loss(
    encoder: nn.Module,
    graphs: list[TypedAttributedGraph],
    depth: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-graph vertex logits and stacked depth labels.

    For tuple-returning encoders (the dual-geometric encoder) the two
    components are concatenated so the loss has access to **both** the
    Euclidean and hyperbolic embeddings — a Euclidean-only extraction
    would not be a faithful ablation of the dual encoder.

    Args:
        encoder: The encoder module under evaluation.
        graphs: The graphs to score.
        depth: Maximum tree depth; the output dimension is ``depth + 1``.

    Returns:
        A tuple ``(logits, labels)`` where ``logits`` has shape
        ``[sum_n, depth + 1]`` and ``labels`` has shape ``[sum_n]``.
    """
    logits_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    for g in graphs:
        out = encoder(g)
        if isinstance(out, tuple):
            out = torch.cat([component for component in out], dim=-1)
        logits_list.append(out)
        labels_list.append(g.vertex_labels)
    return torch.cat(logits_list, dim=0), torch.cat(labels_list, dim=0)


def evaluate_accuracy(
    encoder: nn.Module,
    graphs: list[TypedAttributedGraph],
    depth: int,
) -> float:
    """Return the vertex-level depth-prediction accuracy on ``graphs``.

    Args:
        encoder: The encoder module.
        graphs: The graphs to score.
        depth: Maximum tree depth.

    Returns:
        The vertex-level accuracy in ``[0, 1]``.
    """
    encoder.eval()
    with torch.no_grad():
        logits, labels = predict_depth_loss(encoder, graphs, depth)
        if logits.shape[-1] != depth + 1:
            projected = torch.nn.functional.linear(logits, torch.eye(depth + 1, logits.shape[-1]))
        else:
            projected = logits
        preds = projected.argmax(dim=-1)
        correct = (preds == labels).float().mean().item()
    encoder.train()
    return float(correct)


def train_one_encoder(
    name: str,
    encoder: nn.Module,
    train_graphs: list[TypedAttributedGraph],
    test_graphs: list[TypedAttributedGraph],
    depth: int,
    epochs: int,
    lr: float,
) -> float:
    """Train one encoder and return held-out vertex-level accuracy.

    Args:
        name: Display name (currently unused but kept for logging).
        encoder: The encoder module to train.
        train_graphs: The training graphs.
        test_graphs: The held-out test graphs.
        depth: Maximum tree depth (``output_dim = depth + 1``).
        epochs: Number of training epochs.
        lr: Optimiser learning rate.

    Returns:
        Held-out vertex-level accuracy in ``[0, 1]``.
    """
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    encoder.train()
    output_dim = depth + 1
    for _ in range(int(epochs)):
        optimizer.zero_grad()
        logits, labels = predict_depth_loss(encoder, train_graphs, depth)
        if logits.shape[-1] != output_dim:
            projection = nn.Linear(logits.shape[-1], output_dim).to(logits.device)
            projected = projection(logits)
        else:
            projected = logits
        loss = loss_fn(projected, labels)
        loss.backward()
        optimizer.step()
    return evaluate_accuracy(encoder, test_graphs, depth)


def build_encoder(name: str, input_dim: int, output_dim: int) -> nn.Module:
    """Construct the named encoder variant.

    Args:
        name: One of ``"EuclideanMPNN"``, ``"HyperbolicMPNN"``,
          ``"DualGeometricEncoder"``.
        input_dim: Vertex feature dimensionality.
        output_dim: Per-vertex output dimensionality (``depth + 1``).

    Returns:
        An instantiated ``nn.Module``.

    Raises:
        ValueError: If ``name`` is not a known encoder variant.
    """
    if name == "EuclideanMPNN":
        return EuclideanMPNN(
            input_dim=input_dim,
            hidden_dim=HIDDEN_DIM,
            num_layers=3,
            output_dim=output_dim,
        )
    if name == "HyperbolicMPNN":
        return HyperbolicMPNN(
            input_dim=input_dim,
            hidden_dim=HIDDEN_DIM,
            num_layers=3,
            output_dim=output_dim,
        )
    if name == "DualGeometricEncoder":
        return DualGeometricEncoder(
            input_dim=input_dim,
            euclidean_dim=HIDDEN_DIM,
            hyperbolic_dim=output_dim,
            num_layers=3,
        )
    raise ValueError(f"build_encoder: unknown encoder {name!r}")


def run_encoder_ablation(
    config: EncoderAblationConfig | None = None,
) -> dict[str, object]:
    """Run the encoder ablation experiment.

    Args:
        config: Experiment configuration. When ``None``, the smoke
          defaults are used.

    Returns:
        A dictionary with ``rows`` and the output paths.
    """
    if config is None:
        config = EncoderAblationConfig()
    log = get_logger(__name__)
    encoder_names = ("EuclideanMPNN", "HyperbolicMPNN", "DualGeometricEncoder")
    rows: list[dict[str, object]] = []
    for depth in config.depths:
        output_dim = depth + 1
        for seed in range(config.n_seeds):
            set_global_seed(seed * 1013 + int(depth))
            graphs: list[TypedAttributedGraph] = []
            for k in range(config.n_graphs):
                graphs.append(
                    build_ast_like_graph(
                        depth=depth,
                        branching=2,
                        seed=seed * 1000 + k,
                    )
                )
            split_gen = torch.Generator().manual_seed(config.seed + seed * 17 + int(depth))
            perm = torch.randperm(len(graphs), generator=split_gen).tolist()
            n_test = max(1, round(config.n_graphs * config.test_fraction))
            n_test = min(n_test, len(graphs) - 1) if len(graphs) > 1 else 0
            test_indices = set(perm[:n_test])
            test_graphs = [g for i, g in enumerate(graphs) if i in test_indices]
            train_graphs = [g for i, g in enumerate(graphs) if i not in test_indices]
            if not train_graphs:
                train_graphs = list(graphs)
            for name in encoder_names:
                enc = build_encoder(name, input_dim=output_dim, output_dim=output_dim)
                lr = 1e-2 if name != "HyperbolicMPNN" else 5e-3
                accuracy = train_one_encoder(
                    name,
                    enc,
                    train_graphs=train_graphs,
                    test_graphs=test_graphs,
                    depth=depth,
                    epochs=config.epochs,
                    lr=lr,
                )
                rows.append(
                    {
                        "depth": int(depth),
                        "seed": int(seed),
                        "encoder": name,
                        "n_train_graphs": len(train_graphs),
                        "n_test_graphs": len(test_graphs),
                        "n_vertices_per_graph": int(graphs[0].num_vertices()),
                        "accuracy": float(accuracy),
                    }
                )
                log.info(
                    "encoder ablation trial complete",
                    extra={
                        "event": "encoder_ablation.trial",
                        "depth": int(depth),
                        "seed": int(seed),
                        "encoder": name,
                        "accuracy": float(accuracy),
                    },
                )
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "encoder_ablation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "depth",
                "seed",
                "encoder",
                "n_train_graphs",
                "n_test_graphs",
                "n_vertices_per_graph",
                "accuracy",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    png_path = out_dir / "encoder_ablation.png"
    plot_accuracy(rows, png_path)
    log.info(
        "experiment complete",
        extra={
            "event": "encoder_ablation.experiment_complete",
            "n_rows": len(rows),
            "csv": str(csv_path),
            "png": str(png_path),
        },
    )
    return {"rows": rows, "csv": str(csv_path), "png": str(png_path)}


def plot_accuracy(rows: list[dict[str, object]], png_path: Path) -> None:
    """Plot mean held-out accuracy vs depth, one line per encoder.

    Each depth ``D`` is annotated with its random-chance baseline
    ``1 / (D + 1)`` as a short dashed line on the right-hand side of
    the plot, so the legend does not conflate per-depth baselines with
    encoder curves.

    Args:
        rows: Per-trial rows emitted by :func:`run_encoder_ablation`.
        png_path: Destination PNG path.
    """
    set_publication_style()
    grouped: dict[str, dict[int, list[float]]] = {}
    for row in rows:
        grouped.setdefault(str(row["encoder"]), {})
        grouped[str(row["encoder"])].setdefault(int(row["depth"]), []).append(
            float(row["accuracy"])
        )
    fig, ax = plt.subplots()
    depths_present = sorted({int(row["depth"]) for row in rows})
    for idx, (name, by_depth) in enumerate(sorted(grouped.items())):
        depths = sorted(by_depth.keys())
        means = [sum(by_depth[d]) / len(by_depth[d]) if by_depth[d] else 0.0 for d in depths]
        stds = [
            (sum((v - m) ** 2 for v in by_depth[d]) / max(len(by_depth[d]) - 1, 1)) ** 0.5
            if by_depth[d]
            else 0.0
            for d, m in zip(depths, means, strict=False)
        ]
        ax.errorbar(
            depths,
            means,
            yerr=stds,
            marker="o",
            color=color_for(idx),
            label=name,
        )
    for d in depths_present:
        chance = 1.0 / float(d + 1)
        ax.hlines(
            chance,
            xmin=d - 0.15,
            xmax=d + 0.15,
            color="black",
            linestyle=":",
        )
        ax.annotate(
            f"chance D={d}: {chance:.3f}",
            xy=(d, chance),
            xytext=(d + 0.1, chance + 0.02),
            fontsize=7,
            color="black",
        )
    ax.set_xlabel("AST depth D")
    ax.set_ylabel("Held-out vertex-level depth accuracy")
    ax.set_title("Encoder ablation: hierarchical depth prediction")
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="upper right")
    fig.savefig(png_path)
    plt.close(fig)


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run the encoder ablation experiment.")
    parser.add_argument("--depths", type=int, nargs="*", default=list(DEFAULT_DEPTHS))
    parser.add_argument("--n-graphs", type=int, default=DEFAULT_N_GRAPHS)
    parser.add_argument("--seeds", type=int, default=DEFAULT_N_SEEDS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    configure_logging(level="INFO", fmt=LogFormat.JSON)
    config = EncoderAblationConfig(
        depths=tuple(args.depths),
        n_graphs=int(args.n_graphs),
        n_seeds=int(args.seeds),
        epochs=int(args.epochs),
        output_dir=str(args.output_dir),
    )
    summary = run_encoder_ablation(config)
    log = get_logger(__name__)
    log.info(
        "encoder ablation summary",
        extra={
            "event": "encoder_ablation.summary",
            "n_rows": len(summary["rows"]),
        },
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
