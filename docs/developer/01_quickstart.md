# Quickstart for Developers

> New to `pjepa`? This guide takes you from install to first
> experiment in 10 minutes.

## 1. Install

The project uses Python 3.12 (3.10 and 3.11 are also supported).
We strongly recommend the included Makefile workflow:

```bash
git clone https://github.com/sachncs/pjepa.git
cd pjepa
make install
```

`make install` creates a virtual environment at `.venv`, installs
the package in editable mode, and pulls in development
dependencies (pytest, ruff, pytype, mkdocs, optuna, etc.).

If you prefer not to use the Makefile:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,ogb]"
```

Or use the canonical `setup.sh` script (also runs `pjepa doctor`
and the test suite):

```bash
bash setup.sh
```

## 2. Verify Your Environment

`pjepa` ships six capability probes that exercise the active
compute backend. Run them all at once:

```bash
source .venv/bin/activate
pjepa doctor
```

You should see output similar to:

```
Backend:    mps
Device:     Apple Silicon (MPS)
Python:     3.12.7
PyTorch:    2.13.0
Platform:   macOS-26.6.2-arm64-arm-64bit
CPU count:  12

Capability probes:
  [GREEN ] matmul
  [GREEN ] scatter_add
  [GREEN ] torch.compile
  [GREEN ] hyperbolic
  [GREEN ] pyg_scatter
  [GREEN ] cpu_fallback
```

If any probe reports RED, the corresponding feature is unavailable
and `pjepa` will fall back to a CPU implementation. The doctor
command exits with code 2 when at least one probe is RED.

## 3. Run the Cheapest Validation Benchmarks

The paper makes three claims that have cheap, fast validations.
Each prints a structured JSON summary to stdout:

```bash
# Theorem 3: greedy retrieval achieves (1 - 1/e) ≈ 0.632 of optimal
pjepa benchmark retrieval

# Proposition 7: hyperbolic per-edge distortion is Θ(log D / (D log b))
pjepa benchmark distortion

# Proposition 3: dual-geometric beats Euclidean-only
pjepa benchmark encoder-ablation
```

`all_pass: true` means every row met its threshold.

## 4. Tour the Code

The repository is organised as:

```
src/pjepa/
├── graphs/       # Graph (immutable), State (commit/reject), Working
├── encoders/     # Encoder ABC, Euclidean, Hyperbolic, DualGeometric
│                 # Head ABC, Predictor, Target
├── retrieval/    # Retrieval, Utility ABC, Facility, InfoGain, Result
├── rewriting/    # HRG, Bisimulation, Criterion ABC, FourConditions
├── scheduler/    # PPOTrainer, Buffer, Storage ABC, Cadence ABC, Sleep
├── objectives/   # FreeEnergy, ib_lagrangian, description_length
├── dynamics/     # EvolutionOperator, contractivity_bound, fixed_point_iteration
├── augmentations/ # Transform ABC, Pipeline, DropEdge, DropNode, …
├── data/         # TUDataset, OGB-arxiv, class-incremental splits
├── baselines/    # Naive, GCN, GIN, GraphSAGE, GraphCL, GraphMAE,
│                 # InfoGraph, BGRL, EWC, GEM, PackNet
├── training/     # pretrain_loop, supervised_train_loop, SWA, TTA,
│                 # Ensemble, Distillation, linear_probe_eval, Checkpoint
├── eval/         # metrics, bootstrap CI, statistical tests, aggregator
├── perf/         # safe_compile, autocast, EMATarget, fused scatter, sync
├── cli/          # Typer-based CLI (doctor, hardware, benchmark, train,
│                 # tune, baseline-smoke, decoupling, ablation,
│                 # sensitivity, aggregate)
├── utils/        # deterministic seeding
├── logging_setup.py # structured logging (HUMAN and JSON)
├── hardware.py   # backend detection and capability probes
├── config.py     # YAML configuration loading and validation
├── exceptions.py # PJEPAError hierarchy
└── __init__.py   # public API
```

Every public symbol has a Google-style docstring; run `help(obj)`
in a Python REPL to see the full documentation.

## 5. Write Your First Experiment

Create a file `experiments/my_experiment.py`:

```python
"""My first experiment with pjepa."""

import torch

from pjepa.graphs import Graph
from pjepa.retrieval import Retrieval, Facility
from pjepa.seeding import set_global_seed


def main() -> None:
    """Demonstrate retrieval on a random persistent graph."""
    set_global_seed(42)

    # Build a graph with random vertex features.
    graph = Graph(
        vertex_features=torch.randn((50, 8)),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
    )

    # Build a utility function from the vertex features.
    utility = Facility(vertex_features=graph.vertex_features)

    # Choose a working graph via greedy retrieval.
    observation = torch.randn((1, 8))
    result = Retrieval(budget=16).select(
        graph, observation, utility=utility
    )

    print(f"selected {result.working.num_vertices()} vertices")
    print(f"utility: {result.utility:.4f}")
    print(f"iterations: {result.iterations}")


if __name__ == "__main__":
    main()
```

Run it:

```bash
.venv/bin/python experiments/my_experiment.py
```

## 6. Run the Test Suite

The test suite uses the eight-class taxonomy:

- **happy** — typical inputs produce expected outputs
- **bad** — malformed inputs raise typed errors
- **ugly** — edge cases (NaN, Inf, empty graphs, single vertices) don't crash
- **leaky** — long-running operations don't grow memory unbounded
- **round-trip** — save → load → continue is equivalent to save → continue
- **cross-backend** — same code on MPS/CUDA/CPU gives same output within tolerance
- **distributional** — statistical properties hold across runs
- **property** — hypothesis-driven invariants (submodularity, monotonicity, etc.)

Run everything:

```bash
pytest
```

Run only fast tests:

```bash
pytest -m "not slow"
```

## 7. Add a New Encoder or Baseline

The cleanest extension point is a new encoder. Define a class that
subclasses the polymorphic `Encoder` ABC:

```python
from torch import nn
from pjepa.graphs import Graph
from pjepa.encoders.base import Encoder


class MyEncoder(Encoder):
    """My new encoder."""

    output_dim: int = 64

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        self.input_dim = input_dim
        self.proj = nn.Linear(input_dim, self.output_dim)

    def forward(self, graph: Graph) -> torch.Tensor:
        """Return per-vertex embeddings of shape ``[N, output_dim]``."""
        return self.proj(graph.vertex_features)
```

That's it — `MyEncoder` is a polymorphic `Encoder` and can be used
wherever one is expected (the `Encoder.encode` method defaults to
`forward`, and the `Encoder.summary` method is inherited).

## 8. Add a Test

Add a test in the corresponding `tests/test_<module>.py` file.
Every test should follow the eight-class taxonomy:

```python
import pytest

from experiments.my_experiment import MyEncoder
from pjepa.graphs import Graph

import torch


def test_happy_my_encoder_forward() -> None:
    """My encoder returns per-vertex embeddings of the right shape."""
    g = Graph(
        vertex_features=torch.randn((5, 4)),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
    )
    encoder = MyEncoder(input_dim=4)
    out = encoder(g)
    assert out.shape == (5, 64)


def test_bad_my_encoder_zero_dim() -> None:
    """Zero input dimension is rejected."""
    with pytest.raises(ValueError):
        MyEncoder(input_dim=0)
```

Run the new test in isolation:

```bash
pytest tests/test_my_encoder.py -v
```

## 9. Where Next?

- [Architecture](../researcher/01_persistent_graph_world_model.md) — for the full picture.
- [Adding a custom encoder](03_adding_an_encoder.md) — more detail.
- [Adding a custom baseline](04_adding_a_baseline.md) — for SOTA comparison.
- [Reproducing paper results](05_reproducing_paper_results.md) — one-command reproduction.
