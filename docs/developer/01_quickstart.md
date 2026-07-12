# Quickstart for Developers

> New to `pjepa`? This guide takes you from install to first experiment
> in 10 minutes.

## 1. Install

The project uses Python 3.12 (3.10 and 3.11 are also supported). We
strongly recommend the included Makefile workflow:

```bash
git clone https://github.com/sachncs/persistent-jepa.git
cd persistent-jepa
make install
```

This creates a virtual environment at `.venv`, installs the package
in editable mode, and pulls in development dependencies (pytest, ruff,
pytype, mkdocs).

If you prefer not to use the Makefile:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,ogb]"
```

## 2. Verify Your Environment

`pjepa` has six capability probes that exercise the active compute
backend. Run them all at once:

```bash
make doctor
```

You should see output similar to:

```
Backend:    mps
Device:     Apple Silicon (MPS)
Python:     3.12.4
PyTorch:    2.13.0
Platform:   macOS-15.0-arm64-arm-64bit-Mach-O
CPU count:  12

Capability probes:
  [GREEN ] matmul
  [GREEN ] scatter_add
  [GREEN ] torch.compile
  [GREEN ] hyperbolic
  [GREEN ] pyg_scatter
  [GREEN ] cpu_fallback
```

If any probe reports RED, the corresponding feature is unavailable and
`pjepa` will fall back to a CPU implementation.

## 3. Run the Cheapest Validation Benchmarks

The paper makes two claims that have cheap, fast validations. Both are
runnable without any training:

```bash
# Validates Theorem 3: greedy retrieval achieves (1 - 1/e) ≈ 0.632 of optimal
make bench-retrieval

# Validates Proposition 7: hyperbolic per-edge distortion is Θ(log D / (D log b))
make bench-distortion
```

Each script prints a structured JSON summary to stdout.

## 4. Tour the Code

The repository is organised as:

```
src/pjepa/
├── graphs/       # typed attributed graphs, persistent state, working graph
├── encoders/     # Euclidean MPNN, hyperbolic projection, dual-geometric encoder, JEPA predictor
├── retrieval/    # greedy submodular retrieval with (1 - 1/e) guarantee
├── rewriting/    # hyperedge-replacement grammar, bisimulation metric, four-conditions, DPO
├── scheduler/    # PPO trainer, replay buffer, sleep cadence
├── objectives/   # unified free-energy functional 𝒥, IB Lagrangian, MDL
├── dynamics/     # evolution operator F, contraction analysis, fixed-point iteration
├── augmentations/ # DropEdge, DropNode, DropFeature, FeatureMask, RandomWalkSubgraph
├── data/         # TUDataset, OGB-arxiv, class-incremental splits
├── baselines/    # GCN, GIN, GraphMAE, GraphCL, InfoGraph, EWC, GEM
├── training/     # pretrain_loop, supervised_train_loop, linear_probe_eval, Checkpoint
├── eval/         # metrics, bootstrap CI, statistical tests
├── perf/         # performance infrastructure (compile, autocast, EMA)
├── cli/          # Typer-based CLI (doctor, hardware, benchmark, train, eval)
├── utils/        # deterministic seeding
├── logging_setup.py # structured logging (HUMAN and JSON)
├── hardware.py   # backend detection and capability probes
├── config.py     # YAML configuration loading and validation
├── exceptions.py # PJEPAError hierarchy
└── __init__.py   # public API
```

Every public symbol has a Google-style docstring; run `help(obj)` in
a Python REPL to see the full documentation.

## 5. Write Your First Experiment

Create a file `experiments/my_experiment.py`:

```python
"""My first experiment with pjepa."""

import torch

from pjepa.graphs import TypedAttributedGraph
from pjepa.retrieval import GreedyRetrieval, FacilityLocationUtility
from pjepa.seeding import set_global_seed


def main() -> None:
    """Demonstrate retrieval on a random persistent graph."""
    set_global_seed(42)

    # Build a persistent graph with random vertex features.
    persistent = TypedAttributedGraph(
        vertex_features=torch.randn((50, 8)),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
    )

    # Build a utility function from the vertex features.
    utility = FacilityLocationUtility(vertex_features=persistent.vertex_features)

    # Choose a working graph via greedy retrieval.
    observation = torch.randn((1, 8))
    result = GreedyRetrieval(budget=16).select(
        persistent, observation, utility=utility
    )

    print(f"selected {result.working.num_vertices()} vertices")
    print(f"utility: {result.utility:.4f}")


if __name__ == "__main__":
    main()
```

Run it:

```bash
.venv/bin/python experiments/my_experiment.py
```

## 6. Run the Test Suite

The test suite uses the eight-class taxonomy:

* **happy** — typical inputs produce expected outputs
* **bad** — malformed inputs raise typed errors
* **ugly** — edge cases (NaN, Inf, empty graphs, single vertices) don't crash
* **leaky** — long-running operations don't grow memory unbounded
* **round-trip** — save → load → continue is equivalent to save → continue
* **cross-backend** — same code on MPS/CUDA/CPU gives same output within tolerance
* **distributional** — statistical properties hold across runs
* **property** — hypothesis-driven invariants (submodularity, monotonicity, etc.)

Run everything:

```bash
make test
```

Run only fast tests:

```bash
make test-fast
```

## 7. Add a New Encoder or Baseline

The cleanest extension point is a new encoder. Define a class that
satisfies the `Encoder` protocol:

```python
from torch import nn
from pjepa.graphs import TypedAttributedGraph
from pjepa.encoders.base import Encoder


class MyEncoder(nn.Module):
    """My new encoder."""

    output_dim: int = 64

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, self.output_dim)

    def forward(self, graph: TypedAttributedGraph) -> torch.Tensor:
        return self.proj(graph.vertex_features)

    def to(self, device: torch.device) -> "MyEncoder":
        return super().to(device)
```

That's it — `MyEncoder` satisfies the protocol and can be used
wherever an `Encoder` is expected.

## 8. Add a Test

Add a test in the corresponding `tests/test_<module>.py` file. Every
test should follow the eight-class taxonomy:

```python
def test_happy_my_encoder_forward() -> None:
    """My encoder returns per-vertex embeddings of the right shape."""
    g = TypedAttributedGraph(
        vertex_features=torch.randn((5, 4)),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
    )
    encoder = MyEncoder(input_dim=4)
    out = encoder(g)
    assert out.shape == (5, 64)


def test_bad_my_encoder_zero_dim() -> None:
    """Zero dimensions are rejected."""
    with pytest.raises(ValueError):
        MyEncoder(input_dim=0)
```

Run the new test in isolation:

```bash
pytest tests/test_my_encoder.py -v
```

## 9. Where Next?

* [Architecture](../researcher/01_persistent_graph_world_model.md) — for the full picture.
* [Adding a custom encoder](03_adding_an_encoder.md) — more detail.
* [Adding a custom baseline](04_adding_a_baseline.md) — for SOTA comparison.
* [Reproducing paper results](05_reproducing_paper_results.md) — one-command reproduction.