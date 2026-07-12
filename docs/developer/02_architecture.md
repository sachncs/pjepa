# Architecture Overview

> A developer-focused description of how the modules fit together.

## The Big Picture

`pjepa` is organised around a single variational objective $\mathcal{J}$
that governs every component. The architecture has three layers:

```
+--------------------------------------------------+
|  Persistent state (G_t) — slow weight             |
|    +----------------------------------------+    |
|    |  Working graph (W_t) — bounded subgraph|    |
|    |    +------------------------------+   |    |
|    |    |  Encoder, Predictor, Rewriter|   |    |
|    |    +------------------------------+   |    |
|    +----------------------------------------+    |
+--------------------------------------------------+
       ↑                                   ↑
       | commit / reject                   | observe
       |                                   |
+--------------------------------------------------+
|  Kernel parameters (Θ) — fast weight             |
+--------------------------------------------------+
```

The persistent state is the **only** state that survives across
observations. Everything else is ephemeral.

## Module Dependency Graph

```
        CLI  ─────►  Training loops  ─────►  Encoders, Retrieval, Rewriting
                            │                          │
                            ▼                          ▼
                     FreeEnergy J ◄───────────────  Hardware, Seeding
                            │
                            ▼
                  Baselines, Data loaders, Eval metrics, Checkpointing
```

Key observations:

* The free-energy functional $\mathcal{J}$ is the *only* loss shared
  across components. There is no contrastive loss, no reconstruction
  loss, no policy loss. Everything is unified.
* The hardware module sits at the bottom: every other module routes
  through it for device placement and capability detection.
* The CLI is a thin wrapper; the actual logic lives in
  `pjepa.training.*`.

## Public API Contract

Every public symbol must:

1. Have a Google-style docstring with Args, Returns, Raises, Example.
2. Be listed in the module's `__all__`.
3. Pass the eight-class test suite.
4. Type-check under `pytype --strict`.
5. Lint clean under `ruff check`.

The lint config enforces these rules at every commit via pre-commit
hooks.

## Critical Invariants

The framework maintains four invariants (§3.8 of the paper):

| Invariant | Enforcement | Test class |
|---|---|---|
| I1: $G_t$ is an approximate sufficient statistic | bisimilarity condition (iii) of §7.7 | property |
| I2: $|V(G_t)| \le N_{\max}$ | bounded-cost condition (iv) of §7.7 | property |
| I3: Reasoning executes on $W_t$, not $G_t$ | by construction in §5.3 | happy + property |
| I4: $\mathcal{J}$ is strictly decreasing | acceptance criterion §3.7 | property |

Each invariant has a runtime assertion (where possible) and a
property test (always).

## Where to Read the Code

| File | Purpose | Lines |
|---|---|---|
| `src/pjepa/graphs/typed_graph.py` | Core immutable graph | ~210 |
| `src/pjepa/graphs/persistent_state.py` | Wrapper with commit/reject | ~140 |
| `src/pjepa/graphs/working_graph.py` | Bounded working subgraph | ~80 |
| `src/pjepa/encoders/euclidean_mpnn.py` | MPNN encoder | ~70 |
| `src/pjepa/encoders/hyperbolic.py` | Poincaré ball projection | ~75 |
| `src/pjepa/encoders/dual_geometric.py` | Combined encoder | ~55 |
| `src/pjepa/encoders/jepa_predictor.py` | Predictor + target EMA | ~85 |
| `src/pjepa/retrieval/utility.py` | Submodular utilities | ~135 |
| `src/pjepa/retrieval/greedy.py` | Greedy retrieval | ~120 |
| `src/pjepa/rewriting/hrg.py` | Hyperedge-replacement grammar | ~75 |
| `src/pjepa/rewriting/bisimulation.py` | Bisimulation metric | ~95 |
| `src/pjepa/rewriting/four_conditions.py` | Acceptance criterion | ~140 |
| `src/pjepa/rewriting/dpo.py` | DPO loss | ~75 |
| `src/pjepa/objectives/free_energy.py` | The 𝒥 functional | ~75 |
| `src/pjepa/objectives/ib_lagrangian.py` | IB Lagrangian + VIB bound | ~75 |
| `src/pjepa/objectives/mdl.py` | Description length | ~40 |
| `src/pjepa/dynamics/__init__.py` | F operator analysis | ~100 |
| `src/pjepa/scheduler/ppo.py` | PPO trainer | ~165 |
| `src/pjepa/scheduler/buffer.py` | Replay buffer | ~100 |
| `src/pjepa/scheduler/cadence.py` | Sleep cadence | ~80 |
| `src/pjepa/augmentations/structural.py` | DropEdge, DropNode, RandomWalkSubgraph | ~110 |
| `src/pjepa/augmentations/feature.py` | DropFeature, FeatureMask | ~75 |
| `src/pjepa/data/tu.py` | TUDataset loader | ~80 |
| `src/pjepa/data/cl_splits.py` | CL splits | ~110 |
| `src/pjepa/data/ogb.py` | OGB-arxiv loader | ~80 |
| `src/pjepa/baselines/gcn.py` | GCN | ~55 |
| `src/pjepa/baselines/gin.py` | GIN | ~75 |
| `src/pjepa/baselines/graphmae.py` | GraphMAE | ~85 |
| `src/pjepa/baselines/graphcl.py` | GraphCL | ~55 |
| `src/pjepa/baselines/infograph.py` | InfoGraph | ~50 |
| `src/pjepa/baselines/ewc.py` | EWC | ~55 |
| `src/pjepa/baselines/gem.py` | GEM | ~95 |
| `src/pjepa/training/pretrain.py` | JEPA pretraining loop | ~110 |
| `src/pjepa/training/train.py` | Supervised training loop | ~65 |
| `src/pjepa/training/eval.py` | Linear-probe evaluation | ~75 |
| `src/pjepa/training/checkpoint.py` | Save/load with sharding | ~120 |
| `src/pjepa/eval/metrics.py` | Accuracy, MPCA, forgetting | ~80 |
| `src/pjepa/eval/bootstrap.py` | Paired bootstrap CI | ~70 |
| `src/pjepa/eval/stats.py` | Wilcoxon, Bonferroni | ~70 |
| `src/pjepa/cli/app.py` | Typer CLI | ~130 |
| `src/pjepa/hardware.py` | Backend detection, probes | ~270 |
| `src/pjepa/logging_setup.py` | Structured logging | ~140 |
| `src/pjepa/config.py` | YAML config + schema | ~115 |
| `src/pjepa/exceptions.py` | PJEPAError hierarchy | ~50 |
| `src/pjepa/utils/seeding.py` | Deterministic seeding | ~85 |

Total: ~4,200 lines of production code, ~3,000 lines of tests.

## Extension Points

Three places to add new functionality:

1. **New encoder:** subclass `nn.Module`, implement `forward(graph)`. Register it via `pjepa.encoders.registry` (Phase 2).
2. **New retrieval utility:** implement the `RetrievalUtility` Protocol. Pass it to `GreedyRetrieval.select`.
3. **New baseline for SOTA comparison:** subclass `nn.Module` like the existing baselines.

## Performance Considerations

| Operation | Cost | When to optimise |
|---|---|---|
| Retrieval | $O(NB)$ | When $N > 10^4$ |
| Bisimulation | $O(B^2)$ | When $B > 256$ |
| HRG production | $O(|\mathcal{H}| \cdot B)$ | When $|\mathcal{H}| > 100$ |
| JEPA encoding | $O(B^2 d)$ | When $B > 256$ or $d > 256$ |

For very large graphs ($N > 10^6$), consider:

* Submodular streaming retrieval (Phases 5+)
* Neighbour sampling for OGB-scale
* Sharded persistent graph storage

## Where to Look Next

* [Adding a custom encoder](03_adding_an_encoder.md)
* [Adding a custom baseline](04_adding_a_baseline.md)
* [Reproducing paper results](05_reproducing_paper_results.md)
* [Eight-class test taxonomy](06_test_taxonomy.md)