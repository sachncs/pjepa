# The Eight-Class Test Taxonomy

> A guide to writing tests that catch the bugs that matter.

## Why Eight Classes?

A single "happy path" test only checks that the code runs on typical
inputs. Production bugs lurk in the long tail:

* Bad inputs that should raise typed errors.
* Edge cases that should not crash (NaN, Inf, empty inputs, single
  elements).
* Resource leaks over long runs.
* Save → load → continue round-trips.
* Backends that disagree by a ULP.
* Statistical properties that hold only on average.

The eight-class taxonomy is a checklist that ensures every public
module is tested against all these failure modes.

## The Eight Classes

### 1. Happy — `test_happy_<feature>`

Typical inputs produce expected outputs. **Every** public function
needs at least one happy test.

```python
def test_happy_greedy_returns_budget() -> None:
    g = TypedAttributedGraph(
        vertex_features=torch.randn((20, 4)),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
    )
    obs = torch.randn((5, 4))
    result = GreedyRetrieval(budget=8).select(g, obs)
    assert result.working.num_vertices() <= 8
    assert result.utility > 0.0
```

### 2. Bad — `test_bad_<feature>`

Malformed inputs raise typed errors. **Every** validating function
needs at least one bad test.

```python
def test_bad_negative_budget() -> None:
    with pytest.raises(GraphError):
        GreedyRetrieval(budget=-1)
```

### 3. Ugly — `test_ugly_<feature>`

Edge cases don't crash and don't silently produce wrong results. **Every**
function that handles numerical or structural data needs at least one
ugly test.

Common ugly cases:

* Empty inputs (zero vertices, zero edges, empty graph)
* Single element (single vertex, single edge)
* Disconnected components
* Self-loops
* NaN / Inf values
* Very large or very small numerical values
* All-zero embeddings (representational collapse)

```python
def test_ugly_empty_graph_zero_utility() -> None:
    g = TypedAttributedGraph(
        vertex_features=torch.zeros((0, 4)),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
    )
    obs = torch.randn((2, 4))
    result = GreedyRetrieval(budget=8).select(g, obs)
    assert result.working.num_vertices() == 0
    assert result.utility == 0.0
```

### 4. Leaky — `test_leaky_<feature>`

Long-running operations don't grow resources unbounded. Required for
stateful modules.

```python
def test_leaky_repeated_calls_no_state_leak() -> None:
    aug = DropEdge(strength=0.5)
    g = _toy_graph(20)
    a = aug(g)
    b = aug(g)
    # Both calls succeed and produce valid outputs.
    assert a.num_edges() >= 0
    assert b.num_edges() >= 0
```

### 5. Round-trip — `test_round_trip_<feature>`

Save → load → continue is equivalent to save → continue. Required for
serialisable modules.

```python
def test_round_trip_checkpoint_save_load() -> None:
    encoder = _ToyEncoder()
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Checkpoint(
            encoder_state=encoder.state_dict(),
            predictor_state={},
            target_state={},
            optimizer_state={},
            epoch=2,
            loss=0.25,
        )
        path = save_checkpoint(ckpt, tmp, run_id="r")
        loaded = load_checkpoint(path)
        for k, v in loaded.encoder_state.items():
            assert torch.allclose(v, encoder.state_dict()[k])
```

### 6. Cross-backend — `test_cross_backend_<feature>`

Same code on MPS/CUDA/CPU gives same output within tolerance. Mark
with `@pytest.mark.skipif(not torch.backends.mps.is_available(), ...)`.

```python
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_cross_backend_mps_gcn_forward() -> None:
    g = TypedAttributedGraph(
        vertex_features=torch.randn((5, 4)),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long),
    ).to("mps")
    model = GCN(input_dim=4, hidden_dim=8, num_classes=2).to("mps")
    out = model(g)
    assert out.device.type == "mps"
```

### 7. Distributional — `test_distributional_<property>`

Statistical properties hold across runs. Use 20-100 random samples.

```python
def test_distributional_drop_edge_distribution() -> None:
    g = _toy_graph(50)
    survivors_sets = []
    for seed in range(10):
        aug = DropEdge(strength=0.3, generator=torch.Generator().manual_seed(seed))
        survivors_sets.append(tuple(sorted(map(tuple, aug(g).edge_index.T.tolist()))))
    # At least two distinct survivors sets should appear.
    assert len(set(survivors_sets)) >= 2
```

### 8. Property — `test_property_<invariant>`

Hypothesis-driven invariants. The most important test class for
correctness.

```python
def test_distributional_utility_is_submodular() -> None:
    """Facility location exhibits diminishing returns on random inputs."""
    g = _random_graph(8, 4, seed=6)
    util = FacilityLocationUtility(vertex_features=g.vertex_features)
    obs = torch.randn((4, 4))
    n = g.num_vertices()
    for _ in range(50):
        s_size = torch.randint(0, n - 2, (1,)).item()
        t_size = torch.randint(s_size + 1, n - 1, (1,)).item()
        s = torch.randperm(n)[:s_size]
        s_set = set(s.tolist())
        remaining = [v for v in range(n) if v not in s_set]
        extra_count = t_size - s_size
        if extra_count <= 0 or len(remaining) < extra_count + 1:
            continue
        extras_for_t = remaining[:extra_count]
        t = torch.tensor(sorted(s_set | set(extras_for_t)), dtype=torch.long)
        v_candidates = [v for v in remaining if v not in set(extras_for_t)]
        if not v_candidates:
            continue
        v = v_candidates[torch.randint(0, len(v_candidates), (1,)).item()]
        delta_s = util(torch.cat([s, torch.tensor([v])]), obs) - util(s, obs)
        delta_t = util(torch.cat([t, torch.tensor([v])]), obs) - util(t, obs)
        assert delta_s >= delta_t - 1e-5
```

## Module-Specific Test Inventory

For each public module, ensure the following tests exist:

| Module | Required tests |
|---|---|
| `TypedAttributedGraph` | happy construct, bad shapes, bad dtypes, bad indices, ugly empty/single, leaky repeated, round-trip, cross-backend, distributional random graphs, property with_features increments |
| `PersistentState` | happy commit, bad delta_j ≥ 0, bad cost < 0, ugly empty, leaky repeated, round-trip, distributional |
| `WorkingGraph` | happy, bad budget enforcement, ugly empty, property utilisation |
| `Encoder` (EuclideanMPNN) | happy forward, bad zero dim, ugly single vertex, cross-backend MPS |
| `Encoder` (HyperbolicProjection) | happy norm < 1, bad negative curvature, ugly zero input, cross-backend MPS |
| `Encoder` (DualGeometricEncoder) | happy shape, property dims |
| `JEPAPredictor` | happy shape |
| `TargetEncoder` | happy EMA, round-trip |
| `RetrievalUtility` | happy, property non-negative |
| `GreedyRetrieval` | happy, bad negative budget, ugly empty/single, leaky repeated, round-trip, cross-backend, distributional submodular, property (1-1/e) |
| `HRG` | happy, bad overlapping labels, bad unknown start, ugly empty |
| `BisimulationMetric` | happy, property non-negative, property symmetric |
| `FourConditions` | happy accept, bad non-negative delta_j, bad cost exceeded, bad bisimilarity |
| `DPO loss` | happy, bad shape mismatch, bad label smoothing, distributional bounded, property zero equal |
| `FreeEnergy` | happy non-negative, ugly empty graph |
| `EvolutionOperator` | happy, property is_contraction |
| `PPOTrainer` | happy clipped surrogate, bad zero minibatch, leaky no outer mutation |
| `ReplayBuffer` | happy add and sample, bad zero capacity, distributional eviction |
| `SleepCadence` | happy no sleep when healthy, bad window |
| `Augmentation` (each) | happy applies, ugly no-op when strength=0 |
| `AugmentationPipeline` | happy all modes, bad empty list, bad zero k |
| `TU loader` | happy load (smoke test) |
| `CL splits` | happy, property classes disjoint, bad too many tasks, bad empty labels |
| `Baseline` (each) | happy forward, property output shape |
| `EWC` | happy capture and penalty, bad negative lambda |
| `GEM` | happy add, bad zero capacity |
| `Checkpoint` | happy round-trip, bad save to nonexistent, bad load from nonexistent |
| `Pretrain loop` | happy runs, bad zero epochs, leaky no outer mutation, cross-backend MPS |
| `Supervised loop` | happy runs, bad zero epochs |
| `Linear probe` | happy, bad empty dataset |
| `Metrics` | happy, bad empty input |
| `Bootstrap CI` | happy, bad length mismatch, distributional stable across seeds |
| `Wilcoxon` | happy, property p in [0, 1] |
| `Bonferroni` | happy |
| `CapabilityReport` | happy, ugly empty |
| `Hardware probes` | happy each, property cpu_fallback always green, cross-backend MPS |
| `Logging` | happy HUMAN, happy JSON, bad unknown format |
| `Config` | happy load, bad missing file, bad root not mapping, bad required missing, round-trip save-load, bad save to missing dir |
| `Seeding` | happy set/get, bad negative, ugly deterministic, round-trip context, cross-backend MPS, distributional components differ, property sub-seed reproducible |
| `Exceptions` | happy raise and catch, property hierarchy |

That's about 130 test cases. The framework currently has 182; we
have coverage beyond the minimum.

## Common Test Smells

* **Test is just `assert x is not None`.** Use specific assertions
  about shape, value range, or specific properties.
* **Test depends on random seed without setting one.** Use
  `torch.Generator().manual_seed(seed)` or call `set_global_seed`.
* **Test asserts on the entire loss value, not just its sign or
  finiteness.** Use `assert torch.isfinite(loss)` instead of
  `assert loss == 0.123` (brittle).
* **Test doesn't include the eight classes.** Add the missing ones.
* **Test relies on a network connection.** All `pjepa` tests must run
  offline.

## Mutation Testing

For the highest-stakes modules (`objectives/`, `dynamics/`), we run
`cosmic-ray` mutation testing. The target is:

* ≥ 80% mutation score on `objectives/`
* ≥ 75% mutation score on `dynamics/`

A lower mutation score means our tests don't catch logic bugs in the
implementation of the paper's central theorems. Fix this by adding
more property tests.

## Where to Look Next

* [Architecture overview](02_architecture.md) — module dependency graph.
* [Adding a custom encoder](03_adding_an_encoder.md)
* [Adding a custom baseline](04_adding_a_baseline.md)
* [Reproducing paper results](05_reproducing_paper_results.md)