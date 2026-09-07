# INTERFACES.md — Cross-Module Interface Contract

> The contract is part of the public API. Updates require a
> deprecation cycle for any existing entry; new interfaces follow
> the same shape.

This document records the **interface contracts** between the
sub-packages of `pjepa`. Each piece is responsible for
implementing its own slice; all other slices must consume the
interface — never the concrete class — for any kind of
dependency.

---

## 1. `pjepa.graphs.Graph`

The immutable substrate for both the persistent graph `G_t` and
the working graph `W_t`. All other modules operate on this type.
The class was previously named `TypedAttributedGraph`; the
backward-compatible alias `make_typed_graph` is exported from
`pjepa.compat`.

```python
@dataclass(frozen=True)
class Graph:
    vertex_features: torch.Tensor            # [N, d_v]
    edge_index: torch.Tensor                 # [2, E], long
    edge_features: torch.Tensor              # [E, d_e]
    vertex_labels: torch.Tensor | None = None
    edge_labels: torch.Tensor | None = None
    global_features: torch.Tensor | None = None
    version: int = 0

    def num_vertices(self) -> int: ...
    def num_edges(self) -> int: ...
    def with_features(self, **kwargs) -> "Graph": ...
    def subgraph(self, vertex_mask: torch.Tensor) -> "Graph": ...
    def to(self, device: torch.device) -> "Graph": ...
```

**Invariants**

- `vertex_features.shape[0]` equals the number of vertices.
- `edge_index` is in COO format; both rows are `long` and `int64`.
- `edge_features.shape[0]` equals `edge_index.shape[1]` (`E`).
- Mutations produce a new instance (`frozen=True`).

## 2. `pjepa.encoders.Encoder`

A graph encoder maps a `Graph` to an embedding tensor. The
:class:`Encoder` is now a real `ABC` (was previously a runtime
checkable `Protocol`). The backward-compatible alias
`EncoderProtocol = Encoder` is kept.

```python
class Encoder(ABC, torch.nn.Module):
    @property
    @abstractmethod
    def output_dim(self) -> int: ...
    @abstractmethod
    def forward(self, graph: Graph) -> torch.Tensor | tuple[torch.Tensor, ...]: ...
    def encode(self, graph: Graph) -> torch.Tensor: ...   # default delegates to forward
    def summary(self) -> dict[str, Any]: ...
```

**Consumers**: retrieval, rewriting (via bisimulation metric),
the JEPA predictor. **Producers**: `Euclidean`, `Hyperbolic`,
`DualGeometric`, `Predictor`.

## 3. `pjepa.encoders.Head`

The polymorphic root of the head hierarchy. The concrete
subclasses are :class:`Predictor` (the learned predictor) and
:class:`Target` (the EMA shadow encoder). The trainer iterates
over a list of heads uniformly.

```python
class Head(ABC, torch.nn.Module):
    @abstractmethod
    def forward(self, context: torch.Tensor) -> torch.Tensor: ...
    def update(self) -> None: ...   # default no-op; overridden by Target
```

## 4. `pjepa.augmentations.Transform`

Callable `Graph → Graph`. Composed via `Pipeline`. The classes
were renamed from `Augmentation` / `AugmentationPipeline` to
`Transform` / `Pipeline`.

```python
class Transform(ABC):
    def __init__(self, strength: float = 0.2, generator: torch.Generator | None = None): ...
    @abstractmethod
    def __call__(self, graph: Graph) -> Graph: ...

class Pipeline:
    def __init__(self, augmentations, mode: str = "random_sample_one", k: int = 2, generator=None): ...
    def __call__(self, graph: Graph) -> Graph: ...
```

Built-ins: `DropEdge`, `DropNode`, `Subgraph`,
`ConnectedSubgraph`, `DropFeature`, `FeatureMask`, `Identity`
(plus tensor adapter `TensorDropFeature`).

`Pipeline(augmentations, mode, k, generator)` supports three
modes (`SEQUENTIAL`, `RANDOM_SAMPLE_ONE`, `RANDOM_SAMPLE_K`).

## 5. `pjepa.retrieval.Utility`

The polymorphic root of the retrieval-utility hierarchy. The
concrete subclasses are `Facility` (provably submodular) and
`InfoGain` (information-gain with per-vertex cost). The class
was renamed from `RetrievalUtility` to `Utility`.

```python
class Utility(ABC):
    @abstractmethod
    def __call__(self, vertex_subset: torch.Tensor, observation: torch.Tensor) -> float: ...
    def score(self, vertex_subset, observation) -> float: ...   # default delegates to __call__
```

The retriever class :class:`pjepa.retrieval.Retrieval` (was
`GreedyRetrieval`) accepts any `Utility` and returns a
:class:`Result` (was `RetrievalResult`).

## 6. `pjepa.rewriting.Criterion`

The polymorphic root of the acceptance-criterion hierarchy. The
concrete subclass is :class:`FourConditions`.

```python
class Criterion(ABC):
    @abstractmethod
    def evaluate(self, candidate, current, observation, grammar) -> tuple[bool, dict[str, object]]: ...
    def accept(self, candidate, current, observation, grammar) -> tuple[bool, dict[str, object]]: ...
```

The convenience function :func:`pjepa.rewriting.accept` wraps a
`FourConditions` instance for callers that prefer the
functional style.

## 7. `pjepa.scheduler.Storage` and `pjepa.scheduler.Cadence`

The polymorphic roots of the replay-storage and sleep-cadence
hierarchies.

```python
class Storage(ABC):
    @abstractmethod
    def add(self, step: Step) -> None: ...
    @abstractmethod
    def minibatches(self, batch_size: int) -> Iterator[tuple[torch.Tensor, ...]]: ...
    @abstractmethod
    def evict_stale(self) -> None: ...
    @abstractmethod
    def __len__(self) -> int: ...

class Cadence(ABC):
    @abstractmethod
    def should_sleep(self) -> bool: ...
    @abstractmethod
    def update(self, accepted: bool, utilisation: float) -> None: ...
    def reset(self) -> None: ...   # default no-op
```

Concrete subclasses: `Buffer` (FIFO), `Sleep` (rolling-statistic
trigger). Backward-compatible aliases: `ReplayBuffer = Buffer`,
`Transition = Step`, `SleepCadence = Sleep`.

## 8. `pjepa.hardware` capability interface

```python
def detect_backend() -> Backend: ...             # Backend ∈ {CUDA, MPS, CPU}
def current_device(backend: Backend | None = None) -> torch.device: ...
def detect_capabilities() -> CapabilityReport: ...
def sync_if_mps() -> None: ...
```

`CapabilityReport` carries a tuple of `ProbeResult` with a
status (`GREEN` / `YELLOW` / `RED`). All performance / runtime
decisions read this report before activating optimisation
paths.

## 9. `pjepa.perf` adapters

```python
def safe_compile(module: nn.Module, *, mode: str | None = None, fullgraph: bool = False) -> nn.Module: ...
def autocast_context(enabled: bool = True, dtype: torch.dtype | None = None) -> AbstractContextManager[None]: ...
class EMATarget:
    def __init__(self, online: nn.Module, momentum: float = 0.996, schedule: str = "constant", final_momentum: float = 0.999, total_steps: int = 1000): ...
    def update(self) -> None: ...
    def forward(self, *args, **kwargs): ...
def fused_scatter_add(out: Tensor, index: Tensor, src: Tensor, dim: int = 0) -> Tensor: ...
def fused_scatter_mean(out: Tensor, count: Tensor, index: Tensor, src: Tensor, dim: int = 0) -> Tensor: ...
def sync_mps() -> None: ...
class DatasetCache:
    def __init__(self, root: str | os.PathLike[str] | None = None): ...
    def has(self, key: str) -> bool: ...
    def put(self, key: str, value: object) -> Path: ...
    def get(self, key: str) -> object: ...
    def get_or_compute(self, key: str, compute: Callable[[], object]) -> object: ...
def cache_key(parts: Iterable[object]) -> str: ...
def memmap_array(path: Path, shape: tuple[int, ...], dtype: str) -> np.memmap: ...
class Microbenchmark:
    def __init__(self, name: str = "operation", n_warmup: int = 3, n_iter: int = 10): ...
    def run(self, fn: Callable[[], object]) -> MicrobenchmarkResult: ...
def compare_benchmarks(baseline: MicrobenchmarkResult, candidate: MicrobenchmarkResult) -> dict[str, float]: ...
```

The `EMATarget` perf wrapper and the `Target` JEPA target are
intentionally separate classes; `EMATarget` adds a cosine
schedule on top of the byol-style EMA that `Target` implements.
The trainer can use either depending on the regime.

## 10. `pjepa.baselines` surface

```python
class Naive(nn.Module): ...     # mean-pool linear, sanity baseline
class GCN(nn.Module): ...
class GIN(nn.Module): ...
class GraphSAGE(nn.Module): ...
class GraphCL(nn.Module): ...
class GraphMAE(nn.Module): ...
class InfoGraph(nn.Module): ...
class BGRL(nn.Module): ...
class GEM: ...                  # gradient episodic memory, buffer
class EWC:                      # continual-learning regulariser
    def capture(self, named_parameters, loss: torch.Tensor) -> None: ...
    def penalty(self, named_parameters) -> torch.Tensor: ...
    def fisher_state(self) -> dict[str, dict[str, torch.Tensor]]: ...
    def reset(self) -> None: ...
class PackNet(nn.Module): ...
```

## 11. Registries (extension points)

Every module that introduces a polymorphic interface ships a
registry. New implementations register themselves without
modifying the core library.

```python
pjepa.augmentations.register(name: str) -> Callable[[type[Transform]], type[Transform]]
pjepa.augmentations.get_augmentation(name: str) -> type[Transform]
pjepa.augmentations.available_augmentations() -> tuple[str, ...]

pjepa.encoders.register(name: str) -> Callable[[type[Encoder]], type[Encoder]]
pjepa.encoders.get_encoder(name: str) -> type[Encoder]
pjepa.encoders.available_encoders() -> tuple[str, ...]
```

## 12. Compatibility aliases (`pjepa.compat`)

```python
Graph            = Graph
PersistentGraph  = State
GraphState       = Working
PJEPAEncoder     = Encoder
PJEPATransform   = Transform
make_typed_graph(vertex_features, edge_index, edge_features=None, **kwargs) -> Graph
```

These aliases let downstream code adopt the framework without
depending on internal layout. They are stable exports.

## 13. Package version

`pjepa.__version__` is a `str` exposed by the top-level package
and re-exported from `pjepa.version.__version__`. The package
follows PEP 561 (ships `py.typed`).

---

## Change policy

| Change kind | Required action |
|---|---|
| Add a new field to `Graph` | Update this doc + revision bump |
| Add a new augmentation | Update registry list in §4 |
| Add a new baseline | Update §10 |
| Add a new alias | Append to §12 with rationale |
| Change any signature in §1–§10 | Open a deprecation PR first |

The CI workflow runs `ruff check src tests` and an *advisory*
`pytype src/pjepa` (informational only; failures do not gate PRs).
