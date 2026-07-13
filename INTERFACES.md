# INTERFACES.md — Cross-Module Interface Contract (Phases 0–3)

> Frozen in Phase 0. Updates require a Phase-0 sub-agent sign-off.
> New interfaces follow the same shape; existing ones never change
> without a deprecation cycle.

This document records the **interface contracts** between the
sub-packages of `pjepa` that Phases 0 through 3 introduce or depend
on. Each sub-agent is responsible for implementing its own slice;
all other slices must consume the interface — never the concrete
class — for any kind of dependency.

---

## 1. `pjepa.graphs.TypedAttributedGraph`

The immutable substrate for both the persistent graph `G_t` and the
working graph `W_t`. All other modules operate on this type.

```python
@dataclass(frozen=True)
class TypedAttributedGraph:
    vertex_features: torch.Tensor            # [N, d_v]
    edge_index: torch.Tensor                 # [2, E], long
    edge_features: torch.Tensor              # [E, d_e]
    vertex_labels: torch.Tensor | None = None
    edge_labels: torch.Tensor | None = None
    global_features: torch.Tensor | None = None
    version: int = 0

    def num_vertices(self) -> int: ...
    def num_edges(self) -> int: ...
    def with_features(self, **kwargs) -> "TypedAttributedGraph": ...
    def subgraph(self, vertex_mask: torch.Tensor) -> "TypedAttributedGraph": ...
    def to(self, device: torch.device) -> "TypedAttributedGraph": ...
```

**Invariants**

- `vertex_features.shape[0]` equals the number of vertices.
- `edge_index` is in COO format; both rows are `long` and `int64`.
- `edge_features.shape[0]` equals `edge_index.shape[1]` (`E`).
- Mutations produce a new instance (`frozen=True`).

## 2. `pjepa.encoders.Encoder` (and `EncoderProtocol`)

A graph encoder maps a `TypedAttributedGraph` to an embedding tensor.
Implementations typically subclass `torch.nn.Module`; the protocol is
runtime-checkable.

```python
@runtime_checkable
class Encoder(Protocol):
    output_dim: int
    def forward(self, graph: TypedAttributedGraph) -> torch.Tensor: ...
    def to(self, device: torch.device) -> Encoder: ...
```

`EncoderProtocol = Encoder` (alias) is exported for documentation.

**Consumers**: retrieval, rewriting (via bisimulation metric), the
JEPA predictor. **Producers**: `EuclideanMPNN`, `HyperbolicProjection`,
`DualGeometricEncoder`, `JEPAPredictor`.

## 3. `pjepa.augmentations.Augmentation`

Callable `Graph → Graph`. Composed via `AugmentationPipeline`.

```python
class Augmentation(ABC):
    def __init__(self, strength: float = 0.2, generator: torch.Generator | None = None): ...
    @abstractmethod
    def __call__(self, graph: TypedAttributedGraph) -> TypedAttributedGraph: ...
```

Built-ins: `DropEdge`, `DropNode`, `Subgraph`, `RandomWalkSubgraph`,
`DropFeature`, `FeatureMask`, `Identity` (plus tensor adapter
`TensorDropFeature`).

`AugmentationPipeline(augmentations, mode, k, generator)` supports
three modes (`SEQUENTIAL`, `RANDOM_SAMPLE_ONE`, `RANDOM_SAMPLE_K`).

## 4. `pjepa.hardware` capability interface

```python
def detect_backend() -> Backend: ...             # Backend ∈ {CUDA, MPS, CPU}
def current_device(backend: Backend | None = None) -> torch.device: ...
def detect_capabilities() -> CapabilityReport: ...
def sync_if_mps() -> None: ...
```

`CapabilityReport` carries a tuple of `ProbeResult` with a status
(`GREEN` / `YELLOW` / `RED`). All performance / runtime decisions
in Phases 1–3 read this report before activating optimisation paths.

## 5. `pjepa.perf` adapters

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

## 6. `pjepa.baselines` surface

```python
class Naive(nn.Module): ...     # mean-pool linear, sanity baseline
class GCN(nn.Module): ...
class GIN(nn.Module): ...
class GEM: ...                  # gradient episodic memory, buffer
class EWC:                      # continual-learning regulariser
    def capture(self, named_parameters, loss: torch.Tensor) -> None: ...
    def penalty(self, named_parameters) -> torch.Tensor: ...
    def fisher_state(self) -> dict[str, dict[str, torch.Tensor]]: ...
    def reset(self) -> None: ...
class GraphCL(nn.Module): ...
class GraphMAE(nn.Module): ...
class InfoGraph(nn.Module): ...
class PackNet(nn.Module): ...
```

## 7. Registries (extension points)

Every module that introduces a polymorphic interface ships a
registry. New implementations register themselves without
modifying the core library.

```python
pjepa.augmentations.register(name: str) -> Callable[[type[Augmentation]], type[Augmentation]]
pjepa.augmentations.get_augmentation(name: str) -> type[Augmentation]
pjepa.augmentations.available_augmentations() -> tuple[str, ...]

pjepa.encoders.register(name: str) -> Callable[[type[Encoder]], type[Encoder]]
pjepa.encoders.get_encoder(name: str) -> type[Encoder]
pjepa.encoders.available_encoders() -> tuple[str, ...]
```

## 8. Compatibility aliases (`pjepa.compat`)

```python
Graph            = TypedAttributedGraph
PersistentGraph  = PersistentState
GraphState       = WorkingGraph
PJEPAEncoder     = Encoder
PJEPAAugmentation = Augmentation
make_typed_graph(vertex_features, edge_index, edge_features=None, **kwargs) -> TypedAttributedGraph
```

These aliases let downstream code adopt the framework without
depending on internal layout. They are stable exports.

## 9. Package version

`pjepa.__version__` is a `str` exposed by the top-level package and
re-exported from `pjepa._version.__version__`. The package follows
PEP 561 (ships `py.typed`).

---

## Change policy

| Change kind | Required action |
|---|---|
| Add a new field to `TypedAttributedGraph` | Update this doc + revision bump |
| Add a new augmentation | Update registry list in §3 |
| Add a new baseline | Update §6 |
| Add a new alias | Append to §8 with rationale |
| Change any signature in §1–§6 | Open a deprecation PR first |

The CI workflow runs `ruff check src tests` and an *advisory*
`pytype src/pjepa` (informational only; failures do not gate PRs).
