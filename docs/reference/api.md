# API Reference

The `pjepa` API is organised into a small number of subpackages,
each with a focused responsibility. This page gives a high-level
tour; for per-symbol documentation, run `help(pjepa.X)` in a
Python REPL.

## Top-level

| Symbol | Description |
|---|---|
| `pjepa.__version__` | The current package version (e.g., `"1.0.0"`). |

## `pjepa.graphs`

The persistent and working graph primitives.

| Symbol | Description |
|---|---|
| `Graph` | Immutable typed attributed graph dataclass. |
| `State` | Wrapper around the persistent graph with commit/reject audit trail. |
| `Working` | Bounded working subgraph enforced to \|V\| ≤ `budget`. |

See [`paper/graphs.md`](../paper/graphs.md) for the architectural
rationale.

## `pjepa.encoders`

Encoder polymorphic roots and concrete implementations.

| Symbol | Description |
|---|---|
| `Encoder` | ABC — every encoder must satisfy this polymorphic root. |
| `EncoderProtocol` | Backward-compatibility alias for `Encoder`. |
| `Euclidean` | GIN-style message-passing neural network. |
| `Hyperbolic` | Maps Euclidean features into the Poincaré ball. |
| `DualGeometric` | Combines Euclidean MPNN with hyperbolic projection. |
| `Head` | ABC — predictor / target polymorphic root. |
| `Predictor` | Predictor head for the JEPA objective. |
| `Target` | EMA target encoder. |

## `pjepa.retrieval`

Submodular retrieval with the (1 − 1/e) approximation guarantee.

| Symbol | Description |
|---|---|
| `Retrieval` | Greedy algorithm that achieves the NWF bound. |
| `Utility` | ABC — every retrieval utility must satisfy this polymorphic root. |
| `Facility` | Provably submodular facility-location utility. |
| `InfoGain` | Information-gain utility with per-vertex cost. |
| `Result` | Dataclass returned by `Retrieval.select`. |
| `uniform_weights(n)` | Return uniform per-vertex weights. |
| `facility_location_weights(features, observation)` | Cosine-similarity weights. |

## `pjepa.rewriting`

Verified rewriting engine.

| Symbol | Description |
|---|---|
| `HRG` | Hyperedge-replacement grammar class. |
| `HRGProduction` | A single production rule. |
| `Bisimulation` | Configuration for the bisimulation metric. |
| `bisimulation_distance(graph_a, graph_b, metric)` | Compute the bisimulation pseudometric. |
| `Criterion` | ABC — every acceptance criterion must satisfy this polymorphic root. |
| `FourConditions` | The four acceptance thresholds. |
| `accept(candidate, current, observation, grammar, thresholds)` | Evaluate the four-conditions criterion. |
| `compute_delta_j(candidate, current, observation, beta_ib, lambda_mdl, gamma_forward)` | Compute `Δ𝒥`. |
| `DPOConfig` | Configuration for the DPO loss. |
| `dpo_loss(...)` | Compute the DPO loss for preference pairs. |

## `pjepa.objectives`

The unified free-energy functional and its components.

| Symbol | Description |
|---|---|
| `FreeEnergy` | The four-term `𝒥` functional. |
| `ib_lagrangian(ix_z, iy_z, beta)` | Symbolic IB Lagrangian. |
| `variational_ib_bound(posterior_logits, prior_logits, beta)` | Variational IB upper bound. |
| `description_length(graph)` | Description length of a graph under MDL. |

## `pjepa.dynamics`

Evolution operator analysis (Propositions 4–6 of the paper).

| Symbol | Description |
|---|---|
| `EvolutionOperator` | Configuration for analysing `F`. |
| `contractivity_bound(eta_g, eta_o, epsilon, t)` | Upper bound on trajectory distance. |
| `fixed_point_iteration(state, operator, max_steps, epsilon)` | Iterate until a fixed point. |

## `pjepa.scheduler`

PPO scheduler with replay buffer and sleep cadence.

| Symbol | Description |
|---|---|
| `Storage` | ABC — every replay storage backend must satisfy this polymorphic root. |
| `Buffer` | FIFO replay buffer (concrete `Storage`) with staleness eviction. |
| `Step` | A single replay-buffer transition. |
| `Cadence` | ABC — every sleep cadence must satisfy this polymorphic root. |
| `Sleep` | Rolling-statistic sleep cadence (concrete `Cadence`). |
| `PPOConfig` | PPO hyperparameters. |
| `PPOTrainer` | Clipped-surrogate PPO trainer. |

## `pjepa.augmentations`

Graph and tensor augmentations.

| Symbol | Description |
|---|---|
| `Transform` | ABC — every augmentation must satisfy this polymorphic root. |
| `Pipeline` | Sequential / random-sample-one / random-sample-k composition. |
| `PipelineMode` | Composition-mode enumeration. |
| `DropEdge(strength)` | Drop a fraction of edges. |
| `DropNode(strength)` | Drop a fraction of vertices. |
| `Subgraph(strength)` | Vertex-induced random-sample subgraph. |
| `ConnectedSubgraph(strength)` | Vertex-induced BFS-connected subgraph. |
| `DropFeature(strength)` | Zero a fraction of feature dimensions. |
| `FeatureMask(feature_dim, strength)` | Replace features with a learnable mask token. |
| `TensorDropFeature(strength)` | Tensor-compatible feature drop for non-graph models. |
| `Identity()` | No-op augmentation. |

## `pjepa.training`

Training loops and wrappers.

| Symbol | Description |
|---|---|
| `PretrainConfig` | JEPA-pretraining configuration. |
| `pretrain_loop(...)` | JEPA-style pretraining loop. |
| `SupervisedConfig` | Supervised-training configuration. |
| `supervised_train_loop(...)` | Supervised training loop. |
| `LinearProbeResult` | Result of a linear-probe evaluation. |
| `linear_probe_eval(...)` | Linear-probe evaluation. |
| `Checkpoint` | In-memory checkpoint representation. |
| `save_checkpoint(...)` | Save a checkpoint to disk. |
| `load_checkpoint(...)` | Load a checkpoint from disk. |
| `SWAConfig` | SWA configuration. |
| `SWAWrapper` | Stochastic Weight Averaging wrapper. |
| `TTAConfig` | TTA configuration. |
| `TTAWrapper` | Test-Time Augmentation wrapper. |
| `Aggregator` | Ensemble aggregation-mode enumeration. |
| `Ensemble` | k-model ensemble. |
| `DistillationConfig` | Distillation configuration. |
| `DistillationLoss` | Combined task + distillation loss. |
| `distill_kl(...)` | Temperature-scaled KL divergence. |

## `pjepa.eval`

Evaluation metrics, bootstrap CI, and statistical tests.

| Symbol | Description |
|---|---|
| `accuracy(predictions, targets)` | Fraction of correct predictions. |
| `mean_per_class_accuracy(predictions, targets)` | Average per-class accuracy. |
| `forgetting_rate(per_task_accuracies)` | Average forgetting rate. |
| `backward_transfer(per_task_accuracies)` | Backward-transfer metric. |
| `forward_transfer(per_task_accuracies, baseline_per_task)` | Forward-transfer metric. |
| `BootstrapCI` | Result of a paired bootstrap computation. |
| `paired_bootstrap_ci(...)` | Paired percentile bootstrap CI for the difference in means. |
| `wilcoxon_signed_rank(scores_a, scores_b)` | Two-sided p-value. |
| `permutation_pvalue(diffs, n_resamples, seed)` | Sign-permutation fallback. |
| `bonferroni_correction(p_values)` | Adjust p-values for multiple comparisons. |
| `AggregatedRow` | One row of the per-experiment aggregator output. |
| `AggregationResult` | Aggregator output bundle. |
| `aggregate_all(results_dir)` | Walk `results_dir`, write JSONL / CSV / Markdown artefacts. |

## `pjepa.perf`

Performance infrastructure (capability-aware with graceful fallback).

| Symbol | Description |
|---|---|
| `safe_compile(module, mode, fullgraph)` | Backend-aware `torch.compile` wrapper. |
| `autocast_context(enabled, dtype)` | Backend-aware mixed-precision context manager. |
| `EMATarget` | EMA target wrapper with optional cosine schedule. |
| `fused_scatter_add(out, index, src, dim)` | Fused scatter-add. |
| `fused_scatter_mean(out, count, index, src, dim)` | Fused scatter-mean. |
| `sync_mps()` | Explicit MPS synchronisation. |

## `pjepa.data`

Dataset loaders.

| Symbol | Description |
|---|---|
| `TUGraph` | A single TUDataset graph. |
| `load_tu_dataset(name, root, verify_checksum)` | Load a TUDataset. |
| `ClassIncrementalSplit` | Class-incremental split representation. |
| `make_class_incremental_split(labels, num_tasks, seed_split)` | Construct a class-incremental split. |
| `OGBArxiv` | The OGB-Arxiv dataset. |
| `load_ogb_arxiv(root)` | Load OGB-Arxiv with test-label isolation. |
| `CSRAdj` | Named-tuple CSR adjacency representation. |
| `precompute_adjacency(graph)` | CSR adjacency for fast neighbour sampling. |
| `NeighborSample` | Sampled-subgraph result. |
| `neighbor_sample(...)` | Multi-hop random neighbour sampling. |

## `pjepa.baselines`

Published baselines for SOTA comparison.

| Symbol | Description |
|---|---|
| `Naive` | Mean-pool baseline. |
| `GCN` | Kipf & Welling GCN baseline. |
| `GIN` | Graph Isomorphism Network with optional virtual node. |
| `GraphSAGE` | Hamilton et al. GraphSAGE. |
| `GraphCL` | Contrastive learning with NT-Xent loss. |
| `GraphMAE` | Masked autoencoder for graphs. |
| `InfoGraph` | Mutual-information maximisation. |
| `BGRL` | Bootstrap Graph Latents. |
| `EWC` | Elastic Weight Consolidation regulariser. |
| `GEM` | Gradient Episodic Memory. |
| `PackNet` | Per-task parameter slicing. |

## `pjepa.cli`

Typer-based command-line interface.

See [`cli.md`](cli.md) for the full CLI reference.

## `pjepa.compat`

Backward-compatible aliases for renamed symbols.

| Alias | Maps to |
|---|---|
| `Graph` | `Graph` (no change) |
| `PersistentGraph` | `State` |
| `GraphState` | `Working` |
| `PJEPAEncoder` | `Encoder` |
| `PJEPATransform` | `Transform` |
| `make_typed_graph(...)` | `Graph(...)` constructor wrapper |

## `pjepa.exceptions`

Typed error hierarchy.

| Exception | When raised |
|---|---|
| `PJEPAError` | Base class for all pjepa errors. |
| `ConfigError` | Configuration is missing, malformed, or invalid. |
| `DataError` | Dataset cannot be loaded, parsed, or validated. |
| `GraphError` | Graph violates a structural invariant. |
| `NumericalError` | Numerical operation produces non-finite values. |
| `ContractError` | Protocol not satisfied by a supposed implementation. |
| `CheckpointError` | Checkpoint cannot be saved, loaded, or resumed. |
| `BackendError` | Compute backend cannot perform a required operation. |

## `pjepa.hardware`

Backend detection and capability reporting.

| Symbol | Description |
|---|---|
| `Backend` | Compute-backend enumeration. |
| `ProbeStatus` | Probe-result enumeration. |
| `ProbeResult` | Single probe result. |
| `CapabilityReport` | Aggregated capability report. |
| `detect_backend()` | Detect the most capable backend. |
| `detect_capabilities()` | Build the full capability report. |
| `current_device(backend)` | Get the default device for a backend. |
| `sync_if_mps()` | Sync MPS if MPS is active. |
| `capabilities_as_dict(report)` | JSON-friendly mapping of the report. |

## `pjepa.logging_setup`

Structured logging.

| Symbol | Description |
|---|---|
| `LogFormat` | Log format enumeration. |
| `LOG_FORMAT_HUMAN` | Constant for the human-readable format. |
| `LOG_FORMAT_JSON` | Constant for the machine-readable format. |
| `HumanLogFormatter` | The single-line human format. |
| `JsonLogFormatter` | The JSON-lines machine format. |
| `configure_logging(level, fmt)` | Configure the package logger. |
| `get_logger(name)` | Get a logger under the `pjepa` namespace. |
| `log_event(logger, event, **fields)` | Emit a structured event with keyword fields. |

## `pjepa.config`

YAML configuration loading.

| Symbol | Description |
|---|---|
| `ConfigSchema` | Required-section schema. |
| `load_config(path, schema)` | Load and validate a YAML config. |
| `save_config(config, path)` | Save a config to YAML. |
| `merge_configs(*configs)` | Deep-merge multiple configs. |

## `pjepa.utils.seeding`

Deterministic seeding.

| Symbol | Description |
|---|---|
| `set_global_seed(seed)` | Set the global seed for Python, NumPy, and PyTorch. |
| `get_global_seed()` | Get the current global seed. |
| `seed_for(component, base)` | Derive a deterministic sub-seed. |
| `current_seed()` | Alias for `get_global_seed`. |
