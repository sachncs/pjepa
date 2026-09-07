# CLI Reference

The `pjepa` command-line interface exposes the canonical workflow. Run `pjepa --help` for the top-level summary.

## Synopsis

```
pjepa [--log-format {HUMAN,JSON}] [--log-level LEVEL] COMMAND [ARGS]
```

### Global options

| Option | Description |
|---|---|
| `--log-format {HUMAN,JSON}` | Output format (default: HUMAN). |
| `--log-level LEVEL` | One of DEBUG, INFO, WARNING, ERROR, CRITICAL (default: INFO). |
| `--version` | Print version and exit. |

## Commands

### `pjepa doctor`

Print the capability probe report. Six probes are exercised: matrix multiplication, scatter-add, `torch.compile`, hyperbolic, PyG scatter, CPU fallback. Each probe is reported as GREEN (working), YELLOW (degraded), or RED (broken) with a one-line explanation.

**Exit code:** non-zero if any probe is RED.

```bash
pjepa doctor
```

### `pjepa hardware`

Print a one-line summary of the detected compute backend.

```bash
pjepa hardware
# backend=mps device=mps
```

### `pjepa benchmark <name>`

Run a cheap validation benchmark on the local machine.

| Argument | Choices | Description |
|---|---|---|
| `name` | `retrieval`, `distortion`, `encoder-ablation` | Which benchmark to run. |

Each benchmark prints a structured JSON summary to stdout.

```bash
pjepa benchmark retrieval
pjepa benchmark distortion
pjepa benchmark encoder-ablation
```

#### Benchmark: `retrieval`

Validates Theorem 3 (the (1 − 1/e) approximation guarantee) by comparing greedy retrieval to the brute-force optimum on random monotone submodular functions.

**Output:** JSON object with a `rows` list (per-seed results) and an `all_pass` boolean.

#### Benchmark: `distortion`

Validates Proposition 7 (hyperbolic vs Euclidean distortion) by computing per-edge distortion on synthetic b-ary trees of varying depths.

**Output:** JSON object with a `rows` list (per-tree results).

#### Benchmark: `encoder-ablation`

Validates Proposition 3 (Hierarchical Consistency) by training three encoder variants on synthetic AST depth-prediction.

**Output:** JSON object with a `rows` list of `{encoder, accuracy}` entries.

### `pjepa pretrain <config>`

Pretrain a JEPA encoder using the supplied config file. (Implementation deferred to Phase 5; emits a structured log message for now.)

| Argument | Description |
|---|---|
| `config` | Path to a YAML configuration file. |

### `pjepa train <dataset> <config>`

Train (supervised or continual) on the named dataset family.

| Argument | Choices | Description |
|---|---|---|
| `dataset` | `tu`, `cl`, `ogb` | Which dataset family. |
| `config` | Path to a YAML configuration file. |

The subcommand dispatches to the corresponding experiment runner:
- `tu` → `experiments/run_exp_d_tu_sota.py`
- `cl` → `experiments/run_exp_e_continual.py`
- `ogb` → `experiments/run_exp_f_ogb_arxiv.py`

### `pjepa eval <dataset> <run-dir>`

Evaluate a saved checkpoint on the named dataset family. (Implementation deferred to Phase 5; emits a structured log message for now.)

| Argument | Description |
|---|---|
| `dataset` | One of `tu`, `cl`, `ogb`. |
| `run-dir` | Path to a checkpoint directory. |

## Examples

```bash
# Verify the environment
pjepa doctor

# Run the cheap validation suite
pjepa benchmark retrieval
pjepa benchmark distortion
pjepa benchmark encoder-ablation

# Run the headline TU SOTA experiment
pjepa train tu configs/tu.yaml

# Run the continual-learning experiment
pjepa train cl configs/cl.yaml
```

## Exit Codes

| Code | Meaning |
|---|---|
| 0 | Success. |
| 2 | A capability probe reported RED, or an unknown argument was supplied. |
| Non-zero | An unrecoverable error occurred during benchmark/training/eval. |