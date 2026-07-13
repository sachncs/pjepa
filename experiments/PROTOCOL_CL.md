# Phase 9 — Continual-Learning SOTA Protocol

This document specifies the continual-learning (CL) protocol used to
compare Persistent-JEPA against the published CL baselines (Naive
fine-tune, EWC, GEM, PackNet-style masks). It complements
`plans/phase_9/plan.md`; the experiment runner lives at
`experiments/run_exp_e_continual.py` and the headline outputs at
`results/cl/tables/cl_summary.csv` and
`results/cl/plots/cl_forgetting_curves.png`.

The per-method CL state (EWC Fisher / ``θ*``,
GEM memory, PackNet masks, Persistent-JEPA graph) is preserved
**across tasks within the same `(dataset, method, seed)` cell**;
otherwise the comparison would silently degrade to the Naive
baseline and the protocol would lose its statistical signal.

---

## 1. Splits

Class-incremental splits are constructed by
`pjepa.data.cl_splits.make_class_incremental_split(labels, num_tasks, seed_split)`.
Every class is assigned to exactly one task; tasks receive a roughly
equal share of classes. The split is deterministic in `seed_split`, so
the same split is reproduced across methods and seeds.

The Phase 9 plan defaults to five tasks (`n_tasks = 5`) on three TU
datasets: PROTEINS, MUTAG, NCI1. `CLExperimentConfig.datasets`,
`CLExperimentConfig.n_tasks`, and `CLExperimentConfig.n_seeds` expose
the defaults; the runner also accepts `--datasets`, `--n-tasks`,
`--seeds`, and `--epochs` on the CLI.

## 2. Methods

Five methods are compared. Every method trains the same backbone
(`DualGeometricEncoder` → mean-pool → MLP head) and the same loss
(Cross-Entropy on integer class labels).

| Method | Mechanism | State carried across tasks |
|---|---|---|
| Naive | Sequential fine-tuning; no CL strategy. | None. |
| EWC | Elastic Weight Consolidation. The diagonal Fisher information is accumulated during training and used as a quadratic penalty on parameter drift. The Fisher diagonal and the reference parameter snapshot are stored on the `EWC` instance. | `EWC._fisher`, `EWC._star`. |
| GEM | Gradient Episodic Memory. The classifier gradient is projected onto the half-space that does not increase loss on memory samples. The memory buffer is a deque of `(feature, label)` pairs. | `GEM.memory`. |
| PackNet | Per-task binary masks; the current task owns a disjoint slice of every parameter tensor and prior slices are frozen. The masks are deterministic per `(task_idx, num_tasks)`. | `PackNet._task_masks`, `PackNet._frozen_mask`. |
| PersistentJEPA | The persistent graph is the Knoblauch sufficient statistic. A bounded working graph is retrieved from the persistent state via `GreedyRetrieval`; the working-graph vertices are prepended to the candidate graph and committed back to `PersistentState` after training. | `PersistentState` (graph + history). |

Mathematical summary:

* **EWC**: penalty `L_ewc = λ · Σ_i F_i (θ_i - θ*_i)²`, with `F_i`
  the empirical diagonal Fisher information (mean of squared
  per-step gradients) and `θ*_i` the parameter value at the end of
  the most recent task. Both are refreshed on the `EWC` instance
  per task via `set_fisher_state`.
* **GEM**: Lagrangian projection `min_{ĝ} ½‖ĝ - g‖²  s.t.
  ⟨ĝ, ∇L_mem⟩ ≥ 0`. Closed-form solution via
  `gem.project_gradient`.
* **PackNet**: binary mask `M_t ∈ {0,1}^|θ|`; gradient update is
  masked so `θ_new = θ - η · (M_t ∨ ¬F_{<t}) ∘ ∇L`, where `F_{<t}`
  is the frozen mask accumulated from prior tasks.
* **Persistent-JEPA**: each task's mean-pooled observation is
  committed to the persistent state via
  `PersistentState.commit(candidate, cost, timestamp)`. The
  working graph is the greedy solution to a submodular maximisation
  (Algorithm 1 in the paper) and contributes its vertices to the
  next-state candidate so the persistent state genuinely
  influences subsequent tasks.

## 3. Per-task protocol

For each `(dataset, method, seed)`:

1. Build one `(EWC, GEM, PackNet, PersistentState)` per cell. Each
   is the canonical state carrier; reusing a single instance across
   tasks is what makes EWC / GEM / PackNet / Persistent-JEPA
   genuinely continual-learning methods.
2. Build the backbone and head with `set_global_seed(seed * 7919)`.
3. For each `task_idx ∈ [0, n_tasks)`:
   1. Split the task's indices into 80% train / 20% test
      deterministically.
   2. Train for `epochs_per_task` epochs on the train half
      (method-specific). The EWC / GEM / PackNet trainers
      update the cell-level state after training each task, so the
      Fisher / memory / masks propagate forward.
   3. Evaluate on the test halves of every task seen so far
      (`seen_idx ∈ [0, task_idx + 1]`).
   4. Record the per-task evaluation vector.
4. The final snapshot's `per_task_accuracies` is a `[T][T]`
   ragged matrix where the entry at position `[i][j]` (after
   normalisation via `build_accuracy_matrix`) is the accuracy on
   task `i` after training task `j`.

## 4. Metrics

The matrix `R` feeds the standard CL metrics, all in `pjepa.eval`:

- `forgetting_rate(R)` — mean over tasks `i` of
  `max(R[i][:i+1]) - R[i][-1]`. Positive values indicate
  catastrophic forgetting.
- `backward_transfer(R)` — `-forgetting_rate(R)`. Standard CL
  sign convention (negative = forgetting).
- `forward_transfer(R)` — mean over training steps `j > 0` of
  `R[j][j-1] - baseline[j]`. Positive values indicate positive
  transfer; the baseline defaults to zero.

`build_accuracy_matrix` (renamed from `_square_matrix`) fills the
lower triangle (`j < i`) with the diagonal value
`R[i][i] = snapshot_i[i]` — the accuracy on task `i` immediately
after training task `i`. This is a stable, non-fabricated
reference; it makes the matrix square without altering any metric
value because `forgetting_rate` only reads `R[i][:i+1]` and
`R[i][-1]`.

## 5. Aggregation

`aggregate_cl_results(rows)` produces a `(dataset, method)` summary
with mean ± std accuracy, the mean forgetting / backward / forward
transfer, a paired bootstrap CI of the accuracy against the Naive
baseline (`paired_bootstrap_ci`), the Wilcoxon signed-rank p-value
(`wilcoxon_signed_rank`), and the Bonferroni-adjusted p-value
(`bonferroni_correction`) across all `(dataset, method)` cells.

**Pairing by seed.** Every method-vs-Naive comparison uses
`{seed_index → accuracy}` matched pairs; the Naive accuracies are
looked up by **seed** rather than being sliced by length. This is
what gives the bootstrap CI and Wilcoxon test their statistical
power. `Naive` is selected by `method == "Naive"` (and only
Naive),not by string prefix; the Naive accuracy surface contains
exactly one entry per `(dataset, seed)` cell.

## 6. Outputs

| Path | Format | Description |
|---|---|---|
| `results/cl/cl_results.csv` | long | One row per `(dataset, method, seed, task)`. |
| `results/cl/tables/cl_<dataset>_<method>.csv` | wide | Per-seed `[T][T]` accuracy matrix and transfer metrics. |
| `results/cl/tables/cl_summary.csv` | wide | Headline table: mean ± std accuracy, BWT, FWT, bootstrap CI, Wilcoxon / Bonferroni p-values. |
| `results/cl/plots/cl_forgetting_curves.png` | figure | Per-method mean-accuracy-on-task-`i` curve, one panel per dataset. |

## 7. Smoke mode

`CLExperimentConfig(smoke=True)` collapses the experiment to a
single dataset (MUTAG), the `"Naive"` and `"PersistentJEPA"`
methods only, a single seed, two tasks, and two epochs per task.
Smoke **always overrides** `datasets / methods / n_tasks /
n_seeds / epochs_per_task` when `smoke=True` — the override is
the smoke configuration's whole point, regardless of any other
values the caller passed. Smoke runs to completion in a few
seconds and is the canonical CI entry point:

```bash
python experiments/run_exp_e_continual.py --smoke --output-dir results/cl_smoke
```

## 8. Reproducibility

- Splits are deterministic in `(dataset, seed_split)`;
  `seed_split = seed` for every `(dataset, seed)` cell.
- Model initialisation uses `set_global_seed(seed * 7919)` before
  the per-`(dataset, method, seed)` loop.
- The CL trainer does not consume RNG in any method-specific way
  beyond the standard `torch` initialisation and `AdamW` step.

## 9. Statistical rigour

- Bootstrap CI: `paired_bootstrap_ci` with `n_resamples=2000`,
  paired by seed against the Naive baseline.
- Pairwise comparison: `wilcoxon_signed_rank`, again paired by
  seed against the Naive baseline.
- Multiple-comparison correction: `bonferroni_correction` across
  all `(dataset, method)` cells in the summary.

## 10. Acceptance criteria

A run is considered successful when:

1. `cl_summary.csv` is non-empty and has all twelve expected
   columns.
2. `cl_forgetting_curves.png` is a valid PNG with non-trivial
   size.
3. `cl_results.csv` has one row per `(dataset, method, seed,
   task)` combination actually executed.
4. The smoke configuration runs to completion in under 30
   seconds.
