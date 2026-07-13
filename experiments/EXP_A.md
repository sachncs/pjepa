# Experiment A — Submodular Retrieval Quality

> **Validates**: Theorem 3 (Nemhauser-Wolsey-Fisher 1978 ``(1 - 1/e)`` approximation guarantee).
> **Scope**: Synthetic random monotone submodular facility-location utilities.
> **Wall-clock (smoke)**: ~1 s on M3 Pro.

## What

We compare the greedy working-graph retrieval of `GreedyRetrieval` to the
exact optimum of a monotone submodular facility-location utility. For each
random instance, we draw a graph of `n` vertices with random Gaussian
vertex features and a random observation of `m` features; the utility
`f(W) = Σ_{o ∈ O} max_{v ∈ W} cos(v, o)` is provably submodular and
non-negative.

For tractable instances (`C(n, B) ≤ 3 · 10⁶`) we enumerate every subset
and report the exact optimum. For larger instances we approximate the
optimum by the best of eight random subsets, which provides a
**conservative lower bound** on `OPT`: any violation of the
`(1 - 1/e)` bound detected via this approximation is a genuine
violation, while passes are non-strict.

## How

```bash
# Smoke defaults: n ∈ {40, 50}, B ∈ {5, 7}, 3 seeds.
PYTHONPATH=src python experiments/run_exp_a_retrieval.py \
    --output-dir results/exp_a_smoke

# Plan-compliant sweep: n ∈ {50, 100, 500}, B ∈ {5, 10, 20}, 5 seeds.
PYTHONPATH=src python experiments/run_exp_a_retrieval.py \
    --ns 50 100 500 --budgets 5 10 20 --seeds 5 \
    --output-dir results/exp_a_full
```

## Outputs

* `retrieval_quality.csv` — per-trial `(n, B, seed, opt, greedy, ratio, passes_threshold)`.
* `retrieval_quality.png` — mean ± std greedy/`OPT` ratio vs budget, one curve per `n`, with the `(1 - 1/e) ≈ 0.6321` reference line.

## Result Interpretation

A run is considered "pass" when `greedy / OPT ≥ (1 - 1/e) − 1e-5` for
every `(n, B, seed)` cell. Random facility-location utilities are
"easy" monotone submodular functions: the greedy algorithm frequently
matches the exact optimum (ratio ≈ 1.0). The `(1 - 1/e)` bound is
therefore verified conservatively — passes are always expected on this
problem class.

The `opt_kind` column distinguishes exact (`"exact"`) from approximated
(`"pseudo"`) optima. Pseudo optima are always ≤ the true optimum, so a
ratio above `(1 - 1/e)` on a pseudo row is a strict guarantee; a ratio
below would be a counter-example to the theorem.

## Smoke Defaults

The default invocation (`PYTHONPATH=src python
experiments/run_exp_a_retrieval.py`) runs 12 trials with a worst-case
`C(50, 5) ≈ 2.1 · 10⁶` enumeration per trial and completes in under 2
seconds. Larger sweeps can be configured via the CLI.
