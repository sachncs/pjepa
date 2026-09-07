# Dynamics Module

> Analyses of the evolution operator $F = \Phi \circ \mathcal{R}$.

The persistent graph evolves by a discrete dynamical system:

$$G_{t+1} = F(G_t, O_t) = \Phi(\mathcal{R}(G_t, O_t), G_t).$$

This module provides the analytical tools (Propositions 4–6 of the paper) for reasoning about the long-term behaviour of $F$.

## Assumptions

| Assumption | Statement |
|---|---|
| **A1 (Compactness)** | $\mathcal{K} = \{G : \mathcal{J}(G) \le \mathcal{J}(G_0),\, \mathrm{DL}(G) \le M\}$ is non-empty and finite. |
| **A2 (Bounded Rewrite Error)** | There exist $\eta_G, \eta_O, \eta_{\text{cost}} > 0$ such that $F$ is jointly Lipschitz and $\mathcal{K}$-closed. |

## Main Results

| Result | Statement |
|---|---|
| **Proposition 4** (Existence) | Under A1, $G_t$ attains a fixed point in $\le \|\mathcal{K}\|$ accepted steps. |
| **Proposition 5** (Joint Lipschitz) | $d(F(G, O), F(G', O')) \le \eta_G \cdot d(G, G') + \eta_O \cdot \|O - O'\|$. |
| **Proposition 6** (Contraction) | If $\eta_G < 1$, then $d(G_t, G'_t) \le \eta_G^t \cdot d(G_0, G'_0) + \frac{\eta_O \cdot \varepsilon}{1 - \eta_G}$. |

## Why this matters

These results give the framework *worst-case* guarantees on convergence and stability. In practice the constants are small (companion Paper 4 measures $\eta_G \in [0.3, 0.8]$ empirically), so convergence is fast.

## Implementation

| Symbol | Description |
|---|---|
| `EvolutionOperator` | Configuration for the analysis (Lipschitz constants, etc.). |
| `contractivity_bound(eta_g, eta_o, epsilon, t)` | Upper bound on $d(G_t, G'_t)$. |
| `fixed_point_iteration(state, operator, max_steps, epsilon)` | Iterate $F$ until a fixed point. |

### Example

```python
from pjepa.dynamics import EvolutionOperator, contractivity_bound

op = EvolutionOperator(eta_g=0.5, eta_o=0.1)
bound = contractivity_bound(0.5, 0.1, 0.05, 20)
assert bound < 0.2  # Trajectories contract.
```

## Open Questions

* A theoretical justification of the empirical range $\eta_G \in [0.3, 0.8]$ (companion Paper 4).
* An adaptive controller for $\eta_G$ during training.

## Where to Look Next

* [Graphs](graphs.md) — the state being evolved.
* [Verified Rewriting](verified-rewriting.md) — the discrete-step generator of $F$.
* [Scheduler](../researcher/01_persistent_graph_world_model.md) — the policy that selects rewrites.