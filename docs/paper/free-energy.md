# Free-Energy Functional 𝒥

> The unified variational objective that governs every component.

## Definition

The framework's central object is the four-term free-energy functional (paper §2.7):

$$\boxed{\;\mathcal{J}(G) \;=\; \underbrace{\mathbb{E}\!\bigl[-\log p(O \mid G)\bigr]}_{\text{predictive fit}} \;+\; \underbrace{\beta \cdot D_{\mathrm{KL}}\!\bigl(q(G)\,\|\,p(G)\bigr)}_{\text{complexity}} \;+\; \underbrace{\lambda \cdot \mathrm{DL}(G)}_{\text{description length}} \;-\; \underbrace{\gamma \cdot I(G;\, O_{>t})}_{\text{forward information}}\;}$$

| Term | Provenance | Role |
|---|---|---|
| Predictive fit | Friston 2010 accuracy | How well $G$ predicts observations. |
| KL complexity | Tishby 1999 IB Lagrangian | Bounded deviation from prior. |
| Description length | Rissanen 1978 MDL | Bounded graph size. |
| Forward information | Friston 2017 epistemic value | Information about the future. |

## Acceptance Criterion

Every candidate rewrite $\widehat{G}_{t+1}$ is accepted iff

$$\Delta \mathcal{J} \;:=\; \mathcal{J}(\widehat{G}_{t+1}) - \mathcal{J}(G_t) \;<\; 0.$$

Combined with the four-conditions criterion (§7.7), this gives strict monotonic descent of $\mathcal{J}$ across accepted steps.

## Implementation

`pjepa.objectives.FreeEnergy` is a frozen dataclass holding the three coefficients. Its `__call__` evaluates 𝒥 on a graph + observation pair.

```python
from pjepa.objectives import FreeEnergy

J = FreeEnergy(beta_ib=0.01, lambda_mdl=0.001, gamma_forward=0.0001)
value = J(graph, observation)
```

## The IB Lagrangian

The Information Bottleneck Lagrangian (Tishby et al. 1999) is

$$\mathcal{L}_\text{IB}(q) \;=\; I(X; Z) \;-\; \beta \cdot I(Y; Z).$$

The variational upper bound (Alemi et al. 2017) used at training time is

$$\mathcal{L}_\text{VIB} \;\le\; \mathbb{E}_{x,y}\!\bigl[\mathbb{E}_{z \sim q(\cdot\mid x)}[-\log p(y \mid z)] + \beta \cdot D_\text{KL}(q(z\mid x)\,\|\,p(z))\bigr].$$

`pjepa.objectives.ib_lagrangian` computes the symbolic form; `pjepa.objectives.variational_ib_bound` computes the variational estimator.

## Description Length

`pjepa.objectives.description_length` returns an MDL estimate:

$$\mathrm{DL}(G) \;=\; \log(|V|+1) + \log(|E|+1) + \log(\mathrm{nnz}(F)+1).$$

A grammar-aware DL is planned for Phase 5+.

## Where to Look Next

* [Retrieval](retrieval.md) — uses utility functions related to 𝒥.
* [Verified Rewriting](verified-rewriting.md) — uses four-conditions acceptance derived from 𝒥.
* [Dynamics](dynamics.md) — analyses the contraction of $G_t$ under bounded perturbations.