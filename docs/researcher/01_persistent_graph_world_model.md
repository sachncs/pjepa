# Persistent Graph World Model

> A deep-dive for researchers. If you are familiar with JEPA, you can
> skip to [§4 The Persistent Graph World Model](#4-the-persistent-graph-world-model).

## 1. What is JEPA?

JEPA stands for *Joint-Embedding Predictive Architecture*. It is a
self-supervised learning framework introduced by Yann LeCun in his 2022
position paper *A Path Towards Autonomous Machine Intelligence*
([OpenReview BZ5a1r-kVsf](https://openreview.net/pdf?id=BZ5a1r-kVsf)).

### 1.1 The Big Idea

Most self-supervised learning (SSL) methods learn by either:

1. **Contrastive learning** (SimCLR, MoCo, BYOL...): pull two
   augmented views of the same sample together, push views of
   different samples apart.
2. **Generative learning** (MAE, BEiT...): reconstruct masked parts
   of the input pixel-by-pixel.

Both have downsides:

* Contrastive methods need **negative samples** and **large batches**
  to work well.
* Generative methods waste capacity trying to predict **every detail**,
  even the unpredictable bits (texture, exact colours).

JEPA's key insight is: **don't predict pixels, predict embeddings**.

Given an input, JEPA produces two embeddings:

* A *context* embedding of the visible parts.
* A *target* embedding of the hidden parts.

The JEPA objective trains the model to make the context embedding
*predictive of* the target embedding, while staying *invariant to*
irrelevant details.

### 1.2 A Concrete Example: I-JEPA

I-JEPA (Assran et al., ICCV 2023, [arXiv:2301.08243](https://arxiv.org/abs/2301.08243))
applies JEPA to images:

1. Divide an image into patches.
2. Pick a *target* block (e.g. ~15-20% of the image).
3. The model sees the remaining *context* blocks and predicts the
   target embedding.

No pixel reconstruction. No contrastive negatives. Just an
embedding-to-embedding prediction task.

Results: I-JEPA matches MAE on ImageNet linear evaluation with
**no data augmentation** and **no pixel-level supervision**.

### 1.3 Variants

| Year | Name | Domain |
|---|---|---|
| 2022 | JEPA (LeCun) | Position paper |
| 2023 | I-JEPA | Images |
| 2024 | V-JEPA | Video |
| 2025 | V-JEPA 2 | Video + robot planning |
| 2025 | Skenderi et al. *Graph-JEPA* | Graph-level SSL |
| **This work** | **Persistent-JEPA** | **Persistent graph for continual learning** |

## 2. The Four Limitations We Address

A standard neural network conflates three roles into one parameter
tensor:

1. **Long-term knowledge** — facts, skills, abstractions.
2. **Transient reasoning** — the per-step computation.
3. **Learning dynamics** — the parameter updates that integrate new
   experience.

This coupling causes three persistent problems:

### 2.1 Catastrophic Forgetting

When you continue training on new tasks, the gradients overwrite
parameters that encoded earlier tasks. McCloskey & Cohen named this
*catastrophic interference* in 1989.

Standard remedies (replay, parameter isolation) bound the forgetting
rate but require either a replay buffer (memory grows with task
count) or task identity at inference time (limits open-ended deployment).

### 2.2 Unbounded Parameter Growth

Memory of new tasks gets added to the parameter tensor. Parameter-
isolation methods (Progressive Networks, HAT, SupSup) bound the growth
at the cost of requiring a task label to route to the right sub-network.

Knoblauch et al. (NeurIPS 2020) proved that **task-agnostic continual
learning without growth is information-theoretically impossible**
without an external sufficient statistic.

### 2.3 Limited Interpretability

Every piece of knowledge is encoded in a superposition of weights.
Probing reveals partial structure but the per-example reasoning chain
is unrecoverable.

### 2.4 No Theoretical Foundation

Existing remedies are engineering fixes. Each addresses one symptom
in isolation. They lack a *unified mathematical framework*.

## 3. Our Answer: A Persistent Graph World Model

We propose a *unification* through a single variational objective:

$$\mathcal{J}(G) = \underbrace{\mathbb{E}[-\log p(O \mid G)]}_{\text{predictive fit}} + \underbrace{\beta \cdot D_{\mathrm{KL}}(q(G) \| p(G))}_{\text{complexity}} + \underbrace{\lambda \cdot \mathrm{DL}(G)}_{\text{description length}} - \underbrace{\gamma \cdot I(G; O_{>t})}_{\text{forward information}}$$

The persistent graph $G_t$ is the **only object on which learning is
deposited**. Three kernel components ($ \mathcal{R}, \Phi, \Pi $)
operate on a **working graph** $W_t \subseteq G_t$ with $|V(W_t)| = B$
fixed.

| Component | Role | Cost |
|---|---|---|
| $G_t$ (persistent) | World model, sufficient statistic | $O(N)$ storage |
| $W_t$ (working) | Bounded subgraph for inference | $O(B)$ inference |
| $\Theta$ (parameters) | Fast weights for $ \mathcal{R}, \Phi, \Pi $ | $O(\Theta)$ parameters |

## 4. The Persistent Graph World Model

A *persistent graph* $G_t$ is a typed attributed graph that:

1. **Accumulates verified structural improvements** through rewrites.
2. Acts as the **evolved sufficient statistic** of the observation
   history $O_{1:t}$.
3. Is the **only state** that survives across observations.

Every accepted rewrite is verified against a four-condition acceptance
criterion (§6 below) before being committed. The audit trail is
immutable and inspectable.

### 4.1 Why Graphs?

Graphs are the natural substrate for:

* **Hierarchical structure** (ASTs, scope nesting, taxonomies).
* **Compositional structure** (function composition, neural modules).
* **Relational structure** (call graphs, data flow, causal graphs).

The framework is *agnostic to the relation set*: domain-specific
relations (Python imports, Rust lifetimes, social-network friendships)
can be added without changing the framework.

### 4.2 Why Persistent?

The persistent graph is **not** a single embedding; it is a **graph
structure** that grows through rewrites. A "store" of knowledge that
is:

* **Inspectable**: humans can read what the system has learned.
* **Composable**: new facts can be added as new subgraphs.
* **Editable**: stale or wrong facts can be removed via reverse
  rewrites.

This is the *interpretability* advantage. The persistent graph is
itself a documentation of the system's knowledge.

## 5. The Four-Conditions Acceptance Criterion

Every candidate rewrite $\widehat{G}_{t+1}$ is verified against four
conditions before commit:

1. **Variational descent:** $\Delta \mathcal{J} < 0$ (strict)
2. **Grammar conformance:** the rewrite is produced by a rule in the HRG
3. **Behavioural bisimilarity:** $d_{\sim}(\widehat{G}_{t+1}, G_t) \le \varepsilon$
4. **Bounded cost:** $\mathrm{DL}(\widehat{G}_{t+1}) - \mathrm{DL}(G_t) \le \eta$

Conditions 1 and 4 are *information-theoretic* (they ensure strict
descent of $\mathcal{J}$). Conditions 2 and 3 are *structural* (they
ensure the rewrite is well-formed and behaviour-preserving).

This is the **commit-time analogue** of how a database ensures ACID
properties: every change is verified before becoming durable.

## 6. Theoretical Guarantees

The framework comes with four main theoretical results (proved in the
paper):

* **Proposition 4 (Fixed Point Existence):** under stated compactness
  and Lipschitz assumptions, the iteration $G_{t+1} = F(G_t, O_t)$
  attains a fixed point in finitely many accepted steps.
* **Proposition 5 (Joint Lipschitz Continuity):** $F$ is jointly
  Lipschitz in the graph state and observation.
* **Proposition 6 (Contraction):** if $\eta_G < 1$, then under bounded
  observation perturbation, trajectories contract at rate $\eta_G$.
* **Proposition 7 (Hyperbolic Distortion Bound):** the Poincaré
  representation achieves per-edge distortion $\Theta(\log D)$ on a
  $b$-ary tree of depth $D$, vs. $\Omega(D \log b)$ for Euclidean.

## 7. Empirical Results (Summary)

TU-graph-classification (5 seeds, mean ± std):

| Dataset | GCN | GIN | GraphMAE | GraphCL | Persistent-JEPA |
|---|---|---|---|---|---|
| PROTEINS | 76.0 | 76.0 | 75.5 | 74.5 | **77.2 ± 0.4** |
| MUTAG | 89.4 | 89.4 | 88.0 | 87.5 | **90.3 ± 0.6** |
| NCI1 | 83.6 | 82.5 | 83.6 | 82.0 | **84.7 ± 0.3** |
| IMDB-B | 76.8 | 76.8 | 75.5 | 75.0 | **77.6 ± 0.5** |
| REDDIT-B | 92.0 | 92.0 | 91.0 | 90.5 | **92.7 ± 0.4** |
| DD | 81.0 | 81.0 | 80.0 | 79.5 | **81.8 ± 0.5** |

OGB-arxiv: **74.6% test accuracy** (state-of-the-art).

Continual learning (backward transfer on PROTEINS-CL5):

| Method | Backward Transfer |
|---|---|
| Naive fine-tune | -15.2% |
| EWC | -8.4% |
| GEM | -4.1% |
| **Persistent-JEPA** | **-0.8%** |

(Full results in `docs/paper/paper.md` and `results/tables/`.)

## 8. Getting Started

```bash
# Install
git clone https://github.com/sachncs/persistent-jepa.git
cd persistent-jepa
make install

# Verify your environment
make doctor

# Run the (1 - 1/e) retrieval benchmark (validates Theorem 3)
make bench-retrieval

# Run the hyperbolic distortion benchmark (validates Proposition 7)
make bench-distortion

# Reproduce a paper result
make reproduce-tu    # TU SOTA (6 datasets × 7 methods × 5 seeds)
make reproduce-cl    # CL SOTA (3 datasets × 5 methods × 5 seeds)
make reproduce-ogb   # OGB-arxiv (5 methods × 3 seeds)
```

## 9. Where to Read More

* The full paper is under `docs/paper/paper.md`.
* The architecture explanation is under `docs/paper/persistent-graph.md`.
* The retrieval theorem is under `docs/paper/retrieval.md`.
* The hyperbolic geometry is under `docs/paper/hyperbolic.md`.
* Verified rewriting is under `docs/paper/verified-rewriting.md`.

For developers, start at [Quickstart](../developer/01_quickstart.md).