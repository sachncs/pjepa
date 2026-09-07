# Configuration Schema

`pjepa` configurations are YAML files loaded via [`pjepa.config.load_config`](api.md). The schema below documents the keys recognised by each experiment runner.

## Common schema

```yaml
experiment:
  name: <string>           # Human-readable experiment name (e.g. "tu_proteins_baseline")
  dataset: <string>         # Dataset identifier (e.g. "PROTEINS", "ogbn-arxiv")
  seed_split: <int>         # Seed used for dataset splits
  seed_model: <int>         # Seed used for model initialisation

training:
  epochs: <int>            # Training epochs (default: 200)
  batch_size: <int>        # Batch size (default: 32)
  optimizer: <string>      # One of "adam", "adamw", "sgd"
  lr: <float>              # Learning rate (default: 5e-4)
  weight_decay: <float>    # Weight decay (default: 1e-5)
  scheduler: <string>      # One of "none", "cosine", "step"

model:
  encoder: <string>        # One of "euclidean_mpnn", "hyperbolic", "dual_geometric"
  hidden_dim: <int>        # Width of the encoder layers
  num_layers: <int>        # Number of message-passing layers
  output_dim: <int>        # Encoder output dimension
  dropout: <float>         # Dropout rate (0.0 to 0.5)

pjepa:
  B: <int>                 # Working-graph budget (Persistent-JEPA retrieval)
  beta_ib: <float>         # IB KL coefficient
  lambda_mdl: <float>      # MDL coefficient
  gamma_forward: <float>   # Forward-information bonus coefficient
  ema_momentum: <float>    # EMA momentum for the target encoder
  bisimulation_eps: <float> # Maximum allowed bisimulation distance
  max_cost: <float>         # Maximum allowed rewrite cost

optuna:
  n_trials: <int>          # Number of trials per dataset
  pruner: <string>         # One of "hyperband", "median", "none"
```

## TU experiment schema (`configs/tu.yaml`)

```yaml
experiment:
  name: tu_sota
  datasets: [PROTEINS, MUTAG, NCI1, IMDB-BINARY, REDDIT-BINARY, DD]
  seed_split: 0
  seed_model: 42

training:
  epochs: 200
  batch_size: 32
  optimizer: adamw
  lr: 5.0e-4
  weight_decay: 1.0e-5
  scheduler: cosine

model:
  encoder: dual_geometric
  hidden_dim: 128
  num_layers: 4
  output_dim: 128

pjepa:
  B: 64
  beta_ib: 1.0e-2
  lambda_mdl: 1.0e-3
  gamma_forward: 1.0e-4
  ema_momentum: 0.996
  bisimulation_eps: 1.0e-2
  max_cost: 1.0
```

## CL experiment schema (`configs/cl.yaml`)

```yaml
experiment:
  name: cl_sota
  datasets: [PROTEINS, MUTAG, NCI1]
  n_tasks: 5
  seed_split: 0
  seed_model: 42

training:
  epochs_per_task: 30
  optimizer: adamw
  lr: 1.0e-2
  weight_decay: 1.0e-4

model:
  encoder: dual_geometric
  hidden_dim: 64
  num_layers: 2
```

## OGB-arxiv schema (`configs/ogb.yaml`)

```yaml
experiment:
  name: ogb_arxiv
  dataset: ogbn-arxiv
  seed_split: 0
  seed_model: 42

training:
  epochs: 100
  batch_size: 1024
  optimizer: adamw
  lr: 1.0e-2
  weight_decay: 5.0e-4

model:
  encoder: gin
  hidden_dim: 256
  num_layers: 3
  dropout: 0.5
```

## Validation

Configuration is validated at load time via [`pjepa.config.ConfigSchema`](api.md). Missing required sections raise [`ConfigError`](api.md).