# Getting Started

## Installation

Persistent-JEPA requires Python 3.10, 3.11, or 3.12. We recommend 3.12.

### From source

```bash
git clone https://github.com/sachncs/persistent-jepa.git
cd persistent-jepa
make install
```

This will:

1. Create a Python 3.12 virtual environment at `.venv`.
2. Install the package in editable mode (`pip install -e .`).
3. Install all development dependencies (pytest, ruff, pytype, etc.).
4. Install OGB extras (`ogb` package).

### Manual installation

If you prefer not to use the Makefile:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,ogb]"
```

## Verify your environment

```bash
make doctor
```

This runs six capability probes and reports GREEN/YELLOW/RED for each:

```
Backend:    mps
Device:     Apple Silicon (MPS)
Python:     3.12.4
PyTorch:    2.13.0
Platform:   macOS-15.0-arm64-arm-64bit-Mach-O
CPU count:  12

Capability probes:
  [GREEN ] matmul
  [GREEN ] scatter_add
  [GREEN ] torch.compile
  [GREEN ] hyperbolic
  [GREEN ] pyg_scatter
  [GREEN ] cpu_fallback
```

## Run the cheap validation benchmarks

The paper makes two central claims that have fast, deterministic
validations. Both run in seconds:

```bash
# Validates Theorem 3: greedy retrieval achieves (1 - 1/e) ≈ 0.632 of optimal
make bench-retrieval

# Validates Proposition 7: hyperbolic per-edge distortion is Θ(log D / (D log b))
make bench-distortion
```

## Next steps

* For researchers: [Persistent Graph World Model](researcher/01_persistent_graph_world_model.md)
* For developers: [Quickstart for Developers](developer/01_quickstart.md)