# Security Policy

## Supported Versions

| Version | Supported |
|---|---|
| 1.0.x | ✅ Active development |
| 0.x.x | ⚠️ Best-effort; please upgrade to 1.0 |

## Reporting a Vulnerability

If you discover a security vulnerability in `pjepa`, please report it privately:

- **Email**: sachncs@gmail.com (please prefix subject with `[pjepa-security]`)
- **GitHub Security Advisories**: [Report a vulnerability](../../security/advisories/new)

We aim to acknowledge security reports within 3 business days and provide
a fix or mitigation within 30 days for critical vulnerabilities.

## Security Practices

`pjepa` follows these practices by default:

- **No `eval` or `exec`** anywhere in the library.
- **No `pickle.load`** on untrusted sources; checkpoints are loaded via
  `torch.load(..., weights_only=True)` which restricts deserialisation.
  The OGB-Arxiv loader temporarily patches `torch.load` to default
  `weights_only=False` for the duration of one call inside a
  `try`/`finally` so the process-wide default is restored even on
  exception; this is a deliberate compatibility workaround for
  OGB 1.3 with PyTorch 2.6+, not a permanent demotion of the
  default.
- **No `shell=True`** subprocess calls.
- **All file paths validated** against path traversal.
- **All YAML configs** are loaded with `yaml.safe_load`, which
  prevents arbitrary code execution via Python-object YAML tags.
  The loader accepts arbitrary extra keys (no schema validator is
  bundled); callers should validate required sections via
  `pjepa.config.load_config(path, required_sections=...)`.
- **Dependencies** are declared with lower-bound version specifiers
  (e.g. `torch>=2.13.0`) in `pyproject.toml`. No resolved lockfile
  is committed; users wanting reproducible installs should run
  `pip-compile` locally.
- **`pip-audit`** can be run locally via `make audit`; CI integration
  is planned but not yet wired into `.github/workflows/ci.yml`.

## Out of Scope

The following are explicitly out of scope for `pjepa`'s security model:

- The training data itself — `pjepa` is a library, not a service.
- The model weights produced by training — users are responsible
  for ensuring their model files are trustworthy.
- The deployment environment — users are responsible for their own
  infrastructure security.

## Acknowledgements

We thank the open-source community for responsible disclosure practices.