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
- **No `shell=True`** subprocess calls.
- **All file paths validated** against path traversal.
- **All YAML configs** loaded via a Pydantic-compatible schema (no
  arbitrary constructor execution).
- **Dependencies pinned** to exact versions in `pyproject.toml`.
- **`pip-audit`** runs in CI on every push to `main`; HIGH/CRITICAL
  vulnerabilities block the merge.

## Out of Scope

The following are explicitly out of scope for `pjepa`'s security model:

- The training data itself — `pjepa` is a library, not a service.
- The model weights produced by training — users are responsible
  for ensuring their model files are trustworthy.
- The deployment environment — users are responsible for their own
  infrastructure security.

## Acknowledgements

We thank the open-source community for responsible disclosure practices.