# Security Audit Coverage

## Automated CVE Scanning

| Package Source | Tool | Coverage |
|---|---|---|
| Python (`requirements*.txt`) | `safety` + `pip-audit` | Full — PyPI advisory DB |
| PyPI packages in `pixi.lock` | `pip-audit` (pixi-audit job) | Full — PyPI advisory DB |
| conda-forge packages in `pixi.lock` | None (see gap below) | Manual review only |

## Known Gap: conda-forge / Mojo Runtime Packages

No public, machine-readable CVE feed covers conda-forge packages with the same
fidelity as PyPI. The Mojo runtime (`max`, `mojo`, `modular`) is distributed via
`conda.modular.com` and has no external CVE database.

**Mitigation (manual review checklist, run weekly with the audit job):**

- [ ] Check [Modular security advisories](https://www.modular.com/security) for
  Mojo/MAX runtime issues.
- [ ] Review the conda-forge GitHub repo for CVE-related issues on packages locked
  in `pixi.lock` that are not also available on PyPI.
- [ ] Verify `pixi.lock` has not pinned any conda package to a version with a known
  upstream CVE (check NVD at <https://nvd.nist.gov/> manually for high-profile
  packages such as `openssl`, `libxml2`, `zlib`).

## Updating This Document

When a new conda package class is added to `pixi.toml`, add a row to the table
above and update the manual checklist if needed.
