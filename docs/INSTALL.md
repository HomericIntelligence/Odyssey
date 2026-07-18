# External Consumer Install Guide

This guide is for developers who want to use ML Odyssey's `src/odyssey/` library
(tensor ops, autograd, layers, training infrastructure) in their own Mojo project.

## Prerequisites

| Requirement | Version | Notes |
| --- | --- | --- |
| Mojo | 1.0.0b2+ | Installed via `uv sync --locked` from the Modular PyPI index, or [mojolang.org](https://mojolang.org/) |
| glibc | ≥ 2.32 | Ubuntu 22.04+, Debian 12+, or use Podman (see below) |
| [uv](https://docs.astral.sh/uv/) | latest | Environment and package manager |

### GLIBC Compatibility

Mojo 1.0.0b2 requires glibc ≥ 2.32. Older distributions (Debian 10/11, Ubuntu 20.04)
will see `GLIBC_2.32 not found` errors. Compatible OS versions:

| OS | glibc | Status |
| --- | --- | --- |
| Ubuntu 22.04 (Jammy) | 2.35 | Compatible |
| Ubuntu 24.04 (Noble) | 2.39 | Compatible (CI environment) |
| Debian 12 (Bookworm) | 2.36 | Compatible |
| Ubuntu 20.04 / Debian 11 | 2.31 | **Incompatible** — use Podman |

If your host is incompatible, run everything inside the project's Podman container:

```bash
just podman-up
just shell   # opens a shell with Mojo 1.0.0b2 and glibc 2.39
```

See [`docs/dev/mojo-glibc-compatibility.md`](dev/mojo-glibc-compatibility.md) for details.

## Quick Start: Clone the Repo

The simplest way to consume `src/odyssey/` is to clone this repo and add it to your
Mojo package path:

```bash
git clone https://github.com/HomericIntelligence/Odyssey.git
cd Odyssey
uv sync --locked      # installs Mojo 1.0.0b2 and all Python tooling
just build            # compiles src/odyssey/ into .mojopkg artifacts
```

Then, in your project, add the `src/odyssey/` directory to `MOJO_PATH`:

```bash
export MOJO_PATH="/path/to/Odyssey/shared"
```

## Importing the Shared Library

Once `MOJO_PATH` includes the `src/odyssey/` directory you can import directly:

```mojo
from odyssey.core.any_tensor import AnyTensor, zeros
from odyssey.tensor.tensor import Tensor
from odyssey.autograd.variable import Variable
from odyssey.training.trainer import Trainer
```

The dual-type tensor system (`Tensor[dtype]` for compile-time dispatch, `AnyTensor`
for runtime-typed collections) is documented in
[`docs/adr/ADR-012-parametric-dtype-tensor-architecture.md`](adr/ADR-012-parametric-dtype-tensor-architecture.md).

## Building a Distributable Package

```bash
just package   # compiles src/odyssey/ into a .mojopkg archive
```

The output `.mojopkg` file can then be distributed and added to downstream
projects via `MOJO_PATH` or `--import-path`.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `GLIBC_2.32 not found` | Use Ubuntu 22.04+, Debian 12+, or `just shell` in Podman |
| `cannot find module 'shared'` | Verify `MOJO_PATH` includes `Odyssey/shared` |
| `mojo: command not found` | Run `uv sync --locked` first; Mojo is installed by uv |
| Build fails with import errors | Run `just build` before importing; compiled artifacts are required |
