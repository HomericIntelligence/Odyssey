# Quick Start

## Prerequisites

Mojo Odyssey requires Python 3.13 and the Mojo toolchain. Container builds
need Podman; see [installation.md](installation.md) for the full setup.

## Install

```bash
git clone https://github.com/HomericIntelligence/Odyssey.git
cd Odyssey
just podman-up
```

## Run the Tests

```bash
just test
```

This runs both suites: `just test-mojo` for the Mojo sources under
`src/odyssey` and `just test-python` for the repository's own Python checks.

## Run an Example

```bash
just podman-run mojo run examples/lenet_emnist/train.mojo
```

Model implementations live under `examples/`, one directory per paper.

## Next Steps

- [installation.md](installation.md) for toolchain and container details
- [repository-structure.md](repository-structure.md) for a tour of the tree
- [first_model.md](first_model.md) for a walkthrough of a model implementation
