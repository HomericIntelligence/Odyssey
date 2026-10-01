# Quick Start

## Prerequisites

ML Odyssey requires Python 3.13 and the Mojo toolchain. Container builds need
Podman; see [installation.md](installation.md) for the full setup.

## Start the Development Container

```bash
git clone https://github.com/HomericIntelligence/Odyssey.git
cd Odyssey
just podman-up
```

`just podman-status` reports whether the container is up, and
`just podman-logs` tails its output.

## Run the Tests

```bash
just test
```

This runs both suites: `just test-mojo` for the Mojo test suite under `tests/`
and `just test-python` for the repository's Python checks.

## Run an Example

Open a shell inside the container and invoke the Mojo runner directly:

```bash
just podman-run-shell
```

Then, inside that shell:

```bash
mojo run examples/lenet_emnist/train.mojo
```

`just podman-run-tests` does the same thing non-interactively for the Mojo
test suite.

## Where Things Live

Model implementations live under `examples/`, one directory per paper, with
some standalone examples as loose `.mojo` files in the same directory.

## Next Steps

- [installation.md](installation.md) for toolchain and container details
- [repository-structure.md](repository-structure.md) for a tour of the tree
- [first_model.md](first_model.md) for a walkthrough of a model implementation
