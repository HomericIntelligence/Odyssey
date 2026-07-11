# ML Odyssey

A Mojo-based platform for reproducing classic AI/ML research papers with production-quality
implementations. ML Odyssey provides a shared library of SIMD-optimized tensor operations,
an autograd engine, and a full training infrastructure — all implemented in Mojo for
maximum performance and type safety.

[![Mojo](https://img.shields.io/badge/Mojo-1.0.0b2-orange.svg)](https://mojolang.org/)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-298+-brightgreen.svg)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-pending-lightgrey.svg)](#coverage-status)
[![CI](https://github.com/HomericIntelligence/Odyssey/actions/workflows/comprehensive-tests.yml/badge.svg)](https://github.com/HomericIntelligence/Odyssey/actions/workflows/comprehensive-tests.yml)
[![Build](https://github.com/HomericIntelligence/Odyssey/actions/workflows/build-validation.yml/badge.svg)](https://github.com/HomericIntelligence/Odyssey/actions/workflows/build-validation.yml)
[![ASan Tests](https://github.com/HomericIntelligence/Odyssey/actions/workflows/asan-tests.yml/badge.svg)](https://github.com/HomericIntelligence/Odyssey/actions/workflows/asan-tests.yml)
[![Benchmark](https://github.com/HomericIntelligence/Odyssey/actions/workflows/benchmark.yml/badge.svg)](https://github.com/HomericIntelligence/Odyssey/actions/workflows/benchmark.yml)
[![Pre-commit](https://github.com/HomericIntelligence/Odyssey/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/HomericIntelligence/Odyssey/actions/workflows/pre-commit.yml)
[![Security](https://github.com/HomericIntelligence/Odyssey/actions/workflows/security.yml/badge.svg)](https://github.com/HomericIntelligence/Odyssey/actions/workflows/security.yml)
[![Release](https://github.com/HomericIntelligence/Odyssey/actions/workflows/release.yml/badge.svg)](https://github.com/HomericIntelligence/Odyssey/actions/workflows/release.yml)
[![Container Publish](https://github.com/HomericIntelligence/Odyssey/actions/workflows/container-publish.yml/badge.svg)](https://github.com/HomericIntelligence/Odyssey/actions/workflows/container-publish.yml)
[![Docs](https://github.com/HomericIntelligence/Odyssey/actions/workflows/docs.yml/badge.svg)](https://github.com/HomericIntelligence/Odyssey/actions/workflows/docs.yml)
[![Validate Configs](https://github.com/HomericIntelligence/Odyssey/actions/workflows/validate-configs.yml/badge.svg)](https://github.com/HomericIntelligence/Odyssey/actions/workflows/validate-configs.yml)

## What This Is

ML Odyssey is a **standalone Mojo-based ML framework** for reproducing classic AI/ML
research papers with production-quality implementations. It has two goals:

1. **Reproduce landmark neural network papers** with verified, high-performance Mojo implementations
2. **Provide a reusable shared library** of ML components that paper implementations build on

The project currently has ~198K lines of Mojo code, 7 fully-implemented neural network
architectures, and 298+ tests across layerwise unit tests and end-to-end integration tests.

> **Note on project identity:** The GitHub repo description says "Training framework written
> in Mojo." This repo is sometimes described elsewhere as an "experimental agent research
> sandbox" -- that description is **incorrect**. ML Odyssey is an ML training framework, not
> an agent platform. It has no integration with ai-maestro, NATS, or any distributed agent
> mesh. The "agent system" referenced in this repo refers to
> [Claude Code](https://claude.ai/code) automation for development workflow (code generation,
> PR creation, CI management), not a runtime agent mesh.

## Part of HomericIntelligence

Odyssey is one of several repositories in the
[HomericIntelligence](https://github.com/HomericIntelligence) organization. Here is how the
repos relate:

| Repository | Role |
| --- | --- |
| **Odyssey** (this repo) | ML training framework in Mojo -- neural nets, autograd, shared lib |
| [Odysseus][odysseus] | Ecosystem meta-repo and architecture docs |
| [AchaeanFleet][achaeanfleet] | Container images for the agent mesh -- Dockerfiles, Compose, CI |
| [Myrmidons][myrmidons] | GitOps agent provisioning -- agent definitions as code |
| [ProjectHephaestus][hephaestus] | Shared utilities and tools used across the ecosystem |
| [ProjectMnemosyne][mnemosyne] | Skills marketplace -- collective memory of team learnings |
| [ProjectScylla][scylla] | Testing and optimization framework for agentic workflows |
| [ProjectKeystone][keystone] | Foundation project |
| [ProjectArgus][argus] | Ecosystem project |
| [ProjectHermes][hermes] | Ecosystem project |
| [ProjectProteus][proteus] | Ecosystem project |
| [ProjectTelemachy][telemachy] | Ecosystem project |

[odysseus]: https://github.com/HomericIntelligence/Odysseus
[achaeanfleet]: https://github.com/HomericIntelligence/AchaeanFleet
[myrmidons]: https://github.com/HomericIntelligence/Myrmidons
[hephaestus]: https://github.com/HomericIntelligence/ProjectHephaestus
[mnemosyne]: https://github.com/HomericIntelligence/ProjectMnemosyne
[scylla]: https://github.com/HomericIntelligence/ProjectScylla
[keystone]: https://github.com/HomericIntelligence/ProjectKeystone
[argus]: https://github.com/HomericIntelligence/ProjectArgus
[hermes]: https://github.com/HomericIntelligence/ProjectHermes
[proteus]: https://github.com/HomericIntelligence/ProjectProteus
[telemachy]: https://github.com/HomericIntelligence/ProjectTelemachy

### What Odyssey is NOT

To avoid confusion with other ecosystem repos:

- **Not a distributed agent mesh.** AchaeanFleet and Myrmidons handle agent orchestration.
  Odyssey has zero integration with ai-maestro, NATS, or any agent registration/task
  queue system.
- **Not an agent research sandbox.** It is a straightforward ML training framework. The only
  "agents" here are Claude Code development automation (see `.claude/agents/`), which manage
  code generation and CI -- they do not run as distributed services.
- **No REST API.** There is no REST client, no agent registration endpoint, and no promotion
  path to AchaeanFleet. Implementations live entirely in this repo as Mojo libraries and
  executables.

## Implemented Architectures

| Architecture | Paper | Status |
| --- | --- | --- |
| LeNet-5 | LeCun et al., 1998 | Implemented |
| AlexNet | Krizhevsky et al., 2012 | Implemented |
| VGG-16 | Simonyan & Zisserman, 2014 | Implemented |
| ResNet-18 | He et al., 2015 | Implemented |
| MobileNetV1 | Howard et al., 2017 | Implemented |
| GoogLeNet | Szegedy et al., 2014 | Implemented |

Each architecture has layerwise unit tests (runs on every PR) and end-to-end integration
tests (runs weekly with real datasets).

## Shared Library

The `src/projectodyssey/` directory contains the ML components used by all paper implementations:

### `src/projectodyssey/core/` - Tensor Operations and Layers

- SIMD-optimized tensor type (`AnyTensor`) with compile-time dtype dispatch
- Convolution, linear, pooling, activation, normalization layers
- Matrix operations including Strassen multiplication
- Broadcasting, reduction, elementwise ops
- Dropout, batch normalization, attention

### `src/projectodyssey/autograd/` - Automatic Differentiation

- Tape-based reverse-mode autograd engine
- `Variable` type with gradient tracking
- Backward ops for all core operations
- Gradient utilities and type definitions

### `src/projectodyssey/training/` - Training Infrastructure

- `Trainer` with configurable training loops
- Optimizers: SGD, Adam, AdamW, RMSprop, LARS
- Learning rate schedulers
- Gradient clipping
- Mixed precision training
- Model checkpointing and callbacks
- Evaluation and metrics

## Getting Started

### Prerequisites

- [Pixi](https://pixi.sh/) for environment management
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/homericintelligence/projectodyssey.git
cd projectodyssey

# Install all dependencies (Mojo, Python tools, etc.)
pixi install
```

### Run Tests

```bash
# Run all Mojo tests
just test-mojo

# Run layerwise tests for a specific model
pixi run mojo test tests/models/test_lenet5_layers.mojo

# Run all tests for a model
pixi run mojo test tests/models/test_lenet5_layers.mojo tests/models/test_lenet5_e2e.mojo
```

### Build the Shared Library

```bash
# Build project in debug mode
just build

# Build as distributable package
just package
```

### Quick Reference

```bash
# Show all available commands
just --list

# Format all code
just format

# Run pre-commit hooks on all files
just pre-commit-all

# Full validation (build + test)
just validate
```

## Documentation

- [Installation Guide](docs/getting-started/installation.md)
- [External Consumer Install Guide](docs/INSTALL.md)
- [Roadmap](ROADMAP.md)
- [Quickstart](docs/getting-started/quickstart.md)
- [Your First Model](docs/getting-started/first_model.md)
- [Repository Structure](docs/getting-started/repository-structure.md)
- [Shared Library README](src/projectodyssey/README.md)
- [Contributing Guide](CONTRIBUTING.md)
- [Architecture Decision Records](docs/adr/)
- [Privacy & Data-Handling Policy](docs/PRIVACY.md)

## Project Structure

```text
Odyssey/
├── src/projectodyssey/                  # Reusable ML library
│   ├── core/                # Tensor ops, layers, SIMD kernels
│   ├── autograd/            # Tape-based reverse-mode autograd
│   ├── training/            # Trainers, optimizers, schedulers
│   ├── data/                # Dataset loaders
│   └── testing/             # Shared test utilities
├── tests/
│   ├── models/              # Per-architecture test suites
│   └── src/projectodyssey/              # Shared library tests
├── docs/
│   ├── adr/                 # Architecture Decision Records
│   ├── getting-started/     # Setup and quickstart guides
│   └── dev/                 # Developer documentation
├── benchmarks/              # Performance benchmarks
├── scripts/                 # Python automation scripts
└── justfile                 # Build system recipes
```

## Testing Strategy

Tests are organized in two tiers:

- **Tier 1 (Layerwise Unit Tests)**: Run on every PR. Fast, deterministic tests using
  FP-representable values. Each layer's forward and backward pass is validated independently,
  including gradient checking against numerical finite differences.

- **Tier 2 (End-to-End Tests)**: Run weekly. Full model training on EMNIST and CIFAR-10,
  validating convergence over 5 epochs.

See [ADR-004](docs/adr/ADR-004-testing-strategy.md) for the complete testing strategy rationale.

## Coverage Status

Full code coverage metrics are blocked by [Mojo coverage tooling availability](docs/adr/ADR-008-coverage-tool-blocker.md).

### Current Workarounds

- All `test_*.mojo` files verified in CI via test discovery validation
  (`scripts/validate_test_coverage.py`)
- Source-to-test mapping: every `src/projectodyssey/**/*.mojo` is checked for
  a corresponding `test_*.mojo` file (`scripts/check_source_coverage.py`,
  warn-only as of initial rollout). Run locally:
  `python scripts/check_source_coverage.py`
- Test and source file counts (regenerate via the commands shown):
  `find tests -name 'test_*.mojo' | wc -l` and
  `find src/projectodyssey -name '*.mojo' ! -name '__init__.mojo' | wc -l`
- Manual code review via PR checklist for test coverage verification
- 70%+ threshold enforced for Python automation scripts via pytest-cov
- ADR-008 review cadence enforced quarterly via
  `scripts/check_adr_review_dates.py` in scheduled CI (see
  `.github/workflows/mojo-version-check.yml`)

> **Note on Mojo coverage**: Mojo 1.0 still has no coverage instrumentation (`mojo test --coverage`
> does not exist). The targets in `coverage.toml` are aspirational, not gated in CI. This is a
> known gap; enforcement will be added once Mojo coverage tooling matures.
>
> **Note on gradient coverage**: The gradient-coverage metric reported in CI is a proxy — it counts
> test files against backward-pass functions. It is **not** line-of-code coverage and cannot detect
> which branches within a backward pass are actually exercised.

### When Mojo Coverage Available

```bash
mojo test --coverage tests/
mojo coverage report --format=lcov > coverage.lcov
```

See [ADR-008](docs/adr/ADR-008-coverage-tool-blocker.md) for complete explanation.

## Benchmarks

Performance benchmarks live in `benchmarks/`. They are run as informational snapshots and are
**not a CI pass/fail gate** — a slower result does not block a PR from merging. Use benchmark
output to guide optimization work, not as a correctness signal.

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.
