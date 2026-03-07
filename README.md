# ML Odyssey

A Mojo-based platform for reproducing classic AI/ML research papers with production-quality
implementations. ML Odyssey provides a shared library of SIMD-optimized tensor operations,
an autograd engine, and a full training infrastructure — all implemented in Mojo for
maximum performance and type safety.

[![Mojo](https://img.shields.io/badge/Mojo-0.26+-orange.svg)](https://www.modular.com/mojo)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-223%2B-brightgreen.svg)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-pending-lightgrey.svg)](#coverage-status)

## What This Is

ML Odyssey is a research platform with two goals:

1. **Reproduce landmark neural network papers** with verified, high-performance Mojo implementations
2. **Provide a reusable shared library** of ML components that paper implementations build on

The project currently has ~198K lines of Mojo code, 7 fully-implemented neural network
architectures, and 223+ tests across layerwise unit tests and end-to-end integration tests.

## Implemented Architectures

| Architecture | Paper | Status |
|---|---|---|
| LeNet-5 | LeCun et al., 1998 | Implemented |
| AlexNet | Krizhevsky et al., 2012 | Implemented |
| VGG-16 | Simonyan & Zisserman, 2014 | Implemented |
| ResNet-18 | He et al., 2015 | Implemented |
| MobileNetV1 | Howard et al., 2017 | Implemented |
| GoogLeNet | Szegedy et al., 2014 | Implemented |

Each architecture has layerwise unit tests (runs on every PR) and end-to-end integration
tests (runs weekly with real datasets).

## Shared Library

The `shared/` directory contains the ML components used by all paper implementations:

### `shared/core/` - Tensor Operations and Layers

- SIMD-optimized tensor type (`ExTensor`) with compile-time dtype dispatch
- Convolution, linear, pooling, activation, normalization layers
- Matrix operations including Strassen multiplication
- Broadcasting, reduction, elementwise ops
- Dropout, batch normalization, attention

### `shared/autograd/` - Automatic Differentiation

- Tape-based reverse-mode autograd engine
- `Variable` type with gradient tracking
- Backward ops for all core operations
- Gradient utilities and type definitions

### `shared/training/` - Training Infrastructure

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
- [Quickstart](docs/getting-started/quickstart.md)
- [Repository Structure](docs/getting-started/repository-structure.md)
- [Shared Library README](shared/README.md)
- [Contributing Guide](CONTRIBUTING.md)
- [Architecture Decision Records](docs/adr/)

## Project Structure

```text
ProjectOdyssey/
├── shared/                  # Reusable ML library
│   ├── core/                # Tensor ops, layers, SIMD kernels
│   ├── autograd/            # Tape-based reverse-mode autograd
│   ├── training/            # Trainers, optimizers, schedulers
│   ├── data/                # Dataset loaders
│   └── testing/             # Shared test utilities
├── tests/
│   ├── models/              # Per-architecture test suites
│   └── shared/              # Shared library tests
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
- 247+ test files tracked with 500+ test functions
- Manual code review via PR checklist for test coverage verification
- 70%+ threshold enforced for Python automation scripts

### When Mojo Coverage Available

```bash
mojo test --coverage tests/
mojo coverage report --format=lcov > coverage.lcov
```

See [ADR-008](docs/adr/ADR-008-coverage-tool-blocker.md) for complete explanation.

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.
