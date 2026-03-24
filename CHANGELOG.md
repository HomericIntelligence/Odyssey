# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Enhanced `AnyTensor.__str__` with multi-dimensional and dtype-aware formatting
- Comprehensive tests for native `Tensor[dtype]` operations
- Native `Tensor[dtype]` elementwise, activation, matrix, reduction, convolution,
  shape, comparison, norm, and loss operations
- `hephaestus` package integration replacing local utility code

### Fixed

- `uint64` sign corruption and multi-dimensional truncation in tensor `__str__`
- Training loop crash and VGG16 end-to-end crash
- Four runtime bugs across the tensor and training subsystems
- Compilation errors across 17 test files after typed tensor migration
- Circular dependency between `shared.core` and `shared.tensor.typed`
- Overload ambiguity in typed tensor operations build
- Gitleaks CI action replaced with free CLI binary to fix license error

### Changed

- Moved `AnyTensor` from `shared/core/` to `shared/tensor/` to eliminate circular imports
- `AnyTensor` operators now delegate to typed implementations
- Extracted `shared/base/` package to break circular dependency

## [0.1.0] - 2025-11-07

Initial release of ML Odyssey -- a Mojo-based AI research platform for reproducing
classic research papers.

### Added

- **Tensor system**: `AnyTensor` runtime-typed tensor with full Array API coverage
  (setitem views, hash support, shape ops on non-contiguous inputs)
- **Parametric tensor**: `Tensor[dtype]` compile-time typed tensor with SIMD element
  access and `TensorLike` trait (`as_tensor[dtype]()` / `as_any()` zero-copy conversion)
- **Neural network layers**: Linear, Conv2D, BatchNorm2D, pooling, activations, and
  loss functions with forward and backward passes
- **Model implementations**: LeNet-5, AlexNet, VGG16, GoogLeNet with training and
  inference examples
- **Training infrastructure**: Multi-precision training (FP16, BF16, FP8), mixed-precision
  support, dataset loaders, and script runner
- **Autograd**: Tape-based automatic differentiation with composed operations and
  backward pass catalog
- **Profiling and logging**: File I/O for profiling report export, environment variable
  log level configuration
- **Agent system**: 29 agent configurations with 6-level hierarchy and 58 skills across
  11 categories (GitHub, Worktree, Mojo, CI/CD, Quality, and more)
- **CI/CD**: Comprehensive test workflow, build validation, pre-commit hooks
  (`mojo format`, `markdownlint-cli2`, trailing whitespace, YAML checks),
  and GHCR container publishing
- **Justfile build system**: Unified command runner for local development and CI consistency
- **Automation scripts**: `implement_issues.py` with dependency graph, rollback, and
  health checks; `fix-build-errors.py` with adaptive timeout and metrics
- **Container support**: Podman-based development environment with GHCR image publishing
- **Documentation**: ADR records (12 decisions), developer guides, Mojo anti-patterns
  catalog (64+ failure patterns), and testing strategy guide
- **Licensing**: BSD license

[Unreleased]: https://github.com/HomericIntelligence/ProjectOdyssey/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/HomericIntelligence/ProjectOdyssey/releases/tag/datasets-v2
