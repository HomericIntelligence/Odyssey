# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Typed tensor system**: Native `Tensor[dtype]` with elementwise, activation, matrix,
  reduction, convolution, shape, comparison, norm, and loss operations
- Enhanced `AnyTensor.__str__` with multi-dimensional and dtype-aware formatting
- Parametric `load[dtype]`/`store[dtype]`/`data_ptr[dtype]` accessors on `AnyTensor`
- `hephaestus` package integration replacing local utility code

### Changed

- Moved `AnyTensor` from `shared/core/` to `shared/tensor/` to eliminate circular imports
- `AnyTensor` operators now delegate to typed implementations
- Extracted `shared/base/` package to break circular dependency

### Fixed

- `uint64` sign corruption and multi-dimensional truncation in tensor `__str__`
- Training loop crash and VGG16 end-to-end crash
- Four runtime bugs across the tensor and training subsystems
- Compilation errors across 17 test files after typed tensor migration
- Circular dependency between `shared.core` and `shared.tensor.typed`
- Overload ambiguity in typed tensor operations build
- Gitleaks CI action replaced with free CLI binary to fix license error
- Slice view bad-free: skip `pooled_free` for view tensors in `AnyTensor.__del__`
- Multi-dim slice fast-path ignoring step parameter
- VGG16 numerical overflow via bitcast migration to `load[dtype]`/`store[dtype]`

## [0.2.0-dev] - 2025-11-08 to 2026-03-25

Major development cycle: 1676 commits across tensor architecture, neural network layers,
training infrastructure, CI/CD hardening, and agent system evolution.

### Tensor Architecture and Core Operations

#### Added

- **Dual-type tensor system** (ADR-012): `Tensor[dtype]` for compile-time typed SIMD
  access, `AnyTensor` for runtime-typed collections and autograd tape
- **ExTensor**: Advanced shape manipulation (permute, broadcast, transpose views),
  slicing/indexing, comparison operations with NaN/Inf edge cases, statistical
  reductions (mean, std, var), utility methods (clone, item, diff, `__bool__`,
  `__hash__`, `__str__`/`__repr__`)
- **Lazy evaluation system**: Deferred tensor expression evaluation for optimized
  computation graphs
- **Strassen's algorithm**: Fast matrix multiplication for large matrices
- **SIMD-optimized fused batch normalization** for performance-critical paths
- **Memory pool**: Small tensor allocation pool to reduce malloc overhead
- **DType dispatch**: Ordinal lookup table replacing if/elif chains
- **Named constants**: Extracted magic numbers to named constants module
- **Contiguous fast paths**: Optimized arithmetic operations for contiguous tensors
- `transpose_view()` for zero-copy transposition
- `__bool__` for single-element truthiness on both `AnyTensor` and `ExTensor`
- `copy()` edge case support for non-contiguous, empty, and multi-dtype tensors
- BFloat16 (`BF16`) native type support via Mojo built-in types
- `bfloat16` support in `_set_float64` and `_get_float64` conversion paths

#### Changed

- Consolidated `bfloat16.mojo` into `types/bf16.mojo`, removed custom `BFloat16` alias
- Migrated custom dtypes to Mojo built-in types
- Standardized `dim` parameter to `axis` across all reduction operations
- Replaced `bitcast` pointer patterns with `load[dtype]`/`store[dtype]` accessors
- Rewrote `__getitem__(*slices)` to use element-wise copy semantics
- Consolidated `is_contiguous()` implementations

#### Fixed

- View destructor refcount leak and refcount underflow in `ExTensor.__del__`
- List corruption in `ExTensor` move constructor
- Reverse slicing with negative step in `__getitem__(Slice)`
- `remove_safely()` implemented using Python `os.remove()` interop
- Shallow copy replaced with deep copy in `check_gradients`
- `__setitem__` overload ambiguity resolved via `Int64`/`Float32` priority ordering
- Serialization: use `dtype_to_string()` instead of `String(dtype)`,
  sort glob results for deterministic `load_named_tensors`

### Neural Network Layers and Models

#### Added

- **SELU activation**: Scalar and vectorized implementations with SIMD conditionals
- **Conv2D no-bias gradients**: `Conv2dNoBiasGradient` and
  `DepthwiseConv2dNoBiasGradient` structs
- Multi-channel backward pass tests for Conv2D
- Gradient checks for `grad_gamma`/`grad_beta` in `batch_norm2d_backward`
- Numerical gradient test for `layer_norm_backward`
- Batch norm inference-mode gradient tests for Float64 dtype

#### Changed

- Conv2D backward pass: dtype-generic accumulation to prevent Float16 overflow
- Deprecated `LinearBackwardResult` and Conv backward result type aliases removed

#### Fixed

- `matmul_backward` gradient computation for 2D matrices
- Float16 convolution precision limitations documented and tolerance relaxed
- Compilation errors across AlexNet, VGG16, GoogLeNet, MobileNetV1, ResNet-18,
  and LeNet-5 example training/inference scripts

### Training Infrastructure

#### Added

- **TrainingLoop framework**: Migrated AlexNet, MobileNetV1, GoogLeNet to
  `TrainingLoop` pattern
- **CheckpointManager**: Model checkpointing with save/load support
- **Gradient clipping utilities**: Comprehensive gradient clipping by norm and value
- **Early stopping**: With `best_epoch` tracking and verbose logging
- **Optimizer state persistence**: `save_state`/`load_state` for full checkpoint support
- **Dataset loaders module** and **script runner** for training pipeline automation
- **ValidationLoop**: With `run_subset()` reset guard for zero batches
- **LossTracker** and **AccuracyMetric** as top-level shared exports
- **CSVMetricsLogger** exported from metrics package
- **Progress bar** for training visualization
- **Batch slicing** in training loop for efficient data handling
- BF16 enabled in `recommend_precision_dtype` for large models
- `DataLoader.next()` support for N-D tensors
- `run_epoch()` migrated from `PythonObject` to native `DataLoader`
- In-place gradient operations for memory-efficient training
- Base optimizer trait extracted for code reuse

#### Changed

- Standardized `TrainingArgs` usage across all training scripts
- Training loop callbacks documented with import limitation notes
- Mixed precision FP16-FP32 SIMD vectorization documented with Mojo v0.26.1
  limitations (ADR-010)

#### Fixed

- Gradient accumulation bug in `clip_gradients_by_global_norm`
- `Optional[Int]` replaced with `Int` sentinel in samplers (Mojo limitation)
- Compilation errors in confusion matrix, metrics, and mixed precision modules

### Autograd and Gradient System

#### Added

- **Gradient system integration**: Connected gradient computation with training
  infrastructure
- **Composed operations** backward pass with test coverage
- `sin_backward` and `cos_backward` gradient functions
- Gradient error reporting completion
- Property-based testing framework for gradient verification
- Standard benchmark datasets for gradient validation
- Gradient comparison in layer testers with configurable epsilon/tolerance

#### Changed

- Extracted backward operations from tape to separate module
- Deprecated `@value` decorator replaced; gradients API updated

#### Fixed

- Variable constructor calls and gradient registry usage
- Epsilon defaults corrected for Float32 gradient checking

### CI/CD and Infrastructure

#### Added

- **ADR-009 compliance**: Test file splitting to work around Mojo 0.26.1 heap
  corruption (391 part files merged into 132 unified tests)
- **Composite actions**: Extracted `setup-pixi` composite action for CI reuse
- **Test retry logic** across all test/build workflows
- **Security scanning**: Bandit replacing pygrep `shell=True` hook; Trivy SARIF upload
- **Docker build and publish workflow** with GHCR image publishing
- **Automated dependency auditing** workflow
- **Test count badge** automation to avoid drift
- **Code coverage reporting** infrastructure
- **Performance regression testing** infrastructure
- **Jupyter Notebook support** for ML experimentation
- **Mypy pre-commit hook** for Python type checking
- GLIBC-compatible `mojo-format` hook wrapper (`scripts/mojo-format-compat.sh`)
- Cargo-free enforcement hook for Docker files
- Audit-shared-links job in comprehensive tests

#### Changed

- Consolidated CI from 31 test groups to 16
- Consolidated CI workflows and extracted composite actions
- Docker base image bumped from Ubuntu 22.04 to 24.04
- Claude Code CLI version pinned in Docker for reproducible builds
- Pixi version pinned in Dockerfile
- Replaced deprecated `semgrep-action` with CLI for SAST scanning
- `download-artifact` upgraded from v7 to v4
- Mixed precision tests moved to per-PR CI (ADR-009 compliant)
- Pre-commit `mirrors-mypy` rev upgraded from v1.8.0 to v1.19.1

#### Fixed

- Persistent CI failures from `alias` to `comptime` migration (Mojo 0.26.1)
- Gitleaks allowlist and flaky Core Loss test exclusion
- Docker SBOM generation with lowercase image name
- Stale artifacts in Test Report workflow
- Link-check root-relative link exclusions
- ARM64 Docker builds removed (unsupported)

### Agent System and Skills

#### Added

- **58 skills** across 11 categories (GitHub, Worktree, Mojo, CI/CD, Quality,
  Testing, Documentation, Review, Phase Workflow, Agent System)
- Blog-writer and PR-cleanup converted from agents to skills
- Skills: `fixme-todo-cleanup`, `batch-pr-rebase`, `dtype-native-migration`,
  `investigate-mojo-heap-corruption`, `fix-docker-image-org-mismatch`,
  `verify-issue-before-work`
- Agent-scoped hooks for safety enforcement
- Claude Code Safety Net plugin
- SessionEnd hook for retrospective prompts
- Skill migration script for porting to ProjectMnemosyne

#### Changed

- Consolidated implementation engineer tiers from 4 to 2
- Consolidated documentation engineer tiers from 3 to 2
- Consolidated 13 review specialists into 5
- Consolidated junior-implementation-engineer into implementation-engineer
- Retrospective skills migrated to ProjectMnemosyne

### Automation Scripts

#### Added

- `implement_issues.py`: Curses-based UI with scrolling logs, work queue buffer,
  live worker status, Claude API usage limit detection, dependency graph,
  rollback, and health checks
- `fix-build-errors.py`: Adaptive timeout, per-file PRs, metrics tracking
- `analyze_issues.py`: Dependency analysis and epic creation
- `rebase-all-branches.py`: With worktree support and conflict reporting
- PNG/JPEG to IDX conversion script for inference
- Git bisect scripts for heap corruption investigation
- Code generation tools for boilerplate Mojo code
- Release automation scripts

#### Fixed

- Deadlock, race conditions, and data loss bugs in `implement_issues.py`
- Critical threading fixes (Phase 1 and Phase 2)
- State persistence race conditions
- Per-thread logging and graceful shutdown
- Slot management and UI fixes

### Documentation

#### Added

- **13 ADR records** (ADR-001 through ADR-013) including:
  - ADR-009: Heap corruption workaround for Mojo 0.26.1 test splitting
  - ADR-010: FP16 SIMD Mojo v0.26.1 limitation
  - ADR-012: Parametric dtype tensor architecture
  - ADR-013: Slice view destructor fix
- **Development blog**: 62 daily entries documenting development progress
- **Getting started**: Real installation and quickstart guides
- **Migration guide**: PyTorch to ML Odyssey
- **API reference**: Comprehensive API documentation
- **Performance benchmarking guide**
- **Glossary of terms**
- Mojo anti-patterns catalog expanded to 64+ failure patterns
- Testing strategy guide with Float16 convolution limitations
- Contributing guide with `mojo-format` GLIBC incompatibility notes

#### Changed

- README rewritten with accurate project description and CI badges
- CLAUDE.md trimmed to reduce token consumption
- Agent hierarchy documentation updated after consolidation
- 17 empty placeholder documentation stubs deleted
- 19 stale one-time fix/migration scripts removed

### Security

- BSD license adopted (replacing MIT)
- Bandit security scanner integrated into pre-commit
- `shell=True` usage eliminated from hooks
- Defensive memory safety checks added to serialization
- Bandit B310/B202 skips moved to `.bandit` config file
- GitHub URL migration from `mvillmow` to `HomericIntelligence`

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
[0.2.0-dev]: https://github.com/HomericIntelligence/ProjectOdyssey/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/HomericIntelligence/ProjectOdyssey/releases/tag/datasets-v2
