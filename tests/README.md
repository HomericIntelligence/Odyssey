# Tests Directory

## Overview

This directory contains **122 test files** with comprehensive coverage across the ML Odyssey project. Tests are
organized by component type and test classification (unit, integration, benchmarks).

## Directory Structure

```text
tests/
├── shared/                          # Shared library tests (primary test suite)
│   ├── core/                        # Core tensor & layer operations
│   │   ├── test_*.mojo              # Individual component tests
│   │   └── legacy/                  # Legacy operation tests
│   ├── data/                        # Data loading & transformation tests
│   │   ├── datasets/                # Dataset implementation tests
│   │   ├── loaders/                 # Data loader tests
│   │   ├── samplers/                # Sampler tests
│   │   └── transforms/              # Transformation pipeline tests
│   ├── training/                    # Training loop & optimization tests
│   │   ├── test_optimizers.mojo     # SGD, Adam, RMSprop, etc.
│   │   ├── test_*_scheduler.mojo    # Learning rate schedulers
│   │   ├── test_*_loop.mojo         # Training/validation loops
│   │   └── test_callbacks.mojo      # Callback system tests
│   ├── integration/                 # End-to-end integration tests
│   │   ├── test_end_to_end.mojo
│   │   ├── test_training_workflow.mojo
│   │   └── test_data_pipeline.mojo
│   ├── utils/                       # Utility function tests
│   ├── benchmarks/                  # Performance benchmarks
│   ├── fixtures/                    # Test fixtures and mocks
│   └── test_imports.mojo            # Module import tests
├── configs/                         # Configuration system tests
├── tooling/benchmarks/              # Performance regression detection
├── training/                        # Training infrastructure tests
├── unit/                            # Isolated component tests
├── debug/                           # Debug & analysis tests
├── integration/                     # Top-level integration tests
├── agents/                          # Agent configuration tests (Python)
├── foundation/                      # Foundation/structure tests (Python)
├── conftest.mojo                    # Global test configuration
└── helpers/                         # Test utilities & helpers
    ├── utils.mojo
    ├── assertions.mojo
    ├── fixtures.mojo
    └── gradient_checking.mojo
```text

## Running Tests Locally

### Mojo Tests

Mojo tests are executed individually using `pixi run mojo <file>`:

```bash
# Run a single test file
pixi run mojo tests/shared/core/test_tensors.mojo

# Run test in a subdirectory
pixi run mojo tests/shared/training/test_optimizers.mojo

# Run multiple test files (sequential)
pixi run mojo tests/shared/core/test_arithmetic_part1.mojo
pixi run mojo tests/shared/core/test_layers.mojo
```text

### Python Tests

Python tests use pytest:

```bash
# Run all Python tests
pixi run pytest tests/

# Run specific test file
pixi run pytest tests/agents/test_integration.py

# Run with verbose output
pixi run pytest tests/ -v

# Run with coverage report
pixi run pytest tests/ --cov=. --cov-report=html
```text

### Running Test Groups

The test suite is organized into 17 logical groups (see comprehensive-tests.yml). To run tests locally by group:

```bash
# Core tensor operations
cd tests/shared/core && pixi run mojo test_tensors.mojo

# Training optimizers
cd tests/shared/training && pixi run mojo test_optimizers.mojo

# Data transforms
cd tests/shared/data && pixi run mojo test_transforms.mojo
```text

## Test Organization & Classification

### Test Types

**Unit Tests** (`tests/unit/`, `tests/shared/core/test_*.mojo`)
- Test individual functions and classes in isolation
- Fast execution (< 1 second each)
- No external dependencies
- Test a single responsibility

**Integration Tests** (`tests/shared/integration/`, `tests/integration/`)
- Test interactions between components
- Validate data pipelines (DataLoader -> Transform -> Model)
- Test training workflows (Optimizer -> LossFunction -> BackwardPass)
- Test model inference end-to-end

**Benchmarks** (`tests/shared/benchmarks/bench_*.mojo`)
- Performance benchmarks for critical operations
- Measure operation latency and throughput
- Detect performance regressions

**Configuration Tests** (`tests/configs/test_*.mojo`)
- Validate configuration loading and merging
- Test environment variable handling
- Verify schema validation

## CI/CD Test Workflows

The project runs tests automatically in GitHub Actions across 4 dedicated workflows:

### 1. `unit-tests.yml` - Fast Unit Tests (5 min target)

**Triggers**: All PRs and pushes to main

**Tests**:
- Mojo unit tests in `tests/unit/`
- Python tests in `tests/unit/` and foundation tests
- Quick sanity checks

**Coverage**: Python unit tests with `pytest --cov`

```bash
# Equivalent local run
pixi run pytest tests/unit/ --cov
```text

### 2. `test-gradients.yml` - Gradient Checking (2 min target)

**Triggers**: Changes to backward pass implementations

**Tests**:
- Gradient correctness checking via numerical differentiation
- All backward operations (`*_backward.mojo`)
- Activation function gradients

**File**: `tests/shared/core/test_gradient_checking.mojo`

```bash
# Local equivalent
pixi run mojo tests/shared/core/test_gradient_checking.mojo
```text

### 3. `integration-tests.yml` - Integration Tests (8 min target)

**Triggers**: All PRs (skips drafts) and pushes to main

**Test Suites** (3 parallel):

1. **Mojo Integration** - Component interaction tests
   - Path: `tests/integration/`
   - Tests: Data pipeline, training workflow, model assembly

2. **Python Integration** - System structure tests
   - Path: `tests/agents/`, `tests/foundation/`
   - Tests: Agent configuration, repository structure validation

3. **Shared Integration** - Library-level workflows
   - Path: `tests/shared/integration/`
   - Tests: End-to-end training, packaging, deployment

```bash
# Local equivalent - run each test file individually
pixi run mojo tests/shared/integration/test_end_to_end.mojo
pixi run mojo tests/shared/integration/test_training_workflow.mojo
```text

### 4. `comprehensive-tests.yml` - Complete Test Suite (10 min target)

**Triggers**: All PRs and pushes to main

**17 Test Groups** (parallel execution):

| # | Group Name | Path | Test Count | Focus |
|---|---|---|---|---|
| 1 | Core: Tensors & Operations | `tests/shared/core` | ~20 | Tensor creation, arithmetic, reductions |
| 2 | Core: Layers & Activations | `tests/shared/core` | ~15 | Fully-connected, ReLU, Sigmoid, etc. |
| 3 | Core: Advanced Layers | `tests/shared/core` | ~12 | Conv2D, Pooling, Initialization |
| 4 | Core: Legacy Tests | `tests/shared/core/legacy` | ~18 | Historical test suite, edge cases |
| 5 | Training: Optimizers & Schedulers | `tests/shared/training` | ~18 | SGD, Adam, RMSprop, learning rate schedules |
| 6 | Training: Loops & Metrics | `tests/shared/training` | ~15 | Training/validation loops, accuracy, confusion matrix |
| 7 | Training: Callbacks | `tests/shared/training` | ~12 | Checkpointing, early stopping, logging |
| 8 | Data: Datasets | `tests/shared/data` | ~10 | TensorDataset, FileDataset, custom datasets |
| 9 | Data: Loaders & Samplers | `tests/shared/data` | ~12 | DataLoader, SequentialSampler, WeightedSampler |
| 10 | Data: Transforms | `tests/shared/data` | ~15 | Image transforms, augmentation, pipelines |
| 11 | Integration Tests | `tests/shared/integration` | ~8 | Full workflow tests |
| 12 | Utils & Fixtures | `tests/shared/` | ~12 | Logging, profiling, config, mocks |
| 13 | Benchmarks | `tests/shared/benchmarks` | ~4 | Performance measurements |
| 14 | Configs | `tests/configs` | ~6 | Configuration validation and merging |
| 15 | Tooling | `tests/tooling/benchmarks` | ~6 | Benchmark regression detection |
| 16 | Top-Level Tests | `tests/` | ~12 | Unit tests, training infrastructure |
| 17 | Debug & Integration | `tests/debug`, `tests/integration` | ~10 | Diagnostic and integration tests |

**Total**: 112+ Mojo test files

```bash
# Run one test group locally
cd tests/shared/core
pixi run mojo test_tensors.mojo
pixi run mojo test_arithmetic_part1.mojo
pixi run mojo test_layers.mojo
```text

## Test Naming Conventions

### Mojo Tests

```mojo
# File naming: test_*.mojo or bench_*.mojo
test_tensors.mojo
test_layers.mojo
bench_optimizers.mojo

# Function naming
fn test_tensor_creation[dtype: DType]() raises:
    ...

fn test_elementwise_add[dtype: DType, simd_width: Int]() raises:
    ...

# Use decorators for parameterization
@always_inline
fn test_batch_processing[batch_size: Int]() raises:
    ...
```text

### Python Tests

```python
# File naming: test_*.py
test_integration.py
test_supporting_directories.py

# Class naming
class TestAgentConfiguration:
    def test_agent_loads_correctly(self):
        ...

    def test_agent_has_required_fields(self):
        ...
```text

## Test Execution Details

### How Mojo Tests Run

Mojo tests are **executable files**, not using a test framework:

1. Each `.mojo` file is run as a standalone program
2. Uses `raises` context for assertions
3. Exit code 0 = pass, non-zero = fail
4. Output printed to stdout

```mojo
fn test_addition() raises:
    var result = 1 + 1
    if result != 2:
        raise Error("Addition failed")
    print("✅ Addition test passed")
```text

### How Python Tests Run

Python tests use `pytest` framework:

1. Tests discovered by filename pattern `test_*.py`
2. Uses `assert` statements for validation
3. Fixtures defined in `conftest.py`
4. Parametrization via `@pytest.mark.parametrize`

```python
def test_config_validation():
    config = load_config("example.toml")
    assert config is not None
    assert config.get("version") == "1.0"
```text

### CI/CD Test Execution Flow

```text
Pull Request Created
  ↓
[Parallel] unit-tests.yml (5 min)
  ├─ Mojo unit tests/unit/*.mojo
  └─ Python tests/unit/*.py
  ↓
[Parallel] test-gradients.yml (2 min)
  └─ Gradient correctness tests/shared/core/test_gradient_checking.mojo
  ↓
[Parallel] integration-tests.yml (8 min)
  ├─ Mojo integration tests/integration/*.mojo
  ├─ Python integration tests/agents/*, tests/foundation/*
  └─ Shared library tests/shared/integration/*.mojo
  ↓
[Parallel] comprehensive-tests.yml (10 min, 17 groups)
  ├─ Group 1: Core tensors & operations
  ├─ Group 2: Layers & activations
  ├─ ... (15 more groups)
  └─ Group 17: Debug & integration
  ↓
All Tests Pass → Merge Approved
All Tests Fail → Blocking Issue, Fixes Required
```text

## Adding New Tests

### Adding a Mojo Test

1. Create file in appropriate subdirectory (e.g., `tests/shared/core/test_new_feature.mojo`)
2. Follow naming pattern: `test_*.mojo`
3. Import test helpers from `tests/helpers/`
4. Write test functions with `fn test_*() raises:` signature
5. Use assertions from `tests/helpers/assertions.mojo`

```mojo
# tests/shared/core/test_new_feature.mojo

from tests.helpers import assertions, utils

fn test_new_feature() raises:
    """Test basic functionality of new feature."""
    var result = new_feature(42)
    assertions.assert_equal(result, expected_value, "Feature should compute correctly")
    print("✅ test_new_feature PASSED")

fn main():
    test_new_feature()
    print("All tests passed!")
```text

### Adding a Python Test

1. Create file in appropriate subdirectory (e.g., `tests/foundation/test_new_feature.py`)
2. Follow naming pattern: `test_*.py`
3. Use pytest conventions

```python
# tests/foundation/test_new_feature.py

import pytest
from pathlib import Path

class TestNewFeature:
    """Tests for new feature."""

    def test_basic_functionality(self):
        """Verify basic feature behavior."""
        result = new_feature(42)
        assert result == expected_value, f"Expected {expected_value}, got {result}"

    @pytest.mark.parametrize("input_val,expected", [
        (1, 2),
        (2, 4),
        (3, 6),
    ])
    def test_multiple_values(self, input_val, expected):
        """Test with multiple input values."""
        assert new_feature(input_val) == expected
```text

### Integration with CI/CD

New tests are automatically picked up by CI/CD workflows:

- **Unit tests** (`tests/unit/*.mojo` or `.py`) → Runs in `unit-tests.yml`
- **Gradient tests** (modifications to `test_gradient_checking.mojo`) → Runs in `test-gradients.yml`
- **Integration tests** (`tests/integration/*.mojo` or matching patterns) → Runs in `integration-tests.yml`
- **Comprehensive tests** (any `test_*.mojo` in subdirectories) → Runs in `comprehensive-tests.yml` in appropriate group

## Test Standards & Best Practices

### Naming Conventions

- **Test files**: `test_*.mojo` or `test_*.py`
- **Benchmark files**: `bench_*.mojo`
- **Test functions (Mojo)**: `fn test_*() raises:`
- **Test classes (Python)**: `class Test*:`
- **Test methods (Python)**: `def test_*(self):`

### Documentation

All test files must include:

```mojo
# tests/shared/core/test_example.mojo
"""
Unit tests for tensor operations.

Tests the following components:
- Tensor creation and initialization
- Element-wise operations
- Shape broadcasting
- Memory layout

Test coverage: >95% of core operations
"""
```text

### Assertions

Use descriptive assertion messages:

```mojo
# Bad
if a != b:
    raise Error("Failed")

# Good
if a != b:
    raise Error(f"Expected {b}, but got {a}")

# With helper
assertions.assert_equal(a, b, "Tensor values should match")
```text

### Test Independence

- Each test must be independent and runnable in isolation
- Use fixtures from `tests/shared/fixtures/` for setup
- Clean up resources after test completion
- No test should depend on output of previous tests

### Performance

- Unit tests should complete in < 1 second
- Integration tests should complete in < 10 seconds
- Benchmarks may take longer but should be clearly marked

## Coverage Goals

- **Unit tests**: > 95% coverage of core functions
- **Integration tests**: > 80% coverage of workflows
- **Gradient tests**: > 85% coverage of backward passes
- **Overall**: > 90% code coverage

## Troubleshooting

### Test Failures in CI

Check the detailed test output in GitHub Actions:

1. View workflow run: `.github/workflows/comprehensive-tests.yml`
2. Click failed test group
3. See "Run test group" step for error details
4. Review test file for issues

### Local Test Debugging

```bash
# Run failing test locally for debugging
pixi run mojo tests/shared/core/failing_test.mojo

# Run with additional output
pixi run mojo tests/shared/core/failing_test.mojo 2>&1 | tail -100

# Run with Python debugger (for Python tests)
pixi run pytest tests/agents/failing_test.py -v -s
```text

### Timeout Issues

- Increase timeout in CI workflow if needed (currently 15 min for comprehensive tests)
- For local debugging, increase Mojo memory settings
- Check for infinite loops or blocking operations

## References

- **CI/CD Workflows**: `.github/workflows/`
- **Test Helpers**: `tests/helpers/`
- **Test Fixtures**: `tests/shared/fixtures/`
- **GitHub Issues**: Implementation tracked in GitHub issues per CLAUDE.md
