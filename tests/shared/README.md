# Shared Library Test Suite

## Overview

This directory contains comprehensive tests for the ML Odyssey shared library. The test suite follows Test-Driven
Development (TDD) principles and was designed to guide implementation in Issue #49.

## Quick Start

### Running Tests Locally

```bash
# Run all shared library tests
mojo test tests/shared/

# Run specific module tests
mojo test tests/shared/core/
mojo test tests/shared/training/
mojo test tests/shared/data/
mojo test tests/shared/utils/

# Run specific test file
mojo test tests/shared/core/test_layers.mojo

# Run with coverage
mojo test --coverage tests/shared/

# Run integration tests
mojo test tests/shared/integration/

# Run benchmarks
mojo test tests/shared/benchmarks/
```text

### Running Tests in CI

Tests run automatically on:

- Pull requests affecting `shared/` or `tests/shared/`
- Pushes to `main` branch
- Scheduled nightly builds

See `.github/workflows/test-shared.yml` for CI configuration.

## Directory Structure

```text
tests/shared/
├── README.md (this file)
├── conftest.mojo                    # Shared test fixtures
├── __init__.mojo                    # Test package root
├── core/                            # Core module tests
│   ├── test_layers.mojo             # Layer implementations
│   ├── test_tensors.mojo            # Tensor operations
│   ├── test_activations.mojo        # Activation functions
│   ├── test_initializers.mojo       # Parameter initialization
│   └── test_module.mojo             # Module base class
├── training/                        # Training module tests
│   ├── test_optimizers.mojo         # Optimizer implementations
│   ├── test_schedulers.mojo         # LR schedulers
│   ├── test_metrics.mojo            # Training metrics
│   ├── test_callbacks.mojo          # Training callbacks
│   └── test_loops.mojo              # Training loops
├── data/                            # Data module tests
│   ├── test_datasets_part1.mojo     # Dataset implementations (Part 1, ADR-009 split)
│   ├── test_datasets_part2.mojo     # Dataset implementations (Part 2, ADR-009 split)
│   ├── test_loaders.mojo            # Data loaders
│   └── test_transforms.mojo         # Data transforms
├── utils/                           # Utils module tests
│   ├── test_logging.mojo            # Logging utilities
│   ├── test_visualization.mojo      # Visualization tools
│   └── test_config.mojo             # Configuration management
├── integration/                     # Integration tests
│   ├── test_training_workflow.mojo  # End-to-end training
│   ├── test_data_pipeline.mojo      # Data loading workflow
│   └── test_end_to_end.mojo         # Full workflow tests
├── benchmarks/                      # Performance benchmarks
│   ├── bench_optimizers.mojo        # Optimizer performance
│   ├── bench_layers.mojo            # Layer performance
│   └── bench_data_loading.mojo      # Data loading performance
└── fixtures/                        # Test data and fixtures
    ├── images/                      # Sample images
    ├── tensors/                     # Pre-computed tensors
    ├── models/                      # Small model weights
    └── reference/                   # Reference outputs
```text

## Test Philosophy

### Quality Over Quantity

- **Focus on critical paths**: Test what matters most
- **Test behavior, not implementation**: Tests should survive refactoring
- **Each test adds value**: Not just coverage percentage
- **Skip trivial code**: No tests for simple getters or obvious constructors

### Coverage Goals

- **Line Coverage**: ≥90% of all lines
- **Branch Coverage**: ≥85% of all branches
- **Function Coverage**: 100% of public functions
- **Critical Path Coverage**: 100% of core functionality

**Rule of Thumb**: If deleting a test wouldn't reduce confidence in the code, delete it.

### Real Implementations

- ✅ Use real implementations whenever possible
- ✅ Create simple, concrete test data
- ✅ Minimal mocking (only for complex dependencies)
- ❌ Do NOT create elaborate mock frameworks

## Test Categories

### 1. Unit Tests (Critical)

Test individual functions and components in isolation.

### What to Test

- Core functionality and main use cases
- Security-sensitive code (validation, boundaries)
- Public API contracts
- Error handling for critical paths
- Integration points between modules

### What to Skip

- Trivial getters/setters
- Simple constructors with no logic
- Private implementation details

**Example**: `test_sgd_basic_update()` validates SGD parameter update formula

### 2. Integration Tests (Important)

Test cross-module workflows and component interactions.

### What to Test

- End-to-end training workflows
- Data loading pipelines
- Model construction and forward passes
- Gradient computation flows

**Example**: `test_training_workflow()` runs complete training loop

### 3. Performance Benchmarks (Important)

Establish baseline metrics and prevent regressions.

### What to Test

- Optimizer update throughput
- Layer forward/backward pass speed
- Data loading performance
- Memory allocation patterns

### Targets

- Within 2x of PyTorch for optimizers
- SIMD-vectorized operations
- < 5 minutes total CI test time

**Example**: `bench_sgd_update_speed()` measures updates per second

### 4. Numerical Accuracy Tests (Critical for ML)

Validate correctness against reference implementations.

### What to Test

- Compare optimizer updates to PyTorch
- Validate layer outputs against known values
- Check gradient computations
- Verify metric calculations

**Example**: `test_sgd_matches_pytorch()` ensures numerical equivalence

## Test Naming Conventions

### Test Functions

```mojo
# Pattern: test_<component>_<behavior>
fn test_sgd_basic_update():
    """Test SGD performs basic parameter update."""
    pass

fn test_conv2d_output_shape():
    """Test Conv2D computes correct output shape."""
    pass
```text

### Property Tests

```mojo
# Pattern: test_<component>_property_<property>
fn test_optimizer_property_decreasing_loss():
    """Property: Optimizer should decrease loss on convex function."""
    pass
```text

### Benchmark Functions

```mojo
# Pattern: bench_<component>_<metric>
fn bench_sgd_update_speed():
    """Benchmark SGD parameter update throughput."""
    pass
```text

## Writing New Tests

### 1. Choose the Right Test Type

- **Unit test**: Testing a single function/component
- **Integration test**: Testing multiple components together
- **Benchmark**: Measuring performance
- **Property test**: Validating mathematical invariants

### 2. Follow Mojo Best Practices

```mojo
# Use 'fn' for test functions (type safety)
fn test_my_feature():
    """Clear docstring describing what is tested."""
    # Arrange: Set up test data
    var data = create_test_data()

    # Act: Execute the code under test
    var result = function_under_test(data)

    # Assert: Verify expected behavior
    assert_equal(result, expected_value)
```text

### 3. Use Test Fixtures

```mojo
# Import shared fixtures
from tests.shared.conftest import TestFixtures

fn test_with_fixture():
    var fixture = TestFixtures.simple_linear_model()
    var output = fixture.forward(test_input)
    assert_shape_equal(output, expected_shape)
```text

### 4. Write Clear Assertions

```mojo
# Good: Specific assertion with clear message
assert_almost_equal(loss, 0.5, tolerance=1e-6)

# Good: Test one thing per test
fn test_sgd_updates_parameters():
    """Test SGD updates parameters (not momentum or weight decay)."""
    pass

# Avoid: Testing multiple unrelated things
fn test_sgd_everything():
    """Test SGD updates, momentum, weight decay, etc."""  # Too broad
    pass
```text

## Test Utilities

### Assertion Functions

Located in `conftest.mojo`:

```mojo
assert_true(condition, message="")           # Bool assertion
assert_equal(a, b, message="")               # Exact equality
assert_almost_equal(a, b, tolerance=1e-6)    # Float near-equality
assert_tensor_equal(a, b, tolerance=1e-6)    # Tensor equality
assert_shape_equal(tensor, expected_shape)   # Shape check
```text

### Test Fixtures

Located in `conftest.mojo`:

```mojo
TestFixtures.small_tensor()                  # 3x3 test tensor
TestFixtures.simple_linear_model()           # Linear layer
TestFixtures.synthetic_dataset(n=100)        # Synthetic data
```text

## Coverage Reporting

### Generate Coverage Report

```bash
# Run tests with coverage
mojo test --coverage tests/shared/

# Check coverage threshold
python scripts/check_coverage.py --threshold 90 --path shared/

# Generate HTML report
mojo test --coverage --html tests/shared/
open coverage_html_report/index.html
```text

### Coverage Exemptions

The following are exempt from coverage requirements:

- Debug-only code
- Unreachable error paths
- Platform-specific code not available on test machine
- Deprecated code marked for removal

## Common Test Patterns

### Testing Layers

```mojo
fn test_linear_forward():
    """Test Linear layer forward pass."""
    # Create layer with known dimensions
    var layer = Linear(in_features=10, out_features=5)
    layer.weights.fill(0.1)  # Known weights
    layer.bias.fill(0.0)

    # Create input
    var input = Tensor.ones(2, 10)  # batch_size=2

    # Forward pass
    var output = layer.forward(input)

    # Check shape
    assert_shape_equal(output, Shape(2, 5))

    # Check values (known computation)
    let expected_value = 10 * 0.1  # sum of weights
    assert_almost_equal(output[0, 0], expected_value)
```text

### Testing Optimizers

```mojo
fn test_sgd_parameter_update():
    """Test SGD updates parameters correctly."""
    # Initial parameters
    var params_list = List[Float32]()
    params_list.append(1.0)
    params_list.append(2.0)
    params_list.append(3.0)
    var params_shape = List[Int]()
    params_shape.append(3)
    var params = ExTensor(params_shape, DType.float32)
    for i in range(len(params_list)):
        params._set_float32(i, params_list[i])

    var grads_list = List[Float32]()
    grads_list.append(0.1)
    grads_list.append(0.2)
    grads_list.append(0.3)
    var grads_shape = List[Int]()
    grads_shape.append(3)
    var grads = ExTensor(grads_shape, DType.float32)
    for i in range(len(grads_list)):
        grads._set_float32(i, grads_list[i])

    # Create optimizer
    var optimizer = SGD(lr=0.1)

    # Perform update
    optimizer.step(params, grads)

    # Verify update: params = params - lr * grads
    assert_almost_equal(params[0], 0.99)  # 1.0 - 0.1*0.1
    assert_almost_equal(params[1], 1.98)  # 2.0 - 0.1*0.2
    assert_almost_equal(params[2], 2.97)  # 3.0 - 0.1*0.3
```text

### Testing Data Loaders

```mojo
fn test_dataloader_batching():
    """Test DataLoader creates correct batch sizes."""
    # Create dataset
    var dataset = TestFixtures.synthetic_dataset(n_samples=100)

    # Create loader
    var loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Verify batch count
    let n_batches = len(loader)
    assert_equal(n_batches, 4)  # 100/32 = 3 full + 1 partial

    # Verify batch sizes
    var batch_sizes = List[Int]()
    for batch in loader:
        batch_sizes.append(batch.size())

    assert_equal(batch_sizes[0], 32)
    assert_equal(batch_sizes[1], 32)
    assert_equal(batch_sizes[2], 32)
    assert_equal(batch_sizes[3], 4)  # Partial batch
```text

### Testing Integration Workflows

```mojo
fn test_basic_training_loop():
    """Test complete training loop with validation."""
    # Create small model
    var model = SimpleModel(in_features=10, out_features=2)

    # Create synthetic data
    var train_data = TestFixtures.synthetic_dataset(n_samples=100)
    var val_data = TestFixtures.synthetic_dataset(n_samples=20)

    # Create optimizer
    var optimizer = SGD(lr=0.01)

    # Train for 5 epochs
    var losses = List[Float32]()
    for epoch in range(5):
        var epoch_loss = train_epoch(model, train_data, optimizer)
        losses.append(epoch_loss)

    # Verify loss decreases
    assert_true(losses[-1] < losses[0])

    # Verify validation accuracy
    var val_acc = evaluate(model, val_data)
    assert_true(val_acc > 0.5)  # Better than random
```text

## Performance Benchmarking

### Benchmark Structure

```mojo
fn bench_sgd_update_speed():
    """Benchmark SGD parameter update throughput."""
    let param_counts = List[Int](1_000_000, 10_000_000)
    var results = List[BenchmarkResult]()

    for n in param_counts:
        var params = Tensor.randn(n, seed=42)
        var grads = Tensor.randn(n, seed=42)
        var optimizer = SGD(lr=0.01)

        # Warmup
        for _ in range(10):
            optimizer.step(params, grads)

        # Benchmark
        let start = time.now()
        let n_iters = 100
        for _ in range(n_iters):
            optimizer.step(params, grads)
        let elapsed = (time.now() - start) / n_iters

        # Record results
        let throughput = n / elapsed  # params/second
        results.append(BenchmarkResult(
            name="SGD-" + str(n),
            duration_ms=elapsed * 1000,
            throughput=throughput
        ))

    # Print results
    print_benchmark_results(results)
```text

### Performance Targets

- **Optimizers**: Within 2x of PyTorch
- **Layers**: ≥ 1 TFLOPS for Conv2D
- **Data Loading**: ≥ 100 images/second
- **Memory**: < 10% overhead beyond theoretical minimum

## Troubleshooting

### Test Failures

```bash
# Run specific failing test for debugging
mojo test tests/shared/core/test_layers.mojo::test_linear_forward

# Add debug output in test
fn test_my_feature():
    var result = function_under_test(data)
    print("Result:", result)  # Debug output
    assert_equal(result, expected)
```text

### Coverage Issues

```bash
# Identify uncovered lines
mojo test --coverage --show-missing tests/shared/

# Exclude files from coverage
# Add to .coveragerc
# [run]
# omit = */__init__.mojo
```text

### Slow Tests

```bash
# Identify slow tests
mojo test --profile tests/shared/

# Run only fast tests
mojo test -m "not slow" tests/shared/
```text

## Contributing

### Before Adding Tests

1. **Check existing tests**: Avoid duplication
1. **Choose appropriate test type**: Unit vs integration vs benchmark
1. **Follow naming conventions**: `test_<component>_<behavior>`
1. **Write clear docstrings**: Explain what is tested

### Test Review Checklist

- [ ] Test name follows convention
- [ ] Clear docstring describing what is tested
- [ ] Tests one specific behavior
- [ ] Uses appropriate assertions
- [ ] Runs quickly (< 1 second for unit tests)
- [ ] Deterministic (no random failures)
- [ ] Independent (doesn't depend on other tests)

## Related Documentation

- [Test Architecture](../../notes/issues/48/test-architecture.md) - Comprehensive test design
- [Shared Library](../../shared/README.md) - Code being tested
- [Mojo Testing Docs](https://docs.modular.com/mojo/tools/testing/) - Official Mojo testing guide
- [Issue #48](https://github.com/HomericIntelligence/ProjectOdyssey/issues/48) - Test implementation issue

## Questions

- Check the test architecture document
- Review existing test files for examples
- Open a discussion in GitHub issues
- Refer to Mojo testing documentation

---

**Test Suite Status**: 📝 In Development

**Coverage Target**: ≥90% line coverage

**Last Updated**: 2025-11-09
