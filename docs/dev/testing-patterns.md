# Standard Test Fixture Patterns

**Date**: 2025-12-05
**Context**: Issue #2478 - Standardize test fixture patterns across test files
**Scope**: Mojo testing conventions, fixture creation, and test organization

This document standardizes all test fixture patterns used across the ML Odyssey test suite. All tests
must follow these patterns for consistency, maintainability, and clarity.

---

## Quick Reference

| Pattern | When to Use | Key Functions |
|---------|------------|---|
| **Simple Test Data** | Basic unit tests | `create_random_tensor()`, `create_zeros_tensor()` |
| **Parameterized Tests** | Multiple input variations | Loop over test cases with similar structure |
| **Gradient Checking** | Backward pass validation | `check_gradients()`, `check_gradients_verbose()` |
| **Model Testing** | Full end-to-end workflows | `TestFixtures.set_seed()`, fixture functions |
| **Statistics Validation** | Distribution testing | Helper functions for mean/variance/min/max |

---

## Pattern 1: Simple Test Data Pattern

**When to use**: Basic unit tests that need simple, predictable tensor data.

### Tensor Factory Functions

Create test tensors using deterministic factory functions from
`tests/shared/fixtures/mock_tensors.mojo`:

```mojo
from tests.shared.fixtures.mock_tensors import (
    create_random_tensor,
    create_zeros_tensor,
    create_ones_tensor,
    create_sequential_tensor,
    create_constant_tensor,
)

fn test_basic_operation() raises:
    """Test basic tensor operation with simple data."""
    print("Testing basic operation...")

    # Create test tensors
    var shape = List[Int](3, 4)
    var zeros = create_zeros_tensor(shape)
    var ones = create_ones_tensor(shape)
    var random = create_random_tensor(shape, random_seed=42)

    # Use tensors in test
    assert_equal(len(zeros), 12, "Zeros tensor should have 12 elements")
    assert_equal(len(ones), 12, "Ones tensor should have 12 elements")

    print("  ✓ Basic operation test passed")
```

### Available Factory Functions

1. **`create_zeros_tensor(shape: List[Int]) -> List[Float32]`**
   - Creates tensor filled with 0.0
   - Use for initialization tests

2. **`create_ones_tensor(shape: List[Int]) -> List[Float32]`**
   - Creates tensor filled with 1.0
   - Use for identity and baseline tests

3. **`create_random_tensor(shape: List[Int], random_seed: Int = 42) -> List[Float32]`**
   - Creates random tensor with deterministic seed (default: 42)
   - Use for stochastic algorithm tests
   - Always specify seed for reproducibility

4. **`create_sequential_tensor(shape: List[Int], start: Float32 = 0.0) -> List[Float32]`**
   - Creates tensor with values [start, start+1, start+2, ...]
   - Use for indexing and reshape tests where you track element positions

5. **`create_constant_tensor(shape: List[Int], value: Float32) -> List[Float32]`**
   - Creates tensor filled with specific constant value
   - Use for testing scaled operations

### Example: Different Data Patterns

```mojo
fn test_various_tensor_patterns() raises:
    """Demonstrate all tensor creation patterns."""
    var shape = List[Int](2, 3)

    # Zero tensor (additive identity)
    var zeros = create_zeros_tensor(shape)

    # One tensor (multiplicative identity)
    var ones = create_ones_tensor(shape)

    # Random tensor (stochastic tests)
    var random = create_random_tensor(shape, random_seed=42)

    # Sequential tensor (indexing verification)
    var sequential = create_sequential_tensor(shape, start=1.0)
    # Contains: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    # Constant tensor (scaling tests)
    var constant = create_constant_tensor(shape, 5.0)
    # Contains: [5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
```

### Common Pitfall

**WRONG** - Creating list directly instead of using factory:

```mojo
# ❌ WRONG - Not using factory pattern
var zeros = List[Float32]()
for _ in range(12):
    zeros.append(0.0)
```

**RIGHT** - Using factory function:

```mojo
# ✅ CORRECT - Using factory pattern
var zeros = create_zeros_tensor(List[Int](3, 4))
```

---

## Pattern 2: Parameterized Tests Pattern

**When to use**: Testing the same logic across multiple input variations.

### Structure

Use a helper function to encapsulate test logic, then call it with different parameters:

```mojo
fn test_operation_with_different_shapes() raises:
    """Test operation across multiple tensor shapes."""
    print("Testing operation with different shapes...")

    # Define test cases: (shape, description)
    var test_cases = List[Tuple[List[Int], String]]()
    test_cases.append((List[Int](2, 3), "2x3 matrix"))
    test_cases.append((List[Int](1, 10), "1x10 vector"))
    test_cases.append((List[Int](5, 5), "5x5 square"))
    test_cases.append((List[Int](3, 4, 5), "3x4x5 tensor"))

    # Run test for each case
    for i in range(len(test_cases)):
        var shape = test_cases[i][0]
        var description = test_cases[i][1]

        print("  Testing", description)
        run_operation_test(shape)

    print("  ✓ All shape tests passed")


fn run_operation_test(shape: List[Int]) raises:
    """Helper: Run operation test for given shape."""
    var input_data = create_random_tensor(shape, random_seed=42)

    # Test operation
    assert_equal(len(input_data), compute_size(shape))
```

### Example: Multiple DType Support

```mojo
fn test_operation_with_different_dtypes() raises:
    """Test operation across multiple data types."""
    print("Testing operation with different dtypes...")

    var shape = List[Int](3, 4)

    # Test float32
    test_with_dtype(DType.float32, "float32")

    # Test float64
    test_with_dtype(DType.float64, "float64")

    # Test float16
    test_with_dtype(DType.float16, "float16")

    print("  ✓ All dtype tests passed")


fn test_with_dtype(dtype: DType, dtype_name: String) raises:
    """Helper: Test with specific dtype."""
    print("  Testing", dtype_name)

    var shape = List[Int](3, 4)
    var input = create_random_tensor(shape, random_seed=42)

    # Run operation and verify
    assert_true(len(input) > 0, "Input should not be empty")
```

### Best Practices for Parameterized Tests

1. **Use helper functions** - Extract test logic into separate functions
2. **Loop over test cases** - Use List of tuples for test parameters
3. **Descriptive names** - Include test case description in output
4. **Consistent seeds** - Use same seed across test variations for reproducibility
5. **Single concern** - Each helper tests one specific aspect

---

## Pattern 3: Gradient Checking Pattern

**When to use**: Validating backward pass implementations using numerical differentiation.

### Using check_gradients Helper

```mojo
from shared.testing import check_gradients, check_gradients_verbose

fn test_activation_gradient() raises:
    """Test activation function backward pass using gradient checking."""
    print("Testing activation gradient...")

    var shape = List[Int](3, 4)
    var input = create_random_tensor(shape, random_seed=42)

    # Define forward function
    fn forward(x: ExTensor) raises escaping -> ExTensor:
        return activation_func(x)

    # Define backward function
    fn backward(grad_out: ExTensor, x: ExTensor) raises escaping -> ExTensor:
        return activation_backward(grad_out, x)

    # Check gradients using numerical differentiation
    var passed = check_gradients(forward, backward, input)
    assert_true(passed, "Gradient check failed")

    print("  ✓ Gradient check passed")
```

### Verbose Mode for Debugging

When gradients fail numerical check, use verbose mode to see differences:

```mojo
fn test_complex_operation_gradient() raises:
    """Test with verbose output for debugging."""
    print("Testing complex operation gradient...")

    var shape = List[Int](2, 3)
    var input = create_random_tensor(shape, random_seed=42)

    fn forward(x: ExTensor) raises escaping -> ExTensor:
        return complex_operation(x)

    fn backward(grad_out: ExTensor, x: ExTensor) raises escaping -> ExTensor:
        return complex_operation_backward(grad_out, x)

    # Use verbose mode to see numerical vs analytical gradients
    var passed = check_gradients_verbose(forward, backward, input)
    assert_true(passed, "Gradient check failed - see output above")
```

### Pattern for Mixed Positive/Negative Inputs

For activations like ReLU with discontinuities, test edge cases:

```mojo
fn test_relu_mixed_inputs_gradient() raises:
    """Test ReLU gradient with mixed positive/negative inputs."""
    print("Testing ReLU gradient (mixed inputs)...")

    var shape = List[Int](3, 4)
    var input = create_zeros_tensor(shape)

    # Set mixed positive and negative values (avoid 0.0 at discontinuity)
    input[0] = 1.0    # Positive - gradient flows
    input[1] = -1.0   # Negative - gradient blocked
    input[2] = 2.0    # Positive
    input[3] = -2.0   # Negative
    input[4] = 0.1    # Small positive
    input[5] = -0.1   # Small negative

    fn forward(x: ExTensor) raises escaping -> ExTensor:
        return relu(x)

    fn backward(grad_out: ExTensor, x: ExTensor) raises escaping -> ExTensor:
        return relu_backward(grad_out, x)

    var passed = check_gradients(forward, backward, input)
    assert_true(passed, "ReLU gradient check failed for mixed inputs")
    print("  ✓ ReLU gradient correct (mixed inputs)")
```

### Gradient Checking Tolerances

Different operations need different tolerances due to floating-point precision:

```mojo
# From backward-pass-catalog.md:
# - Conv2D: rtol=1e-2, atol=1e-2
# - Cross-entropy: rtol=1e-3, atol=1e-3
# - Softmax: rtol=1e-3, atol=5e-4
# - Linear: rtol=1e-3, atol=1e-4
```

### Warning: Normalization Backward — Never Use `grad_output=ones`

**Applies to**: `batch_norm2d_backward`, `layer_norm_backward`, and any normalization layer
that computes normalized inputs `x_norm = (x - mean) / std`.

**The problem**: Passing `grad_output=ones` into a normalization backward pass produces
analytically-zero gradients for `dL/dx`. This is not a bug — it is a mathematical identity.
The zero result will match numerical finite differences (which also converge to zero), so the
gradient check passes trivially without actually validating anything.

**Mathematical explanation**: Batch normalization normalizes each channel across the spatial
and batch dimensions. By construction, the normalized values sum to zero:

```text
x_norm = (x - mean(x)) / std(x)
=> sum(x_norm) = 0   (always, by definition of mean-centering)
```

The gradient of the loss with respect to the input contains a term proportional to
`sum(grad_output * x_norm)`. When `grad_output = ones`, this term is `sum(x_norm) = 0`,
which cancels other gradient terms and yields `dL/dx = 0` for every element — a false result
that hides implementation errors entirely.

**Failing pattern (do not use)**:

```mojo
# ❌ WRONG — grad_output=ones causes sum(x_norm)=0 cancellation
var grad_output = ones_like(output)
var result = batch_norm2d_backward(grad_output, x, gamma, ...)
var grad_input = result[0]
# grad_input is all zeros — gradient check passes trivially, verifies nothing
```

**Correct pattern** — use a non-uniform `grad_output` and a scalar-loss `forward_for_grad`
closure that matches it (canonical implementation: `test_normalization.mojo:316-360`):

```mojo
# ✅ CORRECT — non-uniform grad_output breaks the symmetry
var grad_output = zeros_like(output)
for i in range(output.numel()):
    # Alternating pattern avoids accidental symmetry
    var val = Float32(i % 4) * Float32(0.25) - Float32(0.3)
    grad_output._data.bitcast[Float32]()[i] = val

# forward_for_grad computes the weighted scalar loss sum(output * grad_output),
# so numerical finite differences match what backward should return.
fn forward_for_grad(inp: ExTensor) raises -> ExTensor:
    var result = batch_norm2d(inp, gamma, beta, running_mean, running_var,
                              training=True, epsilon=1e-5)
    var out = result[0]
    var weighted = multiply(out, grad_output)
    var scalar = weighted
    while scalar.dim() > 0:
        scalar = reduce_sum(scalar, axis=0, keepdims=False)
    return scalar

var numerical_grad = compute_numerical_gradient(forward_for_grad, x, epsilon=1e-3)
# Now compare numerical_grad against the analytical grad_input — a real test
```

**Rule**: For normalization layers, never use `grad_output=ones` in backward gradient checks.

---

## Pattern 4: Model Testing Pattern

**When to use**: Testing complete model workflows including initialization and forward/backward passes.

### Using TestFixtures for Determinism

```mojo
from tests.shared.conftest import TestFixtures, assert_true, assert_equal

fn test_model_initialization() raises:
    """Test model initializes correctly."""
    print("Testing model initialization...")

    # Use deterministic seed
    TestFixtures.set_seed()

    # Create model
    var model = SimpleMLP(input_size=10, hidden_size=20, output_size=3)

    # Verify structure
    assert_equal(model.input_size, 10, "Input size should be 10")
    assert_equal(model.hidden_size, 20, "Hidden size should be 20")
    assert_equal(model.output_size, 3, "Output size should be 3")

    print("  ✓ Model initialization test passed")
```

### Complete Training Loop Test

```mojo
fn test_model_training_loop() raises:
    """Test complete training workflow."""
    print("Testing model training loop...")

    # Set seed for reproducibility
    TestFixtures.set_seed()

    # Create model and data
    var model = SimpleMLP(input_size=10, hidden_size=20, output_size=3)
    var batch_size = 32
    var n_samples = 100

    var shape_input = List[Int](batch_size, 10)
    var shape_labels = List[Int](batch_size)

    var input_data = create_random_tensor(shape_input, random_seed=42)
    var labels = create_sequential_tensor(shape_labels, start=0.0)

    # Forward pass
    var output = model.forward(input_data)

    # Compute loss
    var loss = compute_loss(output, labels)

    # Verify output shape
    assert_equal(len(output), batch_size, "Output should match batch size")
    assert_true(loss > 0.0, "Loss should be positive")

    print("  ✓ Training loop test passed")
```

### Testing with Multiple Seeds

```mojo
fn test_model_determinism() raises:
    """Test model produces same results with same seed."""
    print("Testing model determinism...")

    # First run
    TestFixtures.set_seed()
    var model1 = SimpleMLP(10, 20, 3)
    var output1 = model1.forward(input_data)

    # Second run with same seed
    TestFixtures.set_seed()
    var model2 = SimpleMLP(10, 20, 3)
    var output2 = model2.forward(input_data)

    # Outputs should be identical
    assert_tensors_equal(output1, output2, epsilon=1e-6,
                         message="Outputs should be deterministic")

    print("  ✓ Determinism test passed")
```

---

## Pattern 5: Statistics Validation Pattern

**When to use**: Validating statistical properties of tensors (mean, variance, min, max).

### Computing Statistics

```mojo
from tests.shared.fixtures.mock_tensors import (
    tensor_mean,
    tensor_min,
    tensor_max,
)

fn test_tensor_statistics() raises:
    """Test tensor statistical properties."""
    print("Testing tensor statistics...")

    var shape = List[Int](100)
    var random_data = create_random_tensor(shape, random_seed=42)

    # Compute statistics
    var mean = tensor_mean(random_data)
    var min_val = tensor_min(random_data)
    var max_val = tensor_max(random_data)

    # Verify ranges
    assert_true(mean > min_val, "Mean should be greater than min")
    assert_true(mean < max_val, "Mean should be less than max")
    assert_true(max_val - min_val > 0.0, "Range should be positive")

    print("  Mean:", mean)
    print("  Min:", min_val)
    print("  Max:", max_val)
    print("  ✓ Statistics test passed")
```

### Statistical Distribution Validation

```mojo
fn test_normal_distribution_properties() raises:
    """Test random tensor has normal distribution properties."""
    print("Testing normal distribution properties...")

    var shape = List[Int](1000)  # Large sample for stable statistics
    var random_data = create_random_tensor(shape, random_seed=42)

    var mean = tensor_mean(random_data)
    var variance = compute_variance(random_data, mean)
    var std = sqrt(variance)

    # Standard normal should have mean ≈ 0, std ≈ 1
    # Allow tolerance for finite sample
    assert_almost_equal(Float64(mean), 0.0, 0.1,
                        "Mean should be near 0")
    assert_almost_equal(std, 1.0, 0.1,
                        "Std should be near 1")

    print("  Mean:", mean, "(expected: 0.0)")
    print("  Std:", std, "(expected: 1.0)")
    print("  ✓ Distribution test passed")
```

### Tensor Comparison Assertions

```mojo
from tests.shared.fixtures.mock_tensors import (
    assert_tensors_equal,
    assert_shape_equal,
)

fn test_tensor_equality() raises:
    """Test tensor comparison utilities."""
    var a = create_ones_tensor([2, 3])
    var b = create_constant_tensor([2, 3], 1.0)

    # Should be equal within epsilon
    assert_tensors_equal(a, b, epsilon=1e-6,
                         message="Tensors should be equal")

    print("  ✓ Equality test passed")
```

---

## Best Practices and Conventions

### 1. Test Organization

**CORRECT** - Tests organized by functionality:

```mojo
# ============================================================================
# Basic Functionality Tests
# ============================================================================

fn test_initialization() raises:
    """Test basic initialization."""
    pass

fn test_forward_pass() raises:
    """Test forward computation."""
    pass

# ============================================================================
# Edge Case Tests
# ============================================================================

fn test_empty_input() raises:
    """Test with empty input."""
    pass

fn test_large_input() raises:
    """Test with large input."""
    pass
```

### 2. Deterministic Seeds

**ALWAYS** use fixed seeds for reproducible tests:

```mojo
# ✅ CORRECT - Uses fixed seed
var random_data = create_random_tensor([10, 10], random_seed=42)

# ❌ WRONG - Non-deterministic
var random_data = create_random_tensor([10, 10])  # Uses default seed
```

### 3. Assertion Messages

**ALWAYS** include descriptive error messages:

```mojo
# ✅ CORRECT - Clear error message
assert_equal(tensor.numel(), 12, "Tensor should have 12 elements")

# ❌ WRONG - No context on failure
assert_equal(tensor.numel(), 12)
```

### 4. Test Output

**ALWAYS** include progress output:

```mojo
fn test_operation() raises:
    print("Testing operation...")
    # ... test code ...
    print("  ✓ Operation test passed")
```

### 5. Test Structure

**CORRECT** - Each test is self-contained:

```mojo
fn test_specific_feature() raises:
    """Test specific feature in isolation."""
    print("Testing specific feature...")

    # Create clean fixtures
    var shape = List[Int](3, 4)
    var data = create_random_tensor(shape, random_seed=42)

    # Test only this feature
    var result = specific_feature(data)

    # Assert results
    assert_true(result, "Feature should work")

    print("  ✓ Feature test passed")
```

### 6. Fixture Scope

**CORRECT** - Fixtures created fresh in each test:

```mojo
# ✅ CORRECT - Each test gets fresh data
fn test_operation_a() raises:
    var data = create_zeros_tensor([3, 4])
    # ...

fn test_operation_b() raises:
    var data = create_zeros_tensor([3, 4])
    # ...
```

**WRONG** - Shared fixtures between tests:

```mojo
# ❌ WRONG - Tests depend on shared state
var shared_data = create_zeros_tensor([3, 4])

fn test_operation_a() raises:
    modify(shared_data)  # Changes shared data!

fn test_operation_b() raises:
    # Fails because shared_data was modified by test_a
```

---

## Common Test Patterns by Domain

### Activation Function Tests

```mojo
fn test_activation_forward() raises:
    """Test activation forward pass."""
    var shape = List[Int](5, 10)
    var input = create_random_tensor(shape, random_seed=42)

    var output = activation_func(input)

    assert_equal(output.shape(), input.shape(),
                 "Output shape should match input shape")
    print("  ✓ Activation forward test passed")

fn test_activation_gradient() raises:
    """Test activation backward pass."""
    var shape = List[Int](5, 10)
    var input = create_random_tensor(shape, random_seed=42)

    fn forward(x: ExTensor) raises escaping -> ExTensor:
        return activation_func(x)

    fn backward(grad_out: ExTensor, x: ExTensor) raises escaping -> ExTensor:
        return activation_backward(grad_out, x)

    var passed = check_gradients(forward, backward, input)
    assert_true(passed, "Gradient check failed")
    print("  ✓ Activation gradient test passed")
```

### Loss Function Tests

```mojo
fn test_loss_computation() raises:
    """Test loss function computation."""
    var batch_size = 32
    var num_classes = 10

    var logits = create_random_tensor([batch_size, num_classes], random_seed=42)
    var labels = create_sequential_tensor([batch_size], start=0.0)

    var loss = loss_function(logits, labels)

    assert_true(loss > 0.0, "Loss should be positive")
    print("  ✓ Loss computation test passed")
```

### Layer Tests

```mojo
fn test_layer_forward() raises:
    """Test layer forward pass."""
    TestFixtures.set_seed()

    var layer = DenseLayer(input_size=10, output_size=5)
    var input = create_random_tensor([32, 10], random_seed=42)

    var output = layer.forward(input)

    assert_equal(output.shape(), [32, 5],
                 "Output shape should be [batch, output_size]")
    print("  ✓ Layer forward test passed")

fn test_layer_gradient() raises:
    """Test layer backward pass."""
    TestFixtures.set_seed()

    var layer = DenseLayer(input_size=10, output_size=5)
    var input = create_random_tensor([32, 10], random_seed=42)

    fn forward(x: ExTensor) raises escaping -> ExTensor:
        return layer.forward(x)

    fn backward(grad_out: ExTensor, x: ExTensor) raises escaping -> ExTensor:
        return layer.backward(grad_out, x)

    var passed = check_gradients(forward, backward, input)
    assert_true(passed, "Gradient check failed")
    print("  ✓ Layer gradient test passed")
```

---

## File Organization Convention

### Test File Structure

```mojo
"""Module docstring with test overview.

Comprehensive test suite for [component] including:
- Feature 1
- Feature 2
- Feature 3

Testing strategy:
- Functional correctness
- Edge cases
- Numerical stability
"""

from tests.shared.conftest import (
    assert_true,
    assert_equal,
    assert_almost_equal,
    TestFixtures,
)
from tests.shared.fixtures.mock_tensors import (
    create_random_tensor,
    create_zeros_tensor,
)
from shared.core import ExTensor


# ============================================================================
# Category 1: Core Functionality
# ============================================================================

fn test_feature_1() raises:
    """Test specific feature."""
    pass


fn test_feature_2() raises:
    """Test another feature."""
    pass


# ============================================================================
# Category 2: Edge Cases
# ============================================================================

fn test_edge_case_1() raises:
    """Test edge case."""
    pass
```

---

## Migration Guide

### From Old Patterns to Standard Patterns

**OLD** - Manual tensor creation:

```mojo
var zeros = List[Float32](capacity=12)
for _ in range(12):
    zeros.append(0.0)
```

**NEW** - Using factory functions:

```mojo
var zeros = create_zeros_tensor(List[Int](3, 4))
```

**OLD** - Inline test cases:

```mojo
# Test with shape 1
var data = create_zeros_tensor([2, 3])
test_operation(data)

# Test with shape 2
var data = create_zeros_tensor([3, 4])
test_operation(data)
```

**NEW** - Parameterized tests:

```mojo
var shapes = List[List[Int]]()
shapes.append(List[Int](2, 3))
shapes.append(List[Int](3, 4))

for i in range(len(shapes)):
    test_operation(shapes[i])
```

---

## Troubleshooting

### Test Data Not Initialized

**Problem**: Getting "index out of bounds" when accessing tensor data

**Cause**: Using empty shape or wrong size calculation

**Solution**: Verify shape initialization

```mojo
# ❌ WRONG - Empty shape creates 0D scalar (1 element)
var shape = List[Int]()
var tensor = create_zeros_tensor(shape)
tensor[1] = 5.0  # Crash - only 1 element

# ✅ CORRECT - Explicit dimensions
var shape = List[Int](2, 3)  # 2x3 = 6 elements
var tensor = create_zeros_tensor(shape)
tensor[1] = 5.0  # OK
```

### Gradient Check Failing

**Problem**: Numerical and analytical gradients don't match

**Causes**:

1. Input too close to discontinuity (ReLU at 0)
2. Tolerance too strict for float32
3. Numerical differentiation step size wrong

**Solution**: Try different input values or relax tolerance

```mojo
# ❌ WRONG - Input at discontinuity
var input = create_zeros_tensor([3, 4])  # All zeros
var passed = check_gradients(forward, backward, input)  // Fails!

# ✅ CORRECT - Input away from discontinuity
var input = create_constant_tensor([3, 4], 1.0)  // Positive values
var passed = check_gradients(forward, backward, input)
```

### Determinism Issues

**Problem**: Test passes sometimes, fails other times

**Cause**: Using non-deterministic random seed

**Solution**: Always set seed explicitly

```mojo
# ❌ WRONG - Non-deterministic
TestFixtures.set_seed()  // Uses default seed which may vary

# ✅ CORRECT - Explicit seed
var data = create_random_tensor(shape, random_seed=42)
```

---

## See Also

- [Backward Pass Catalog](backward-pass-catalog.md) - Detailed gradient checking patterns
- [Mojo Test Failure Patterns](mojo-test-failure-patterns.md) - Common compilation errors
- `/tests/shared/conftest.mojo` - Assertion function documentation
- `/tests/shared/fixtures/` - Fixture implementations
