"""Integration tests for ExTensor operations - Part 2: Dtype and Multi-dimensional.

Tests multiple dtype operations, multi-dimensional operations, and ML-like patterns.
Split from test_integration.mojo per ADR-009 to avoid heap corruption.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_integration.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""

# Import ExTensor and operations
from shared.core import (
    ExTensor,
    zeros,
    ones,
    full,
    arange,
    eye,
    linspace,
    add,
    subtract,
    multiply,
)

# Import test helpers
from tests.shared.conftest import (
    assert_dtype,
    assert_numel,
    assert_dim,
    assert_value_at,
    assert_all_values,
    assert_all_close,
)


# ============================================================================
# Test multiple dtype operations
# ============================================================================


fn test_same_dtype_consistency() raises:
    """Test that operations preserve dtype consistently."""
    var shape = List[Int]()
    shape.append(5)

    var a32 = ones(shape, DType.float32)
    var b32 = ones(shape, DType.float32)
    var result32 = add(a32, b32)
    assert_dtype(result32, DType.float32, "float32 + float32 should be float32")

    var a64 = ones(shape, DType.float64)
    var b64 = ones(shape, DType.float64)
    var result64 = add(a64, b64)
    assert_dtype(result64, DType.float64, "float64 + float64 should be float64")


fn test_int_dtype_operations() raises:
    """Test operations with integer dtypes."""
    var shape = List[Int]()
    shape.append(5)

    var a = full(shape, 3.0, DType.int32)
    var b = full(shape, 2.0, DType.int32)
    var result = add(a, b)

    assert_dtype(result, DType.int32, "int32 + int32 should be int32")
    assert_all_values(result, 5.0, 1e-6, "3 + 2 = 5")


# ============================================================================
# Test multi-dimensional operations
# ============================================================================


fn test_2d_elementwise_operations() raises:
    """Test element-wise operations on 2D tensors."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var a = full(shape, 5.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)

    var result = subtract(a, b)  # All 2s

    assert_numel(result, 12, "Result should have 12 elements")
    assert_all_values(result, 2.0, 1e-6, "5 - 3 = 2 for all elements")


fn test_3d_operations() raises:
    """Test operations on 3D tensors."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)
    var a = ones(shape, DType.float32)
    var b = full(shape, 0.5, DType.float32)

    var result = multiply(a, b)  # All 0.5s

    assert_numel(result, 24, "Result should have 24 elements")
    assert_all_values(result, 0.5, 1e-6, "1 * 0.5 = 0.5 for all elements")


# ============================================================================
# Test realistic ML-like patterns
# ============================================================================


fn test_linear_transformation_pattern() raises:
    """Test pattern similar to linear layer: W*x + b."""
    var shape = List[Int]()
    shape.append(5)

    # Simulate weights, input, and bias
    var W = full(shape, 2.0, DType.float32)
    var x = ones(shape, DType.float32)
    var b = full(shape, 0.5, DType.float32)

    # Linear transformation: W*x + b
    var Wx = multiply(W, x)  # 2 * 1 = 2
    var result = add(Wx, b)  # 2 + 0.5 = 2.5

    assert_all_values(result, 2.5, 1e-6, "Linear transformation result")


fn test_gradient_descent_update_pattern() raises:
    """Test pattern similar to gradient descent: w - lr * grad."""
    var shape = List[Int]()
    shape.append(5)

    var w = ones(shape, DType.float32)  # weights
    var grad = full(shape, 0.2, DType.float32)  # gradients
    var lr = full(shape, 0.1, DType.float32)  # learning rate

    # Update: w - lr * grad
    var lr_grad = multiply(lr, grad)  # 0.1 * 0.2 = 0.02
    var new_w = subtract(w, lr_grad)  # 1 - 0.02 = 0.98

    assert_all_values(new_w, 0.98, 1e-6, "Weight update pattern")


fn test_batch_normalization_pattern() raises:
    """Test pattern similar to batch normalization: (x - mean) * scale."""
    var shape = List[Int]()
    shape.append(5)

    var x = full(shape, 5.0, DType.float32)
    var mean = full(shape, 3.0, DType.float32)
    var scale = full(shape, 2.0, DType.float32)

    # (x - mean) * scale
    var centered = subtract(x, mean)  # 5 - 3 = 2
    var result = multiply(centered, scale)  # 2 * 2 = 4

    assert_all_values(result, 4.0, 1e-6, "Batch norm pattern")


# ============================================================================
# Main test runner
# ============================================================================


fn main() raises:
    """Run integration tests part 2: dtype, multi-dimensional, and ML patterns."""
    print("Running ExTensor integration tests (part 2)...")

    # Multiple dtypes
    print("  Testing dtype operations...")
    test_same_dtype_consistency()
    test_int_dtype_operations()

    # Multi-dimensional
    print("  Testing multi-dimensional operations...")
    test_2d_elementwise_operations()
    test_3d_operations()

    # ML patterns
    print("  Testing ML-like patterns...")
    test_linear_transformation_pattern()
    test_gradient_descent_update_pattern()
    test_batch_normalization_pattern()

    print("Integration tests part 2 completed!")
