# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_initializers.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Constant initialization and dtype support tests (part 5 of 6).

Tests:
- Constant shape preservation
- Constant value correctness
- Constant zero initialization
- Constant negative value
- Constant ones and zeros
- Xavier uniform with float64 dtype
- Xavier normal with float16 dtype
"""

from tests.shared.conftest import (
    assert_almost_equal,
    assert_equal,
    assert_true,
)
from shared.core.extensor import ExTensor
from shared.core.initializers import (
    xavier_uniform,
    xavier_normal,
    constant,
)
from math import sqrt


# ============================================================================
# Helper Functions for Statistical Tests
# ============================================================================


fn compute_mean(tensor: ExTensor) -> Float64:
    """Compute mean of tensor values."""
    var sum = Float64(0.0)
    var size = tensor.numel()

    if tensor.dtype() == DType.float32:
        var ptr = tensor._data.bitcast[Float32]()
        for i in range(size):
            sum += Float64(ptr[i])
    elif tensor.dtype() == DType.float64:
        var ptr = tensor._data.bitcast[Float64]()
        for i in range(size):
            sum += ptr[i]
    elif tensor.dtype() == DType.float16:
        var ptr = tensor._data.bitcast[Float16]()
        for i in range(size):
            sum += Float64(ptr[i])

    return sum / Float64(size)


fn compute_variance(tensor: ExTensor, mean: Float64) -> Float64:
    """Compute variance of tensor values."""
    var sum_sq_diff = Float64(0.0)
    var size = tensor.numel()

    if tensor.dtype() == DType.float32:
        var ptr = tensor._data.bitcast[Float32]()
        for i in range(size):
            var diff = Float64(ptr[i]) - mean
            sum_sq_diff += diff * diff
    elif tensor.dtype() == DType.float64:
        var ptr = tensor._data.bitcast[Float64]()
        for i in range(size):
            var diff = ptr[i] - mean
            sum_sq_diff += diff * diff
    elif tensor.dtype() == DType.float16:
        var ptr = tensor._data.bitcast[Float16]()
        for i in range(size):
            var diff = Float64(ptr[i]) - mean
            sum_sq_diff += diff * diff

    return sum_sq_diff / Float64(size)


# ============================================================================
# Constant Initialization Tests
# ============================================================================


fn test_constant_shape() raises:
    """Test constant initialization preserves shape."""
    var shape = List[Int]()
    shape.append(50)
    shape.append(30)
    var W = constant(shape, 3.14, DType.float32)

    assert_equal(W.shape()[0], 50)
    assert_equal(W.shape()[1], 30)


fn test_constant_value() raises:
    """Test constant initialization sets all values correctly."""
    var shape = List[Int]()
    shape.append(10)
    shape.append(10)
    var value = 7.5
    var W = constant(shape, value, DType.float32)

    # Check all values are exactly the constant
    for i in range(100):
        assert_almost_equal(
            W._data.bitcast[Float32]()[i], Float32(value), tolerance=1e-5
        )


fn test_constant_zero() raises:
    """Test constant initialization with zero."""
    var shape = List[Int]()
    shape.append(5)
    shape.append(5)
    var W = constant(shape, 0.0, DType.float32)

    for i in range(25):
        assert_almost_equal(
            W._data.bitcast[Float32]()[i], Float32(0.0), tolerance=1e-10
        )


fn test_constant_negative() raises:
    """Test constant initialization with negative value."""
    var shape = List[Int]()
    shape.append(5)
    shape.append(5)
    var value = -2.5
    var W = constant(shape, value, DType.float32)

    for i in range(25):
        assert_almost_equal(
            W._data.bitcast[Float32]()[i], Float32(value), tolerance=1e-5
        )


fn test_constant_ones_and_zeros() raises:
    """Test constant can create ones and zeros."""
    var shape: List[Int] = [5, 5]

    # Test ones
    var ones_tensor = constant(shape, 1.0, DType.float32)
    for i in range(ones_tensor.numel()):
        var val = Float64(ones_tensor._data.bitcast[Float32]()[i])
        assert_equal(val, 1.0)

    # Test zeros
    var zeros_tensor = constant(shape, 0.0, DType.float32)
    for i in range(zeros_tensor.numel()):
        var val = Float64(zeros_tensor._data.bitcast[Float32]()[i])
        assert_equal(val, 0.0)


# ============================================================================
# Dtype Support Tests
# ============================================================================


fn test_xavier_uniform_float64() raises:
    """Test Xavier uniform with float64 dtype."""
    var fan_in = 100
    var fan_out = 50
    var shape: List[Int] = [fan_in, fan_out]
    var W = xavier_uniform(fan_in, fan_out, shape, DType.float64)

    assert_true(
        W._dtype == DType.float64, "Xavier uniform should have float64 dtype"
    )
    assert_equal(W.shape()[0], 100)
    assert_equal(W.shape()[1], 50)

    # Verify variance
    var expected_var = 2.0 / Float64(fan_in + fan_out)
    var mean = compute_mean(W)
    var actual_var = compute_variance(W, mean)
    var tolerance = expected_var * 0.1
    assert_true(abs(actual_var - expected_var) < tolerance)


fn test_xavier_normal_float16() raises:
    """Test Xavier normal with float16 dtype."""
    var fan_in = 100
    var fan_out = 50
    var shape: List[Int] = [fan_in, fan_out]
    var W = xavier_normal(fan_in, fan_out, shape, DType.float16, seed_val=42)

    assert_true(
        W._dtype == DType.float16, "Xavier normal should have float16 dtype"
    )

    # Check variance for float16 (with looser tolerance)
    var expected_var = 2.0 / Float64(fan_in + fan_out)
    var mean = compute_mean(W)
    var actual_var = compute_variance(W, mean)
    var tolerance = expected_var * 0.15  # Float16 has less precision
    assert_true(abs(actual_var - expected_var) < tolerance)


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run all tests in this file."""
    var total = 0
    var passed = 0
    var failed = 0

    print("\n" + "=" * 70)
    print("Running tests from: test_initializers_part5.mojo")
    print("=" * 70 + "\n")

    # test_constant_shape
    total += 1
    try:
        test_constant_shape()
        passed += 1
        print("  ✓ test_constant_shape")
    except e:
        failed += 1
        print("  ✗ test_constant_shape:", e)

    # test_constant_value
    total += 1
    try:
        test_constant_value()
        passed += 1
        print("  ✓ test_constant_value")
    except e:
        failed += 1
        print("  ✗ test_constant_value:", e)

    # test_constant_zero
    total += 1
    try:
        test_constant_zero()
        passed += 1
        print("  ✓ test_constant_zero")
    except e:
        failed += 1
        print("  ✗ test_constant_zero:", e)

    # test_constant_negative
    total += 1
    try:
        test_constant_negative()
        passed += 1
        print("  ✓ test_constant_negative")
    except e:
        failed += 1
        print("  ✗ test_constant_negative:", e)

    # test_constant_ones_and_zeros
    total += 1
    try:
        test_constant_ones_and_zeros()
        passed += 1
        print("  ✓ test_constant_ones_and_zeros")
    except e:
        failed += 1
        print("  ✗ test_constant_ones_and_zeros:", e)

    # test_xavier_uniform_float64
    total += 1
    try:
        test_xavier_uniform_float64()
        passed += 1
        print("  ✓ test_xavier_uniform_float64")
    except e:
        failed += 1
        print("  ✗ test_xavier_uniform_float64:", e)

    # test_xavier_normal_float16
    total += 1
    try:
        test_xavier_normal_float16()
        passed += 1
        print("  ✓ test_xavier_normal_float16")
    except e:
        failed += 1
        print("  ✗ test_xavier_normal_float16:", e)

    # Summary
    print("\n" + "=" * 70)
    print("Results:", passed, "/", total, "passed,", failed, "failed")
    print("=" * 70)

    if failed > 0:
        raise Error("Tests failed")
