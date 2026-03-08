# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_initializers.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Dtype support and edge case tests (part 6 of 6).

Tests:
- Kaiming normal with float64 dtype
- Uniform with float64 dtype
- Normal with float64 dtype
- Constant with float64 dtype
- Small dimension initialization
- Rectangular matrix initialization
- Large matrix initialization
"""

from tests.shared.conftest import (
    assert_almost_equal,
    assert_equal,
    assert_true,
)
from shared.core.extensor import ExTensor
from shared.core.initializers import (
    xavier_uniform,
    kaiming_uniform,
    kaiming_normal,
    uniform,
    normal,
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
# Dtype Support Tests (continued)
# ============================================================================


fn test_kaiming_normal_float64() raises:
    """Test Kaiming normal with float64 dtype."""
    var fan_in = 100
    var fan_out = 50
    var shape: List[Int] = [fan_in, fan_out]
    var W = kaiming_normal(fan_in, fan_out, shape, "fan_in", DType.float64)

    assert_true(
        W._dtype == DType.float64, "Kaiming normal should have float64 dtype"
    )
    assert_equal(W.shape()[0], 100)
    assert_equal(W.shape()[1], 50)

    # Check variance for float64
    var expected_var = 2.0 / Float64(fan_in)
    var mean = compute_mean(W)
    var actual_var = compute_variance(W, mean)
    var tolerance = expected_var * 0.1
    assert_true(abs(actual_var - expected_var) < tolerance)


fn test_uniform_float64() raises:
    """Test uniform with float64 dtype."""
    var shape: List[Int] = [50, 50]
    var weights = uniform(shape, -1.0, 1.0, DType.float64, seed_val=42)

    assert_true(
        weights._dtype == DType.float64, "Uniform should have float64 dtype"
    )

    # Check bounds
    for i in range(weights.numel()):
        var val = weights._data.bitcast[Float64]()[i]
        assert_true(val >= -1.0 and val <= 1.0)


fn test_normal_float64() raises:
    """Test normal with float64 dtype."""
    var shape: List[Int] = [50, 50]
    var weights = normal(shape, 0.0, 0.1, DType.float64, seed_val=42)

    assert_true(
        weights._dtype == DType.float64, "Normal should have float64 dtype"
    )


fn test_constant_float64() raises:
    """Test constant initialization with float64 dtype."""
    var shape = List[Int]()
    shape.append(10)
    shape.append(10)
    var W = constant(shape, 1.5, DType.float64)

    assert_true(W._dtype == DType.float64, "Constant should have float64 dtype")

    for i in range(100):
        assert_almost_equal(
            Float32(W._data.bitcast[Float64]()[i]), Float32(1.5), tolerance=1e-5
        )


# ============================================================================
# Edge Case Tests
# ============================================================================


fn test_small_dimensions() raises:
    """Test initialization with small dimensions."""
    # Xavier with fan_in=1, fan_out=1
    var shape1: List[Int] = [1, 1]
    var W1 = xavier_uniform(1, 1, shape1, DType.float32)
    assert_equal(W1.numel(), 1)

    # Kaiming with fan_in=2
    var shape2: List[Int] = [2, 3]
    var W2 = kaiming_uniform(2, 3, shape2, "fan_in", DType.float32)
    assert_equal(W2.shape()[0], 2)
    assert_equal(W2.shape()[1], 3)


fn test_rectangular_matrices() raises:
    """Test initialization with non-square matrices."""
    # Tall matrix (more rows than columns)
    var shape_tall: List[Int] = [1000, 10]
    var W_tall = xavier_uniform(1000, 10, shape_tall, DType.float32)
    assert_equal(W_tall.shape()[0], 1000)
    assert_equal(W_tall.shape()[1], 10)

    # Wide matrix (more columns than rows)
    var shape_wide: List[Int] = [10, 1000]
    var W_wide = kaiming_uniform(10, 1000, shape_wide, "fan_in", DType.float32)
    assert_equal(W_wide.shape()[0], 10)
    assert_equal(W_wide.shape()[1], 1000)


fn test_large_initialization() raises:
    """Test initialization with large dimensions."""
    # Large matrix
    var fan_in = 5000
    var fan_out = 5000
    var shape: List[Int] = [fan_in, fan_out]
    var W = xavier_uniform(fan_in, fan_out, shape, DType.float32)

    assert_equal(W.shape()[0], 5000)
    assert_equal(W.shape()[1], 5000)
    assert_equal(W.numel(), 25000000)

    # Verify statistical properties still hold
    var mean = compute_mean(W)
    assert_almost_equal(Float32(mean), Float32(0.0), tolerance=0.01)


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run all tests in this file."""
    var total = 0
    var passed = 0
    var failed = 0

    print("\n" + "=" * 70)
    print("Running tests from: test_initializers_part6.mojo")
    print("=" * 70 + "\n")

    # test_kaiming_normal_float64
    total += 1
    try:
        test_kaiming_normal_float64()
        passed += 1
        print("  ✓ test_kaiming_normal_float64")
    except e:
        failed += 1
        print("  ✗ test_kaiming_normal_float64:", e)

    # test_uniform_float64
    total += 1
    try:
        test_uniform_float64()
        passed += 1
        print("  ✓ test_uniform_float64")
    except e:
        failed += 1
        print("  ✗ test_uniform_float64:", e)

    # test_normal_float64
    total += 1
    try:
        test_normal_float64()
        passed += 1
        print("  ✓ test_normal_float64")
    except e:
        failed += 1
        print("  ✗ test_normal_float64:", e)

    # test_constant_float64
    total += 1
    try:
        test_constant_float64()
        passed += 1
        print("  ✓ test_constant_float64")
    except e:
        failed += 1
        print("  ✗ test_constant_float64:", e)

    # test_small_dimensions
    total += 1
    try:
        test_small_dimensions()
        passed += 1
        print("  ✓ test_small_dimensions")
    except e:
        failed += 1
        print("  ✗ test_small_dimensions:", e)

    # test_rectangular_matrices
    total += 1
    try:
        test_rectangular_matrices()
        passed += 1
        print("  ✓ test_rectangular_matrices")
    except e:
        failed += 1
        print("  ✗ test_rectangular_matrices:", e)

    # test_large_initialization
    total += 1
    try:
        test_large_initialization()
        passed += 1
        print("  ✓ test_large_initialization")
    except e:
        failed += 1
        print("  ✗ test_large_initialization:", e)

    # Summary
    print("\n" + "=" * 70)
    print("Results:", passed, "/", total, "passed,", failed, "failed")
    print("=" * 70)

    if failed > 0:
        raise Error("Tests failed")
