# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_initializers.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Kaiming uniform/normal and uniform distribution tests (part 3 of 6).

Tests:
- Kaiming uniform variance (fan_out mode)
- Kaiming uniform reproducibility
- Kaiming normal shape
- Kaiming normal mean
- Kaiming normal std
- Kaiming normal reproducibility
- Uniform distribution shape
"""

from tests.shared.conftest import (
    assert_almost_equal,
    assert_equal,
    assert_true,
)
from shared.core.extensor import ExTensor
from shared.core.initializers import (
    kaiming_uniform,
    kaiming_normal,
    uniform,
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


fn compute_std(tensor: ExTensor, mean: Float64) -> Float64:
    """Compute standard deviation of tensor values."""
    return sqrt(compute_variance(tensor, mean))


# ============================================================================
# Kaiming Uniform Tests (continued)
# ============================================================================


fn test_kaiming_uniform_variance_fan_out() raises:
    """Test Kaiming uniform has correct variance with fan_out mode."""
    var fan_in = 100
    var fan_out = 50
    var shape: List[Int] = [fan_in, fan_out]
    var weights = kaiming_uniform(
        fan_in, fan_out, shape, "fan_out", DType.float32, seed_val=42
    )

    # Expected variance: 2/fan_out = 2/50 = 0.04
    var expected_var = 2.0 / Float64(fan_out)
    var mean = compute_mean(weights)
    var actual_var = compute_variance(weights, mean)

    var tolerance = expected_var * 0.1
    var diff = abs(actual_var - expected_var)

    assert_true(diff < tolerance)


fn test_kaiming_uniform_reproducibility() raises:
    """Test Kaiming uniform with fixed seed is reproducible."""
    var fan_in = 50
    var fan_out = 100
    var shape: List[Int] = [fan_in, fan_out]

    # Generate with same seed twice
    var w1 = kaiming_uniform(
        fan_in, fan_out, shape, "fan_in", DType.float32, seed_val=999
    )
    var w2 = kaiming_uniform(
        fan_in, fan_out, shape, "fan_in", DType.float32, seed_val=999
    )

    # Should be identical
    for i in range(w1.numel()):
        var val1 = w1._data.bitcast[Float32]()[i]
        var val2 = w2._data.bitcast[Float32]()[i]
        assert_equal(val1, val2)


# ============================================================================
# Kaiming Normal Tests
# ============================================================================


fn test_kaiming_normal_shape() raises:
    """Test Kaiming normal initialization preserves shape."""
    var fan_in = 100
    var fan_out = 50
    var shape: List[Int] = [fan_in, fan_out]
    var W = kaiming_normal(fan_in, fan_out, shape, "fan_in", DType.float32)

    assert_equal(W.shape()[0], fan_in)
    assert_equal(W.shape()[1], fan_out)


fn test_kaiming_normal_mean() raises:
    """Test Kaiming normal has approximately zero mean."""
    var fan_in = 1000
    var fan_out = 1000
    var shape: List[Int] = [fan_in, fan_out]
    var W = kaiming_normal(fan_in, fan_out, shape, "fan_in", DType.float32)

    var mean = compute_mean(W)

    assert_almost_equal(Float32(mean), Float32(0.0), tolerance=0.01)


fn test_kaiming_normal_std() raises:
    """Test Kaiming normal has approximately correct standard deviation."""
    var fan_in = 2000
    var fan_out = 2000
    var shape: List[Int] = [fan_in, fan_out]
    var W = kaiming_normal(fan_in, fan_out, shape, "fan_in", DType.float32)

    var mean = compute_mean(W)
    var std_dev = compute_std(W, mean)

    # Kaiming normal: std = sqrt(2 / fan_in)
    var expected_std = sqrt(2.0 / Float64(fan_in))

    # Allow 10% tolerance
    assert_almost_equal(
        Float32(std_dev),
        Float32(expected_std),
        tolerance=Float32(expected_std) * 0.1,
    )


fn test_kaiming_normal_reproducibility() raises:
    """Test Kaiming normal with fixed seed is reproducible."""
    var fan_in = 50
    var fan_out = 100
    var shape: List[Int] = [fan_in, fan_out]

    # Generate with same seed twice
    var w1 = kaiming_normal(
        fan_in, fan_out, shape, "fan_in", DType.float32, seed_val=555
    )
    var w2 = kaiming_normal(
        fan_in, fan_out, shape, "fan_in", DType.float32, seed_val=555
    )

    # Should be identical
    for i in range(w1.numel()):
        var val1 = w1._data.bitcast[Float32]()[i]
        var val2 = w2._data.bitcast[Float32]()[i]
        assert_equal(val1, val2)


# ============================================================================
# Uniform Distribution Tests (first test)
# ============================================================================


fn test_uniform_shape() raises:
    """Test uniform initialization with custom range."""
    var shape = List[Int]()
    shape.append(50)
    shape.append(30)
    var W = uniform(shape, -0.5, 0.5, DType.float32)

    assert_equal(W.shape()[0], 50)
    assert_equal(W.shape()[1], 30)


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run all tests in this file."""
    var total = 0
    var passed = 0
    var failed = 0

    print("\n" + "=" * 70)
    print("Running tests from: test_initializers_part3.mojo")
    print("=" * 70 + "\n")

    # test_kaiming_uniform_variance_fan_out
    total += 1
    try:
        test_kaiming_uniform_variance_fan_out()
        passed += 1
        print("  ✓ test_kaiming_uniform_variance_fan_out")
    except e:
        failed += 1
        print("  ✗ test_kaiming_uniform_variance_fan_out:", e)

    # test_kaiming_uniform_reproducibility
    total += 1
    try:
        test_kaiming_uniform_reproducibility()
        passed += 1
        print("  ✓ test_kaiming_uniform_reproducibility")
    except e:
        failed += 1
        print("  ✗ test_kaiming_uniform_reproducibility:", e)

    # test_kaiming_normal_shape
    total += 1
    try:
        test_kaiming_normal_shape()
        passed += 1
        print("  ✓ test_kaiming_normal_shape")
    except e:
        failed += 1
        print("  ✗ test_kaiming_normal_shape:", e)

    # test_kaiming_normal_mean
    total += 1
    try:
        test_kaiming_normal_mean()
        passed += 1
        print("  ✓ test_kaiming_normal_mean")
    except e:
        failed += 1
        print("  ✗ test_kaiming_normal_mean:", e)

    # test_kaiming_normal_std
    total += 1
    try:
        test_kaiming_normal_std()
        passed += 1
        print("  ✓ test_kaiming_normal_std")
    except e:
        failed += 1
        print("  ✗ test_kaiming_normal_std:", e)

    # test_kaiming_normal_reproducibility
    total += 1
    try:
        test_kaiming_normal_reproducibility()
        passed += 1
        print("  ✓ test_kaiming_normal_reproducibility")
    except e:
        failed += 1
        print("  ✗ test_kaiming_normal_reproducibility:", e)

    # test_uniform_shape
    total += 1
    try:
        test_uniform_shape()
        passed += 1
        print("  ✓ test_uniform_shape")
    except e:
        failed += 1
        print("  ✗ test_uniform_shape:", e)

    # Summary
    print("\n" + "=" * 70)
    print("Results:", passed, "/", total, "passed,", failed, "failed")
    print("=" * 70)

    if failed > 0:
        raise Error("Tests failed")
