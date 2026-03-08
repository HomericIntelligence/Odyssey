# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_initializers.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Xavier uniform initialization tests (part 1 of 6).

Tests:
- Xavier uniform shape preservation
- Xavier uniform value range
- Xavier uniform mean
- Xavier uniform variance
- Xavier uniform reproducibility
- Xavier uniform different seeds
- Xavier normal shape
- Xavier normal mean
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


fn compute_min_max(tensor: ExTensor) -> Tuple[Float64, Float64]:
    """Compute min and max values in tensor."""
    var size = tensor.numel()
    var min_val = Float64(1e308)
    var max_val = Float64(-1e308)

    if tensor.dtype() == DType.float32:
        var ptr = tensor._data.bitcast[Float32]()
        for i in range(size):
            var val = Float64(ptr[i])
            if val < min_val:
                min_val = val
            if val > max_val:
                max_val = val
    elif tensor.dtype() == DType.float64:
        var ptr = tensor._data.bitcast[Float64]()
        for i in range(size):
            var val = ptr[i]
            if val < min_val:
                min_val = val
            if val > max_val:
                max_val = val
    elif tensor.dtype() == DType.float16:
        var ptr = tensor._data.bitcast[Float16]()
        for i in range(size):
            var val = Float64(ptr[i])
            if val < min_val:
                min_val = val
            if val > max_val:
                max_val = val

    return (min_val, max_val)


# ============================================================================
# Xavier Uniform Tests
# ============================================================================


fn test_xavier_uniform_shape() raises:
    """Test Xavier uniform initialization preserves shape."""
    var fan_in = 100
    var fan_out = 50
    var shape: List[Int] = [fan_in, fan_out]
    var W = xavier_uniform(fan_in, fan_out, shape, DType.float32)

    assert_equal(W.shape()[0], fan_in)
    assert_equal(W.shape()[1], fan_out)
    assert_equal(W.numel(), fan_in * fan_out)


fn test_xavier_uniform_range() raises:
    """Test Xavier uniform values are within expected range."""
    var fan_in = 1000
    var fan_out = 500
    var shape: List[Int] = [fan_in, fan_out]
    var W = xavier_uniform(fan_in, fan_out, shape, DType.float32)

    # Xavier uniform limit: sqrt(6 / (fan_in + fan_out))
    var limit = sqrt(6.0 / Float64(fan_in + fan_out))

    var result = compute_min_max(W)
    var min_val = result[0]
    var max_val = result[1]

    # All values should be approximately in [-limit, limit]
    # Allow small tolerance for floating point errors
    assert_true(min_val >= -limit - 0.01)
    assert_true(max_val <= limit + 0.01)


fn test_xavier_uniform_mean() raises:
    """Test Xavier uniform has approximately zero mean."""
    var fan_in = 1000
    var fan_out = 1000
    var shape: List[Int] = [fan_in, fan_out]
    var W = xavier_uniform(fan_in, fan_out, shape, DType.float32)

    var mean = compute_mean(W)

    # Mean should be close to 0 (within tolerance for random sampling)
    assert_almost_equal(Float32(mean), Float32(0.0), tolerance=0.01)


fn test_xavier_uniform_variance() raises:
    """Test Xavier uniform has approximately correct variance."""
    var fan_in = 2000
    var fan_out = 2000
    var shape: List[Int] = [fan_in, fan_out]
    var W = xavier_uniform(fan_in, fan_out, shape, DType.float32)

    var mean = compute_mean(W)
    var std_dev = compute_std(W, mean)

    # For uniform distribution U(-a, a): variance = a²/3
    # Xavier limit: a = sqrt(6 / (fan_in + fan_out))
    # Expected std = a / sqrt(3) = sqrt(6 / (fan_in + fan_out)) / sqrt(3)
    #                             = sqrt(2 / (fan_in + fan_out))
    var expected_std = sqrt(2.0 / Float64(fan_in + fan_out))

    # Allow 10% tolerance for statistical variation
    assert_almost_equal(
        Float32(std_dev),
        Float32(expected_std),
        tolerance=Float32(expected_std) * 0.1,
    )


fn test_xavier_uniform_reproducibility() raises:
    """Test Xavier uniform with fixed seed is reproducible."""
    var fan_in = 50
    var fan_out = 100
    var shape: List[Int] = [fan_in, fan_out]

    # Generate with same seed twice
    var w1 = xavier_uniform(fan_in, fan_out, shape, DType.float32, seed_val=999)
    var w2 = xavier_uniform(fan_in, fan_out, shape, DType.float32, seed_val=999)

    # Should be identical
    for i in range(w1.numel()):
        var val1 = w1._data.bitcast[Float32]()[i]
        var val2 = w2._data.bitcast[Float32]()[i]
        assert_equal(val1, val2)


fn test_xavier_uniform_different_seeds() raises:
    """Test Xavier uniform with different seeds produces different results."""
    var fan_in = 50
    var fan_out = 100
    var shape: List[Int] = [fan_in, fan_out]

    # Generate with different seeds
    var w1 = xavier_uniform(fan_in, fan_out, shape, DType.float32, seed_val=111)
    var w2 = xavier_uniform(fan_in, fan_out, shape, DType.float32, seed_val=222)

    # Should be different (at least some values)
    var differences = 0
    for i in range(w1.numel()):
        var val1 = w1._data.bitcast[Float32]()[i]
        var val2 = w2._data.bitcast[Float32]()[i]
        if val1 != val2:
            differences += 1

    # Expect most values to be different (allow some coincidental matches)
    assert_true(differences > w1.numel() // 2)


# ============================================================================
# Xavier Normal Tests (first 2)
# ============================================================================


fn test_xavier_normal_shape() raises:
    """Test Xavier normal initialization preserves shape."""
    var fan_in = 100
    var fan_out = 50
    var shape: List[Int] = [fan_in, fan_out]
    var W = xavier_normal(fan_in, fan_out, shape, DType.float32)

    assert_equal(W.shape()[0], fan_in)
    assert_equal(W.shape()[1], fan_out)


fn test_xavier_normal_mean() raises:
    """Test Xavier normal has approximately zero mean."""
    var fan_in = 1000
    var fan_out = 1000
    var shape: List[Int] = [fan_in, fan_out]
    var W = xavier_normal(fan_in, fan_out, shape, DType.float32)

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
    print("Running tests from: test_initializers_part1.mojo")
    print("=" * 70 + "\n")

    # test_xavier_uniform_shape
    total += 1
    try:
        test_xavier_uniform_shape()
        passed += 1
        print("  ✓ test_xavier_uniform_shape")
    except e:
        failed += 1
        print("  ✗ test_xavier_uniform_shape:", e)

    # test_xavier_uniform_range
    total += 1
    try:
        test_xavier_uniform_range()
        passed += 1
        print("  ✓ test_xavier_uniform_range")
    except e:
        failed += 1
        print("  ✗ test_xavier_uniform_range:", e)

    # test_xavier_uniform_mean
    total += 1
    try:
        test_xavier_uniform_mean()
        passed += 1
        print("  ✓ test_xavier_uniform_mean")
    except e:
        failed += 1
        print("  ✗ test_xavier_uniform_mean:", e)

    # test_xavier_uniform_variance
    total += 1
    try:
        test_xavier_uniform_variance()
        passed += 1
        print("  ✓ test_xavier_uniform_variance")
    except e:
        failed += 1
        print("  ✗ test_xavier_uniform_variance:", e)

    # test_xavier_uniform_reproducibility
    total += 1
    try:
        test_xavier_uniform_reproducibility()
        passed += 1
        print("  ✓ test_xavier_uniform_reproducibility")
    except e:
        failed += 1
        print("  ✗ test_xavier_uniform_reproducibility:", e)

    # test_xavier_uniform_different_seeds
    total += 1
    try:
        test_xavier_uniform_different_seeds()
        passed += 1
        print("  ✓ test_xavier_uniform_different_seeds")
    except e:
        failed += 1
        print("  ✗ test_xavier_uniform_different_seeds:", e)

    # test_xavier_normal_shape
    total += 1
    try:
        test_xavier_normal_shape()
        passed += 1
        print("  ✓ test_xavier_normal_shape")
    except e:
        failed += 1
        print("  ✗ test_xavier_normal_shape:", e)

    # test_xavier_normal_mean
    total += 1
    try:
        test_xavier_normal_mean()
        passed += 1
        print("  ✓ test_xavier_normal_mean")
    except e:
        failed += 1
        print("  ✗ test_xavier_normal_mean:", e)

    # Summary
    print("\n" + "=" * 70)
    print("Results:", passed, "/", total, "passed,", failed, "failed")
    print("=" * 70)

    if failed > 0:
        raise Error("Tests failed")
