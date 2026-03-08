# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_initializers.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Uniform and normal distribution tests (part 4 of 6).

Tests:
- Uniform distribution range
- Uniform distribution mean
- Uniform distribution reproducibility
- Normal distribution shape
- Normal distribution mean
- Normal distribution std
- Normal distribution reproducibility
"""

from tests.shared.conftest import (
    assert_almost_equal,
    assert_equal,
    assert_true,
)
from shared.core.extensor import ExTensor
from shared.core.initializers import (
    uniform,
    normal,
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
# Uniform Distribution Tests
# ============================================================================


fn test_uniform_range() raises:
    """Test uniform values are within specified range."""
    var shape = List[Int]()
    shape.append(100)
    shape.append(100)
    var low = -0.7
    var high = 0.3
    var W = uniform(shape, low, high, DType.float32)

    var result = compute_min_max(W)
    var min_val = result[0]
    var max_val = result[1]

    # All values should be in [low, high]
    assert_true(min_val >= low - 1e-5)
    assert_true(max_val <= high + 1e-5)


fn test_uniform_mean() raises:
    """Test uniform distribution has approximately correct mean."""
    var shape = List[Int]()
    shape.append(200)
    shape.append(200)
    var low = -1.0
    var high = 1.0
    var W = uniform(shape, low, high, DType.float32)

    var mean = compute_mean(W)

    # Mean of uniform distribution U(a,b) is (a+b)/2
    var expected_mean = (low + high) / 2.0
    assert_almost_equal(Float32(mean), Float32(expected_mean), tolerance=0.05)


fn test_uniform_reproducibility() raises:
    """Test uniform with fixed seed is reproducible."""
    var shape: List[Int] = [50, 50]

    # Generate with same seed twice
    var w1 = uniform(shape, -0.2, 0.2, DType.float32, seed_val=999)
    var w2 = uniform(shape, -0.2, 0.2, DType.float32, seed_val=999)

    # Should be identical
    for i in range(w1.numel()):
        var val1 = w1._data.bitcast[Float32]()[i]
        var val2 = w2._data.bitcast[Float32]()[i]
        assert_equal(val1, val2)


# ============================================================================
# Normal Distribution Tests
# ============================================================================


fn test_normal_shape() raises:
    """Test normal initialization with custom parameters."""
    var shape = List[Int]()
    shape.append(50)
    shape.append(30)
    var W = normal(shape, 0.0, 1.0, DType.float32)

    assert_equal(W.shape()[0], 50)
    assert_equal(W.shape()[1], 30)


fn test_normal_mean() raises:
    """Test normal distribution has approximately correct mean."""
    var shape = List[Int]()
    shape.append(200)
    shape.append(200)
    var target_mean = 2.5
    var target_std = 0.5
    var W = normal(shape, target_mean, target_std, DType.float32)

    var actual_mean = compute_mean(W)

    # Allow 5% tolerance for sampling variability
    assert_almost_equal(
        Float32(actual_mean), Float32(target_mean), tolerance=0.1
    )


fn test_normal_std() raises:
    """Test normal distribution has approximately correct standard deviation."""
    var shape = List[Int]()
    shape.append(300)
    shape.append(300)
    var target_mean = 0.0
    var target_std = 1.5
    var W = normal(shape, target_mean, target_std, DType.float32)

    var actual_mean = compute_mean(W)
    var actual_std = compute_std(W, actual_mean)

    # Allow 10% tolerance for sampling variability
    assert_almost_equal(
        Float32(actual_std),
        Float32(target_std),
        tolerance=Float32(target_std) * 0.1,
    )


fn test_normal_reproducibility() raises:
    """Test normal with fixed seed is reproducible."""
    var shape: List[Int] = [50, 50]

    # Generate with same seed twice
    var w1 = normal(shape, 0.0, 0.05, DType.float32, seed_val=555)
    var w2 = normal(shape, 0.0, 0.05, DType.float32, seed_val=555)

    # Should be identical
    for i in range(w1.numel()):
        var val1 = w1._data.bitcast[Float32]()[i]
        var val2 = w2._data.bitcast[Float32]()[i]
        assert_equal(val1, val2)


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run all tests in this file."""
    var total = 0
    var passed = 0
    var failed = 0

    print("\n" + "=" * 70)
    print("Running tests from: test_initializers_part4.mojo")
    print("=" * 70 + "\n")

    # test_uniform_range
    total += 1
    try:
        test_uniform_range()
        passed += 1
        print("  ✓ test_uniform_range")
    except e:
        failed += 1
        print("  ✗ test_uniform_range:", e)

    # test_uniform_mean
    total += 1
    try:
        test_uniform_mean()
        passed += 1
        print("  ✓ test_uniform_mean")
    except e:
        failed += 1
        print("  ✗ test_uniform_mean:", e)

    # test_uniform_reproducibility
    total += 1
    try:
        test_uniform_reproducibility()
        passed += 1
        print("  ✓ test_uniform_reproducibility")
    except e:
        failed += 1
        print("  ✗ test_uniform_reproducibility:", e)

    # test_normal_shape
    total += 1
    try:
        test_normal_shape()
        passed += 1
        print("  ✓ test_normal_shape")
    except e:
        failed += 1
        print("  ✗ test_normal_shape:", e)

    # test_normal_mean
    total += 1
    try:
        test_normal_mean()
        passed += 1
        print("  ✓ test_normal_mean")
    except e:
        failed += 1
        print("  ✗ test_normal_mean:", e)

    # test_normal_std
    total += 1
    try:
        test_normal_std()
        passed += 1
        print("  ✓ test_normal_std")
    except e:
        failed += 1
        print("  ✗ test_normal_std:", e)

    # test_normal_reproducibility
    total += 1
    try:
        test_normal_reproducibility()
        passed += 1
        print("  ✓ test_normal_reproducibility")
    except e:
        failed += 1
        print("  ✗ test_normal_reproducibility:", e)

    # Summary
    print("\n" + "=" * 70)
    print("Results:", passed, "/", total, "passed,", failed, "failed")
    print("=" * 70)

    if failed > 0:
        raise Error("Tests failed")
