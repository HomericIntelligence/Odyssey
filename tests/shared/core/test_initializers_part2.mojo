# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_initializers.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Xavier normal and Kaiming uniform initialization tests (part 2 of 6).

Tests:
- Xavier normal std
- Xavier normal reproducibility
- Xavier configurations (various fan configs)
- Kaiming uniform shape
- Kaiming uniform range
- Kaiming uniform mean
- Kaiming uniform variance (fan_in mode)
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
    kaiming_uniform,
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
# Xavier Normal Tests (continued)
# ============================================================================


fn test_xavier_normal_std() raises:
    """Test Xavier normal has approximately correct standard deviation."""
    var fan_in = 2000
    var fan_out = 2000
    var shape: List[Int] = [fan_in, fan_out]
    var W = xavier_normal(fan_in, fan_out, shape, DType.float32)

    var mean = compute_mean(W)
    var std_dev = compute_std(W, mean)

    # Xavier normal: std = sqrt(2 / (fan_in + fan_out))
    var expected_std = sqrt(2.0 / Float64(fan_in + fan_out))

    # Allow 10% tolerance
    assert_almost_equal(
        Float32(std_dev),
        Float32(expected_std),
        tolerance=Float32(expected_std) * 0.1,
    )


fn test_xavier_normal_reproducibility() raises:
    """Test Xavier normal with fixed seed is reproducible."""
    var fan_in = 50
    var fan_out = 100
    var shape: List[Int] = [fan_in, fan_out]

    # Generate with same seed twice
    var w1 = xavier_normal(fan_in, fan_out, shape, DType.float32, seed_val=555)
    var w2 = xavier_normal(fan_in, fan_out, shape, DType.float32, seed_val=555)

    # Should be identical
    for i in range(w1.numel()):
        var val1 = w1._data.bitcast[Float32]()[i]
        var val2 = w2._data.bitcast[Float32]()[i]
        assert_equal(val1, val2)


fn test_xavier_configurations() raises:
    """Test Xavier initialization with various fan configurations."""
    # Test several configurations
    var configs = List[Tuple[Int, Int]]()
    configs.append((10, 10))  # Square
    configs.append((100, 50))  # Wide
    configs.append((50, 100))  # Tall
    configs.append((784, 128))  # Typical NN layer
    configs.append((1, 1000))  # Extreme aspect ratio

    for idx in range(len(configs)):
        var fan_in = configs[idx][0]
        var fan_out = configs[idx][1]
        var shape: List[Int] = [fan_in, fan_out]

        # Test uniform
        var w_uniform = xavier_uniform(
            fan_in, fan_out, shape, DType.float32, seed_val=42
        )
        var bound = sqrt(6.0 / Float64(fan_in + fan_out))

        # Check bounds
        for i in range(w_uniform.numel()):
            var val = Float64(w_uniform._data.bitcast[Float32]()[i])
            assert_true(val >= -bound and val <= bound)

        # Test normal
        var w_normal = xavier_normal(
            fan_in, fan_out, shape, DType.float32, seed_val=42
        )
        var expected_var = 2.0 / Float64(fan_in + fan_out)
        var mean = compute_mean(w_normal)
        var actual_var = compute_variance(w_normal, mean)

        # Variance should be reasonable (within 25% for smaller samples)
        var tolerance = expected_var * 0.25
        var diff = abs(actual_var - expected_var)
        assert_true(diff < tolerance)


# ============================================================================
# Kaiming Uniform Tests
# ============================================================================


fn test_kaiming_uniform_shape() raises:
    """Test Kaiming uniform initialization preserves shape."""
    var fan_in = 100
    var fan_out = 50
    var shape: List[Int] = [fan_in, fan_out]
    var W = kaiming_uniform(fan_in, fan_out, shape, "fan_in", DType.float32)

    assert_equal(W.shape()[0], fan_in)
    assert_equal(W.shape()[1], fan_out)


fn test_kaiming_uniform_range() raises:
    """Test Kaiming uniform values are within expected range."""
    var fan_in = 1000
    var fan_out = 500
    var shape: List[Int] = [fan_in, fan_out]
    var W = kaiming_uniform(fan_in, fan_out, shape, "fan_in", DType.float32)

    # Kaiming uniform limit: sqrt(6 / fan_in)
    var limit = sqrt(6.0 / Float64(fan_in))

    var result = compute_min_max(W)
    var min_val = result[0]
    var max_val = result[1]

    # All values should be in [-limit, limit]
    assert_true(min_val >= -limit - 0.01)
    assert_true(max_val <= limit + 0.01)


fn test_kaiming_uniform_mean() raises:
    """Test Kaiming uniform has approximately zero mean."""
    var fan_in = 1000
    var fan_out = 1000
    var shape: List[Int] = [fan_in, fan_out]
    var W = kaiming_uniform(fan_in, fan_out, shape, "fan_in", DType.float32)

    var mean = compute_mean(W)

    assert_almost_equal(Float32(mean), Float32(0.0), tolerance=0.01)


fn test_kaiming_uniform_variance_fan_in() raises:
    """Test Kaiming uniform has correct variance with fan_in mode."""
    var fan_in = 2000
    var fan_out = 2000
    var shape: List[Int] = [fan_in, fan_out]
    var W = kaiming_uniform(fan_in, fan_out, shape, "fan_in", DType.float32)

    var mean = compute_mean(W)
    var std_dev = compute_std(W, mean)

    # Expected std = sqrt(2 / fan_in)
    var expected_std = sqrt(2.0 / Float64(fan_in))

    # Allow 10% tolerance
    assert_almost_equal(
        Float32(std_dev),
        Float32(expected_std),
        tolerance=Float32(expected_std) * 0.1,
    )


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run all tests in this file."""
    var total = 0
    var passed = 0
    var failed = 0

    print("\n" + "=" * 70)
    print("Running tests from: test_initializers_part2.mojo")
    print("=" * 70 + "\n")

    # test_xavier_normal_std
    total += 1
    try:
        test_xavier_normal_std()
        passed += 1
        print("  ✓ test_xavier_normal_std")
    except e:
        failed += 1
        print("  ✗ test_xavier_normal_std:", e)

    # test_xavier_normal_reproducibility
    total += 1
    try:
        test_xavier_normal_reproducibility()
        passed += 1
        print("  ✓ test_xavier_normal_reproducibility")
    except e:
        failed += 1
        print("  ✗ test_xavier_normal_reproducibility:", e)

    # test_xavier_configurations
    total += 1
    try:
        test_xavier_configurations()
        passed += 1
        print("  ✓ test_xavier_configurations")
    except e:
        failed += 1
        print("  ✗ test_xavier_configurations:", e)

    # test_kaiming_uniform_shape
    total += 1
    try:
        test_kaiming_uniform_shape()
        passed += 1
        print("  ✓ test_kaiming_uniform_shape")
    except e:
        failed += 1
        print("  ✗ test_kaiming_uniform_shape:", e)

    # test_kaiming_uniform_range
    total += 1
    try:
        test_kaiming_uniform_range()
        passed += 1
        print("  ✓ test_kaiming_uniform_range")
    except e:
        failed += 1
        print("  ✗ test_kaiming_uniform_range:", e)

    # test_kaiming_uniform_mean
    total += 1
    try:
        test_kaiming_uniform_mean()
        passed += 1
        print("  ✓ test_kaiming_uniform_mean")
    except e:
        failed += 1
        print("  ✗ test_kaiming_uniform_mean:", e)

    # test_kaiming_uniform_variance_fan_in
    total += 1
    try:
        test_kaiming_uniform_variance_fan_in()
        passed += 1
        print("  ✓ test_kaiming_uniform_variance_fan_in")
    except e:
        failed += 1
        print("  ✗ test_kaiming_uniform_variance_fan_in:", e)

    # Summary
    print("\n" + "=" * 70)
    print("Results:", passed, "/", total, "passed,", failed, "failed")
    print("=" * 70)

    if failed > 0:
        raise Error("Tests failed")
