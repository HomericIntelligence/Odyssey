"""Unit tests for new ExTensor methods (_set_float32, _get_float32, randn).

Tests cover:
- _get_float32() - Get Float32 values from tensor
- _set_float32() - Set Float32 values in tensor
- randn() - Random normal distribution tensor creation

Following TDD principles - these tests verify the Track 1 API extensions.
"""

from shared.core.extensor import ExTensor, zeros, ones, randn
from shared.core import zeros as core_zeros
from tests.shared.conftest import assert_true, assert_almost_equal, assert_equal
from math import sqrt


# ============================================================================
# _get_float32 Tests
# ============================================================================


fn test_get_float32_basic() raises:
    """Test _get_float32() returns correct values for Float32 tensor."""
    var tensor = zeros([3, 4], DType.float32)

    # Set some test values using _set_float64
    tensor._set_float64(0, 1.5)
    tensor._set_float64(5, 2.7)
    tensor._set_float64(11, 3.9)

    # Get values using _get_float32
    var val0 = tensor._get_float32(0)
    var val5 = tensor._get_float32(5)
    var val11 = tensor._get_float32(11)

    # Verify values match (within Float32 precision)
    assert_almost_equal(Float64(val0), 1.5, tolerance=1e-6)
    assert_almost_equal(Float64(val5), 2.7, tolerance=1e-6)
    assert_almost_equal(Float64(val11), 3.9, tolerance=1e-6)


fn test_get_float32_dtype_conversions() raises:
    """Test _get_float32() handles different dtypes correctly."""
    # Test Float16 -> Float32
    var tensor_f16 = zeros([5], DType.float16)
    tensor_f16._set_float64(2, 1.5)
    var val_f16 = tensor_f16._get_float32(2)
    assert_almost_equal(
        Float64(val_f16), 1.5, tolerance=1e-3
    )  # Lower precision for Float16

    # Test Float64 -> Float32
    var tensor_f64 = zeros([5], DType.float64)
    tensor_f64._set_float64(2, 1.5)
    var val_f64 = tensor_f64._get_float32(2)
    assert_almost_equal(Float64(val_f64), 1.5, tolerance=1e-6)


# ============================================================================
# _set_float32 Tests
# ============================================================================


fn test_set_float32_basic() raises:
    """Test _set_float32() stores values correctly in Float32 tensor."""
    var tensor = zeros([3, 4], DType.float32)

    # Set values using _set_float32
    tensor._set_float32(0, Float32(1.5))
    tensor._set_float32(5, Float32(2.7))
    tensor._set_float32(11, Float32(3.9))

    # Verify values using _get_float64
    assert_almost_equal(tensor._get_float64(0), 1.5, tolerance=1e-6)
    assert_almost_equal(tensor._get_float64(5), 2.7, tolerance=1e-6)
    assert_almost_equal(tensor._get_float64(11), 3.9, tolerance=1e-6)


fn test_set_float32_all_elements() raises:
    """Test _set_float32() works for all elements in tensor."""
    var tensor = zeros([10], DType.float32)

    # Set all elements
    for i in range(10):
        tensor._set_float32(i, Float32(i) * 1.5)

    # Verify all elements
    for i in range(10):
        var expected = Float32(i) * 1.5
        var actual = tensor._get_float32(i)
        assert_almost_equal(Float64(actual), Float64(expected), tolerance=1e-6)


fn test_set_float32_dtype_conversions() raises:
    """Test _set_float32() handles different dtypes correctly."""
    # Test Float16 (downcast from Float32)
    var tensor_f16 = zeros([5], DType.float16)
    tensor_f16._set_float32(2, Float32(1.5))
    var val_f16 = tensor_f16._get_float64(2)
    assert_almost_equal(
        val_f16, 1.5, tolerance=1e-3
    )  # Lower precision for Float16

    # Test Float64 (upcast from Float32)
    var tensor_f64 = zeros([5], DType.float64)
    tensor_f64._set_float32(2, Float32(1.5))
    var val_f64 = tensor_f64._get_float64(2)
    assert_almost_equal(val_f64, 1.5, tolerance=1e-6)


fn test_set_get_float32_roundtrip() raises:
    """Test _set_float32() -> _get_float32() roundtrip preserves values."""
    var tensor = zeros([20], DType.float32)

    # Set values using _set_float32
    for i in range(20):
        tensor._set_float32(i, Float32(i) * 0.5)

    # Get values using _get_float32 and verify
    for i in range(20):
        var expected = Float32(i) * 0.5
        var actual = tensor._get_float32(i)
        assert_almost_equal(Float64(actual), Float64(expected), tolerance=1e-6)


# ============================================================================
# randn() Tests
# ============================================================================


fn test_randn_basic_creation() raises:
    """Test randn() creates tensor with correct shape and dtype."""
    var tensor = randn([3, 4], DType.float32)

    # Verify shape
    var shape = tensor.shape()
    assert_equal(shape[0], 3)
    assert_equal(shape[1], 4)

    # Verify dtype
    assert_true(tensor.dtype() == DType.float32)

    # Verify numel
    assert_equal(tensor.numel(), 12)


fn test_randn_1d_tensor() raises:
    """Test randn() works for 1D tensors."""
    var tensor = randn([100], DType.float32)

    assert_equal(tensor.numel(), 100)
    assert_equal(len(tensor.shape()), 1)
    assert_equal(tensor.shape()[0], 100)


fn test_randn_values_nonzero() raises:
    """Test randn() produces non-zero values (stochastic test)."""
    var tensor = randn([100], DType.float32)

    # Count non-zero values (should be most/all for random normal distribution)
    var nonzero_count = 0
    for i in range(tensor.numel()):
        var val = tensor._get_float32(i)
        if abs(val) > 1e-6:
            nonzero_count += 1

    # At least 90% of values should be non-zero for random distribution
    assert_true(nonzero_count >= 90)


fn test_randn_distribution_properties() raises:
    """Test randn() produces values with approximately correct mean and std.

    This is a stochastic test that may occasionally fail due to randomness,
    but should pass most of the time for large sample sizes.
    """
    var tensor = randn([10000], DType.float32)

    # Calculate mean
    var sum = Float64(0.0)
    for i in range(tensor.numel()):
        sum += Float64(tensor._get_float32(i))
    var mean = sum / Float64(tensor.numel())

    # Calculate standard deviation
    var sum_squared_diff = Float64(0.0)
    for i in range(tensor.numel()):
        var val = Float64(tensor._get_float32(i))
        var diff = val - mean
        sum_squared_diff += diff * diff
    var variance = sum_squared_diff / Float64(tensor.numel())
    var std = sqrt(variance)

    # For N(0, 1):
    # Mean should be close to 0 (within ±0.05 for 10000 samples)
    # Std should be close to 1 (within ±0.05 for 10000 samples)
    print("Mean:", mean, "Std:", std)
    assert_almost_equal(mean, 0.0, tolerance=0.1)
    assert_almost_equal(std, 1.0, tolerance=0.1)


fn test_randn_different_shapes() raises:
    """Test randn() works with various tensor shapes."""
    # 2D
    var tensor_2d = randn([5, 10], DType.float32)
    assert_equal(tensor_2d.numel(), 50)

    # 3D
    var tensor_3d = randn([2, 3, 4], DType.float32)
    assert_equal(tensor_3d.numel(), 24)

    # 4D (batch of images)
    var tensor_4d = randn([8, 3, 28, 28], DType.float32)
    assert_equal(tensor_4d.numel(), 18816)


fn test_randn_different_dtypes() raises:
    """Test randn() works with different floating-point dtypes."""
    # Float16
    var tensor_f16 = randn([10], DType.float16)
    assert_true(tensor_f16.dtype() == DType.float16)

    # Float32
    var tensor_f32 = randn([10], DType.float32)
    assert_true(tensor_f32.dtype() == DType.float32)

    # Float64
    var tensor_f64 = randn([10], DType.float64)
    assert_true(tensor_f64.dtype() == DType.float64)


fn test_randn_small_tensor() raises:
    """Test randn() works for very small tensors (edge case)."""
    # Single element
    var tensor_1 = randn([1], DType.float32)
    assert_equal(tensor_1.numel(), 1)

    # Two elements
    var tensor_2 = randn([2], DType.float32)
    assert_equal(tensor_2.numel(), 2)


# ============================================================================
# Integration Tests (combining new methods with existing functionality)
# ============================================================================


fn test_integration_simplemlp_get_weights() raises:
    """Test that SimpleMLP.get_weights() can use _set_float32().

    This simulates the use case from Issue #34 where SimpleMLP.get_weights()
    needs to flatten weights into a tensor using _set_float32().
    """
    # Create tensor to hold flattened weights
    var weights_tensor = zeros([100], DType.float32)

    # Simulate setting weights using _set_float32
    for i in range(100):
        weights_tensor._set_float32(i, Float32(i) * 0.01)

    # Verify values can be retrieved
    for i in range(100):
        var expected = Float32(i) * 0.01
        var actual = weights_tensor._get_float32(i)
        assert_almost_equal(Float64(actual), Float64(expected), tolerance=1e-6)


fn test_integration_randn_initialization() raises:
    """Test randn() for neural network weight initialization.

    This tests the common use case of initializing weights with random values.
    """
    # Initialize "layer" weights with Xavier/Glorot initialization pattern
    # (though randn() is just N(0,1), real Xavier would scale by sqrt(fan_in))
    var layer_weights = randn([64, 128], DType.float32)

    # Verify shape is correct
    assert_equal(layer_weights.numel(), 64 * 128)

    # Verify we can access individual weights
    var w_0_0 = layer_weights._get_float32(0)
    var w_last = layer_weights._get_float32(layer_weights.numel() - 1)

    # Both should be non-zero (extremely unlikely to be exactly zero)
    assert_true(abs(w_0_0) > 1e-10 or abs(w_last) > 1e-10)


# ============================================================================
# Test Main
# ============================================================================


fn main() raises:
    """Run all new ExTensor method tests."""
    print("Running _get_float32 tests...")
    test_get_float32_basic()
    test_get_float32_dtype_conversions()

    print("Running _set_float32 tests...")
    test_set_float32_basic()
    test_set_float32_all_elements()
    test_set_float32_dtype_conversions()
    test_set_get_float32_roundtrip()

    print("Running randn() tests...")
    test_randn_basic_creation()
    test_randn_1d_tensor()
    test_randn_values_nonzero()
    test_randn_distribution_properties()
    test_randn_different_shapes()
    test_randn_different_dtypes()
    test_randn_small_tensor()

    print("Running integration tests...")
    test_integration_simplemlp_get_weights()
    test_integration_randn_initialization()

    print("\nAll ExTensor new method tests passed! ✓")
