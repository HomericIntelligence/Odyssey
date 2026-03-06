"""Tests for ExTensor randn() method.

Note: Split from test_extensor_new_methods.mojo due to Mojo 0.26.1 heap
corruption bug that occurs after ~15 cumulative tests. See ADR-009.
"""

from shared.core.extensor import ExTensor, zeros, randn
from tests.shared.conftest import assert_true, assert_almost_equal, assert_equal
from math import sqrt


fn test_randn_basic_creation() raises:
    """Test randn() creates tensor with correct shape and dtype."""
    var tensor = randn([3, 4], DType.float32)
    var shape = tensor.shape()
    assert_equal(shape[0], 3)
    assert_equal(shape[1], 4)
    assert_true(tensor.dtype() == DType.float32)
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
    var nonzero_count = 0
    for i in range(tensor.numel()):
        var val = tensor._get_float32(i)
        if abs(val) > 1e-6:
            nonzero_count += 1
    assert_true(nonzero_count >= 90)


fn test_randn_distribution_properties() raises:
    """Test randn() produces values with approximately correct mean and std."""
    var tensor = randn([10000], DType.float32)
    var sum = Float64(0.0)
    for i in range(tensor.numel()):
        sum += Float64(tensor._get_float32(i))
    var mean = sum / Float64(tensor.numel())

    var sum_squared_diff = Float64(0.0)
    for i in range(tensor.numel()):
        var val = Float64(tensor._get_float32(i))
        var diff = val - mean
        sum_squared_diff += diff * diff
    var variance = sum_squared_diff / Float64(tensor.numel())
    var std = sqrt(variance)

    print("Mean:", mean, "Std:", std)
    assert_almost_equal(mean, 0.0, tolerance=0.1)
    assert_almost_equal(std, 1.0, tolerance=0.1)


fn test_randn_different_shapes() raises:
    """Test randn() works with various tensor shapes."""
    var tensor_2d = randn([5, 10], DType.float32)
    assert_equal(tensor_2d.numel(), 50)

    var tensor_3d = randn([2, 3, 4], DType.float32)
    assert_equal(tensor_3d.numel(), 24)

    var tensor_4d = randn([8, 3, 28, 28], DType.float32)
    assert_equal(tensor_4d.numel(), 18816)


fn test_randn_different_dtypes() raises:
    """Test randn() works with different floating-point dtypes."""
    var tensor_f16 = randn([10], DType.float16)
    assert_true(tensor_f16.dtype() == DType.float16)

    var tensor_f32 = randn([10], DType.float32)
    assert_true(tensor_f32.dtype() == DType.float32)

    var tensor_f64 = randn([10], DType.float64)
    assert_true(tensor_f64.dtype() == DType.float64)


fn test_randn_small_tensor() raises:
    """Test randn() works for very small tensors (edge case)."""
    var tensor_1 = randn([1], DType.float32)
    assert_equal(tensor_1.numel(), 1)

    var tensor_2 = randn([2], DType.float32)
    assert_equal(tensor_2.numel(), 2)


fn test_integration_simplemlp_get_weights() raises:
    """Test that weights can be set/read via _set_float32."""
    var weights_tensor = zeros([100], DType.float32)
    for i in range(100):
        weights_tensor._set_float32(i, Float32(i) * 0.01)
    for i in range(100):
        var expected = Float32(i) * 0.01
        var actual = weights_tensor._get_float32(i)
        assert_almost_equal(Float64(actual), Float64(expected), tolerance=1e-6)


fn test_integration_randn_initialization() raises:
    """Test randn() for neural network weight initialization."""
    var layer_weights = randn([64, 128], DType.float32)
    assert_equal(layer_weights.numel(), 64 * 128)
    var w_0_0 = layer_weights._get_float32(0)
    var w_last = layer_weights._get_float32(layer_weights.numel() - 1)
    assert_true(abs(w_0_0) > 1e-10 or abs(w_last) > 1e-10)


fn main() raises:
    """Run randn and integration tests."""
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

    print("All randn tests passed!")
