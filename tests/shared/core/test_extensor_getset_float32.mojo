"""Tests for ExTensor _get_float32 and _set_float32 methods.

Note: Split from test_extensor_new_methods.mojo due to Mojo 0.26.1 heap
corruption bug that occurs after ~15 cumulative tests. See ADR-009.
"""

from shared.core.extensor import ExTensor, zeros, ones
from tests.shared.conftest import assert_true, assert_almost_equal, assert_equal


fn test_get_float32_basic() raises:
    """Test _get_float32() returns correct values for Float32 tensor."""
    var tensor = zeros([3, 4], DType.float32)
    tensor._set_float64(0, 1.5)
    tensor._set_float64(5, 2.7)
    tensor._set_float64(11, 3.9)
    var val0 = tensor._get_float32(0)
    var val5 = tensor._get_float32(5)
    var val11 = tensor._get_float32(11)
    assert_almost_equal(Float64(val0), 1.5, tolerance=1e-6)
    assert_almost_equal(Float64(val5), 2.7, tolerance=1e-6)
    assert_almost_equal(Float64(val11), 3.9, tolerance=1e-6)


fn test_get_float32_dtype_conversions() raises:
    """Test _get_float32() handles different dtypes correctly."""
    var tensor_f16 = zeros([5], DType.float16)
    tensor_f16._set_float64(2, 1.5)
    var val_f16 = tensor_f16._get_float32(2)
    assert_almost_equal(Float64(val_f16), 1.5, tolerance=1e-3)

    var tensor_f64 = zeros([5], DType.float64)
    tensor_f64._set_float64(2, 1.5)
    var val_f64 = tensor_f64._get_float32(2)
    assert_almost_equal(Float64(val_f64), 1.5, tolerance=1e-6)


fn test_set_float32_basic() raises:
    """Test _set_float32() stores values correctly in Float32 tensor."""
    var tensor = zeros([3, 4], DType.float32)
    tensor._set_float32(0, Float32(1.5))
    tensor._set_float32(5, Float32(2.7))
    tensor._set_float32(11, Float32(3.9))
    assert_almost_equal(tensor._get_float64(0), 1.5, tolerance=1e-6)
    assert_almost_equal(tensor._get_float64(5), 2.7, tolerance=1e-6)
    assert_almost_equal(tensor._get_float64(11), 3.9, tolerance=1e-6)


fn test_set_float32_all_elements() raises:
    """Test _set_float32() works for all elements in tensor."""
    var tensor = zeros([10], DType.float32)
    for i in range(10):
        tensor._set_float32(i, Float32(i) * 1.5)
    for i in range(10):
        var expected = Float32(i) * 1.5
        var actual = tensor._get_float32(i)
        assert_almost_equal(Float64(actual), Float64(expected), tolerance=1e-6)


fn test_set_float32_dtype_conversions() raises:
    """Test _set_float32() handles different dtypes correctly."""
    var tensor_f16 = zeros([5], DType.float16)
    tensor_f16._set_float32(2, Float32(1.5))
    var val_f16 = tensor_f16._get_float64(2)
    assert_almost_equal(val_f16, 1.5, tolerance=1e-3)

    var tensor_f64 = zeros([5], DType.float64)
    tensor_f64._set_float32(2, Float32(1.5))
    var val_f64 = tensor_f64._get_float64(2)
    assert_almost_equal(val_f64, 1.5, tolerance=1e-6)


fn test_set_get_float32_roundtrip() raises:
    """Test _set_float32() -> _get_float32() roundtrip preserves values."""
    var tensor = zeros([20], DType.float32)
    for i in range(20):
        tensor._set_float32(i, Float32(i) * 0.5)
    for i in range(20):
        var expected = Float32(i) * 0.5
        var actual = tensor._get_float32(i)
        assert_almost_equal(Float64(actual), Float64(expected), tolerance=1e-6)


fn main() raises:
    """Run _get_float32 and _set_float32 tests."""
    print("Running _get_float32 tests...")
    test_get_float32_basic()
    test_get_float32_dtype_conversions()

    print("Running _set_float32 tests...")
    test_set_float32_basic()
    test_set_float32_all_elements()
    test_set_float32_dtype_conversions()
    test_set_get_float32_roundtrip()

    print("All _get/_set_float32 tests passed!")
