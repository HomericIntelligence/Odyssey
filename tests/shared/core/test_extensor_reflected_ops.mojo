"""Tests for ExTensor reflected operators (__radd__, __rsub__, __rmul__, __rtruediv__).

Note: Split from test_extensor_operators.mojo due to Mojo 0.26.1 heap
corruption bug that occurs after ~15 cumulative tests. See ADR-009.
"""

from shared.core.extensor import ExTensor, zeros, ones, full
from tests.shared.conftest import assert_true, assert_almost_equal, assert_equal


fn test_radd_tensors() raises:
    """Test __radd__: reflected addition a + b = b + a (commutative)."""
    var a = full([2, 3], 2.0, DType.float32)
    var b = full([2, 3], 3.0, DType.float32)
    var result1 = a + b
    var result2 = b + a
    assert_equal(len(result1.shape()), len(result2.shape()))
    assert_equal(result1.numel(), result2.numel())
    for i in range(result1.numel()):
        assert_almost_equal(Float64(result1._get_float32(i)), 5.0, tolerance=1e-6)
        assert_almost_equal(Float64(result2._get_float32(i)), 5.0, tolerance=1e-6)


fn test_rsub_tensors() raises:
    """Test __rsub__: reflected subtraction (order matters)."""
    var a = full([2, 3], 2.0, DType.float32)
    var b = full([2, 3], 5.0, DType.float32)
    var result1 = a - b
    var result2 = b - a
    for i in range(result1.numel()):
        assert_almost_equal(Float64(result1._get_float32(i)), -3.0, tolerance=1e-6)
        assert_almost_equal(Float64(result2._get_float32(i)), 3.0, tolerance=1e-6)


fn test_rmul_tensors() raises:
    """Test __rmul__: reflected multiplication (commutative)."""
    var a = full([3, 2], 2.0, DType.float32)
    var b = full([3, 2], 3.0, DType.float32)
    var result1 = a * b
    var result2 = b * a
    for i in range(result1.numel()):
        assert_almost_equal(Float64(result1._get_float32(i)), 6.0, tolerance=1e-6)
        assert_almost_equal(Float64(result2._get_float32(i)), 6.0, tolerance=1e-6)


fn test_rtruediv_tensors() raises:
    """Test __rtruediv__: reflected division (order matters)."""
    var a = full([2, 2], 2.0, DType.float32)
    var b = full([2, 2], 8.0, DType.float32)
    var result1 = a / b
    var result2 = b / a
    for i in range(result1.numel()):
        assert_almost_equal(Float64(result1._get_float32(i)), 0.25, tolerance=1e-6)
        assert_almost_equal(Float64(result2._get_float32(i)), 4.0, tolerance=1e-6)


fn main() raises:
    """Run reflected operator tests."""
    test_radd_tensors()
    test_rsub_tensors()
    test_rmul_tensors()
    test_rtruediv_tensors()
    print("All reflected operator tests passed!")
