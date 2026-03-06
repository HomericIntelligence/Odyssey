"""Tests for ExTensor in-place operators (__iadd__, __isub__, __imul__, __itruediv__).

Note: Split from test_extensor_operators.mojo due to Mojo 0.26.1 heap
corruption bug that occurs after ~15 cumulative tests. See ADR-009.
"""

from shared.core.extensor import ExTensor, zeros, ones, full
from tests.shared.conftest import assert_true, assert_almost_equal, assert_equal


fn test_iadd_basic() raises:
    """Test __iadd__: in-place addition tensor += other."""
    var a = full([2, 3], 2.0, DType.float32)
    var b = full([2, 3], 3.0, DType.float32)
    a += b
    for i in range(a.numel()):
        assert_almost_equal(Float64(a._get_float32(i)), 5.0, tolerance=1e-6)


fn test_isub_basic() raises:
    """Test __isub__: in-place subtraction tensor -= other."""
    var a = full([2, 3], 5.0, DType.float32)
    var b = full([2, 3], 2.0, DType.float32)
    a -= b
    for i in range(a.numel()):
        assert_almost_equal(Float64(a._get_float32(i)), 3.0, tolerance=1e-6)


fn test_imul_basic() raises:
    """Test __imul__: in-place multiplication tensor *= other."""
    var a = full([2, 3], 2.0, DType.float32)
    var b = full([2, 3], 3.0, DType.float32)
    a *= b
    for i in range(a.numel()):
        assert_almost_equal(Float64(a._get_float32(i)), 6.0, tolerance=1e-6)


fn test_itruediv_basic() raises:
    """Test __itruediv__: in-place division tensor /= other."""
    var a = full([2, 3], 8.0, DType.float32)
    var b = full([2, 3], 2.0, DType.float32)
    a /= b
    for i in range(a.numel()):
        assert_almost_equal(Float64(a._get_float32(i)), 4.0, tolerance=1e-6)


fn test_inplace_operators_chain() raises:
    """Test chaining multiple in-place operators."""
    var a = full([2, 2], 10.0, DType.float32)
    var b = full([2, 2], 2.0, DType.float32)
    var c = full([2, 2], 3.0, DType.float32)
    a /= b
    a *= c
    for i in range(a.numel()):
        assert_almost_equal(Float64(a._get_float32(i)), 15.0, tolerance=1e-6)


fn main() raises:
    """Run in-place operator tests."""
    test_iadd_basic()
    test_isub_basic()
    test_imul_basic()
    test_itruediv_basic()
    test_inplace_operators_chain()
    print("All in-place operator tests passed!")
