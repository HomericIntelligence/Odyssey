"""Tests for AnyTensor scalar and list constructor consolidation.

Comprehensive test coverage for the refactored constructors that now delegate
to private helper methods. Tests verify correct dtype, shape, numel, strides,
and value initialization for all constructor types.
"""

from std.testing import assert_true, assert_almost_equal
from projectodyssey.tensor.any_tensor import AnyTensor


def test_intliteral_scalar() raises:
    """IntLiteral constructor creates 0D int64 scalar tensor."""
    var t: AnyTensor = 42
    assert_true(t.dtype() == DType.int64, "dtype should be int64")
    assert_true(t.numel() == 1, "numel should be 1")
    var shape = t.shape()
    assert_true(len(shape) == 0, "shape should be 0D (empty)")
    assert_true(t[0] == 42.0, "value should be 42")
    print("PASS: test_intliteral_scalar")


def test_floatliteral_scalar() raises:
    """FloatLiteral constructor creates 0D float64 scalar tensor."""
    var t: AnyTensor = 3.14
    assert_true(t.dtype() == DType.float64, "dtype should be float64")
    assert_true(t.numel() == 1, "numel should be 1")
    var shape = t.shape()
    assert_true(len(shape) == 0, "shape should be 0D (empty)")
    assert_almost_equal(Float64(t[0]), 3.14, atol=1e-10)
    print("PASS: test_floatliteral_scalar")


def test_int_scalar() raises:
    """Int constructor creates 0D int64 scalar tensor."""
    var val: Int = 100
    var t: AnyTensor = val
    assert_true(t.dtype() == DType.int64, "dtype should be int64")
    assert_true(t.numel() == 1, "numel should be 1")
    var shape = t.shape()
    assert_true(len(shape) == 0, "shape should be 0D (empty)")
    assert_true(t[0] == 100.0, "value should be 100")
    print("PASS: test_int_scalar")


def test_float64_scalar() raises:
    """Float64 constructor creates 0D float64 scalar tensor."""
    var val = Float64(2.718)
    var t: AnyTensor = val
    assert_true(t.dtype() == DType.float64, "dtype should be float64")
    assert_true(t.numel() == 1, "numel should be 1")
    var shape = t.shape()
    assert_true(len(shape) == 0, "shape should be 0D (empty)")
    assert_almost_equal(Float64(t[0]), 2.718, atol=1e-10)
    print("PASS: test_float64_scalar")


def test_list_float32() raises:
    """List[Float32] constructor creates 1D float32 tensor."""
    var data = List[Float32]()
    data.append(Float32(1.5))
    data.append(Float32(2.5))
    data.append(Float32(3.5))
    var t = AnyTensor(data)

    assert_true(t.dtype() == DType.float32, "dtype should be float32")
    assert_true(t.numel() == 3, "numel should be 3")
    var shape = t.shape()
    assert_true(len(shape) == 1, "shape should be 1D")
    assert_true(shape[0] == 3, "shape[0] should be 3")
    var strides = t.strides()
    assert_true(len(strides) == 1, "strides should be 1D")
    assert_true(strides[0] == 1, "strides[0] should be 1")

    assert_almost_equal(Float32(t[0]), 1.5, atol=1e-6)
    assert_almost_equal(Float32(t[1]), 2.5, atol=1e-6)
    assert_almost_equal(Float32(t[2]), 3.5, atol=1e-6)
    print("PASS: test_list_float32")


def test_list_int() raises:
    """List[Int] constructor creates 1D int64 tensor."""
    var data = List[Int]()
    data.append(10)
    data.append(20)
    data.append(30)
    var t = AnyTensor(data)

    assert_true(t.dtype() == DType.int64, "dtype should be int64")
    assert_true(t.numel() == 3, "numel should be 3")
    var shape = t.shape()
    assert_true(len(shape) == 1, "shape should be 1D")
    assert_true(shape[0] == 3, "shape[0] should be 3")
    var strides = t.strides()
    assert_true(len(strides) == 1, "strides should be 1D")
    assert_true(strides[0] == 1, "strides[0] should be 1")

    assert_true(t[0] == 10.0, "value[0] should be 10")
    assert_true(t[1] == 20.0, "value[1] should be 20")
    assert_true(t[2] == 30.0, "value[2] should be 30")
    print("PASS: test_list_int")


def test_empty_list_float32() raises:
    """Empty List[Float32] creates valid 1D tensor with numel=0."""
    var data = List[Float32]()
    var t = AnyTensor(data)

    assert_true(t.dtype() == DType.float32, "dtype should be float32")
    assert_true(t.numel() == 0, "numel should be 0 for empty list")
    var shape = t.shape()
    assert_true(len(shape) == 1, "shape should be 1D")
    assert_true(shape[0] == 0, "shape[0] should be 0")
    print("PASS: test_empty_list_float32")


def test_empty_list_int() raises:
    """Empty List[Int] creates valid 1D tensor with numel=0."""
    var data = List[Int]()
    var t = AnyTensor(data)

    assert_true(t.dtype() == DType.int64, "dtype should be int64")
    assert_true(t.numel() == 0, "numel should be 0 for empty list")
    var shape = t.shape()
    assert_true(len(shape) == 1, "shape should be 1D")
    assert_true(shape[0] == 0, "shape[0] should be 0")
    print("PASS: test_empty_list_int")


def test_scalar_strides_empty() raises:
    """All scalar constructors produce empty strides for 0D tensors."""
    var t1: AnyTensor = 42
    var t2: AnyTensor = 3.14
    var t3: AnyTensor = Float64(1.5)
    var val: Int = 5
    var t4: AnyTensor = val

    assert_true(len(t1.strides()) == 0, "int scalar strides should be empty")
    assert_true(len(t2.strides()) == 0, "float literal strides should be empty")
    assert_true(len(t3.strides()) == 0, "float64 scalar strides should be empty")
    assert_true(len(t4.strides()) == 0, "int value strides should be empty")
    print("PASS: test_scalar_strides_empty")


def test_list_strides_one() raises:
    """List constructors produce strides=[1] for 1D tensors."""
    var data1 = List[Float32]()
    data1.append(Float32(1.0))
    data1.append(Float32(2.0))

    var data2 = List[Int]()
    data2.append(1)
    data2.append(2)

    var t1 = AnyTensor(data1)
    var t2 = AnyTensor(data2)

    var strides1 = t1.strides()
    var strides2 = t2.strides()

    assert_true(len(strides1) == 1, "float32 strides should be 1D")
    assert_true(strides1[0] == 1, "float32 stride should be 1")
    assert_true(len(strides2) == 1, "int strides should be 1D")
    assert_true(strides2[0] == 1, "int stride should be 1")
    print("PASS: test_list_strides_one")


def test_negative_intliteral() raises:
    """IntLiteral constructor handles negative values."""
    var t: AnyTensor = -999
    assert_true(t.dtype() == DType.int64, "dtype should be int64")
    assert_true(t[0] == -999.0, "negative value should be preserved")
    print("PASS: test_negative_intliteral")


def test_negative_floatliteral() raises:
    """FloatLiteral constructor handles negative values."""
    var t: AnyTensor = -3.14
    assert_true(t.dtype() == DType.float64, "dtype should be float64")
    assert_almost_equal(Float64(t[0]), -3.14, atol=1e-10)
    print("PASS: test_negative_floatliteral")


def test_list_float32_large() raises:
    """List[Float32] constructor handles larger lists."""
    var data = List[Float32]()
    for i in range(100):
        data.append(Float32(i + 0.5))

    var t = AnyTensor(data)
    assert_true(t.numel() == 100, "numel should be 100")
    assert_almost_equal(Float32(t[0]), 0.5, atol=1e-6)
    assert_almost_equal(Float32(t[99]), 99.5, atol=1e-6)
    print("PASS: test_list_float32_large")


def test_list_int_large() raises:
    """List[Int] constructor handles larger lists."""
    var data = List[Int]()
    for i in range(50):
        data.append(i * 2)

    var t = AnyTensor(data)
    assert_true(t.numel() == 50, "numel should be 50")
    assert_true(t[0] == 0.0, "value[0] should be 0")
    assert_true(t[49] == 98.0, "value[49] should be 98")
    print("PASS: test_list_int_large")
