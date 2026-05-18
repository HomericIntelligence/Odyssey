# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
"""Unit tests for AnyTensor._normalize_slice_indices helper."""

from projectodyssey.tensor.any_tensor import AnyTensor, zeros
from tests.projectodyssey.conftest import assert_equal


def _make_1d(size: Int) raises -> AnyTensor:
    """Create a 1D tensor of the given size."""
    var shape = List[Int]()
    shape.append(size)
    return AnyTensor(shape, DType.float32)


def test_normalize_forward_full() raises:
    """[0:5:1] on size=5 → start=0, end=5, step=1, result_size=5."""
    var t = _make_1d(5)
    var norm = t._normalize_slice_indices(0, 5, 1, 5)
    assert_equal(norm[0], 0)
    assert_equal(norm[1], 5)
    assert_equal(norm[2], 1)
    assert_equal(norm[3], 5)


def test_normalize_forward_basic() raises:
    """[2:7:1] on size=10 → start=2, end=7, step=1, result_size=5."""
    var t = _make_1d(10)
    var norm = t._normalize_slice_indices(2, 7, 1, 10)
    assert_equal(norm[0], 2)
    assert_equal(norm[1], 7)
    assert_equal(norm[2], 1)
    assert_equal(norm[3], 5)


def test_normalize_empty_slice() raises:
    """[3:3:1] on size=5 → result_size=0."""
    var t = _make_1d(5)
    var norm = t._normalize_slice_indices(3, 3, 1, 5)
    assert_equal(norm[0], 3)
    assert_equal(norm[1], 3)
    assert_equal(norm[2], 1)
    assert_equal(norm[3], 0)


def test_normalize_negative_start() raises:
    """[-3:10:1] on size=10 → start=7, end=10, result_size=3."""
    var t = _make_1d(10)
    var norm = t._normalize_slice_indices(-3, 10, 1, 10)
    assert_equal(norm[0], 7)
    assert_equal(norm[1], 10)
    assert_equal(norm[2], 1)
    assert_equal(norm[3], 3)


def test_normalize_negative_end() raises:
    """[0:-2:1] on size=10 → start=0, end=8, result_size=8."""
    var t = _make_1d(10)
    var norm = t._normalize_slice_indices(0, -2, 1, 10)
    assert_equal(norm[0], 0)
    assert_equal(norm[1], 8)
    assert_equal(norm[2], 1)
    assert_equal(norm[3], 8)


def test_normalize_strided_step2() raises:
    """[0:10:2] on size=10 → result_size=5."""
    var t = _make_1d(10)
    var norm = t._normalize_slice_indices(0, 10, 2, 10)
    assert_equal(norm[0], 0)
    assert_equal(norm[1], 10)
    assert_equal(norm[2], 2)
    assert_equal(norm[3], 5)


def test_normalize_reverse_full() raises:
    """[4:-1:-1] on size=5 → start=4, end=-1, result_size=5."""
    var t = _make_1d(5)
    var norm = t._normalize_slice_indices(4, -1, -1, 5)
    assert_equal(norm[0], 4)
    assert_equal(norm[1], -1)
    assert_equal(norm[2], -1)
    assert_equal(norm[3], 5)


def test_normalize_reverse_partial() raises:
    """[3:1:-1] on size=5 → start=3, end=1, result_size=2."""
    var t = _make_1d(5)
    var norm = t._normalize_slice_indices(3, 1, -1, 5)
    assert_equal(norm[0], 3)
    assert_equal(norm[1], 1)
    assert_equal(norm[2], -1)
    assert_equal(norm[3], 2)


def test_normalize_oob_clamp_forward() raises:
    """[8:20:1] on size=10 → end clamped to 10, result_size=2."""
    var t = _make_1d(10)
    var norm = t._normalize_slice_indices(8, 20, 1, 10)
    assert_equal(norm[0], 8)
    assert_equal(norm[1], 10)
    assert_equal(norm[2], 1)
    assert_equal(norm[3], 2)


def test_normalize_oob_clamp_reverse() raises:
    """[20:-1:-1] on size=10 → start clamped to 9, result_size=10."""
    var t = _make_1d(10)
    var norm = t._normalize_slice_indices(20, -1, -1, 10)
    assert_equal(norm[0], 9)
    assert_equal(norm[1], -1)
    assert_equal(norm[2], -1)
    assert_equal(norm[3], 10)


def main() raises:
    """Run all _normalize_slice_indices tests."""
    test_normalize_forward_full()
    test_normalize_forward_basic()
    test_normalize_empty_slice()
    test_normalize_negative_start()
    test_normalize_negative_end()
    test_normalize_strided_step2()
    test_normalize_reverse_full()
    test_normalize_reverse_partial()
    test_normalize_oob_clamp_forward()
    test_normalize_oob_clamp_reverse()
    print("All _normalize_slice_indices tests passed!")
