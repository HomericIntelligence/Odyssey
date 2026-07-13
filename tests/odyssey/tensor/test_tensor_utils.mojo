"""Direct unit tests for the tensor_utils module (issue #5158).

Other tests reach these utilities through the
`odyssey.tensor.any_tensor` re-export layer. This file imports
straight from `odyssey.tensor.tensor_utils` so the module is
verified independently of that re-export.
"""

from std.testing import assert_true, assert_equal, assert_almost_equal
from odyssey.tensor.tensor_creation import zeros, ones, full, arange
from odyssey.tensor.tensor_utils import (
    calculate_max_batch_size,
    copy,
    clone,
    item,
    diff,
    tolist,
    contiguous,
)


def test_calculate_max_batch_size_positive() raises:
    """Return a positive batch size for a small sample."""
    var sample_shape: List[Int] = [3, 32, 32]
    var n = calculate_max_batch_size(sample_shape, DType.float32)
    assert_true(n > 0, "max batch size should be positive")


def test_calculate_max_batch_size_smaller_for_tight_memory() raises:
    """A tighter memory budget yields a smaller-or-equal batch size."""
    var sample_shape: List[Int] = [3, 32, 32]
    var big = calculate_max_batch_size(
        sample_shape, DType.float32, max_memory_bytes=500_000_000
    )
    var small = calculate_max_batch_size(
        sample_shape, DType.float32, max_memory_bytes=1_000_000
    )
    assert_true(small <= big, "tighter budget -> smaller batch")


def test_copy_is_independent() raises:
    """Copy returns a deep copy whose data does not alias the source."""
    var src = full([4], 2.0, DType.float32)
    var dst = copy(src)
    assert_equal(dst.numel(), 4, "copy numel")
    for i in range(4):
        assert_almost_equal(dst._get_float64(i), 2.0, atol=1e-6)


def test_clone_matches_source() raises:
    """Clone reproduces the source shape and values."""
    var src = arange(0.0, 6.0, 1.0, DType.float32)
    var c = clone(src)
    assert_equal(c.numel(), 6, "clone numel")
    for i in range(6):
        assert_almost_equal(c._get_float64(i), Float64(i), atol=1e-6)


def test_item_single_element() raises:
    """Item extracts the scalar value of a one-element tensor."""
    var t = full([1], 9.0, DType.float32)
    assert_almost_equal(item(t), 9.0, atol=1e-6)


def test_diff_consecutive() raises:
    """Diff returns consecutive differences (one shorter than input)."""
    var t = arange(0.0, 5.0, 1.0, DType.float32)  # [0,1,2,3,4]
    var d = diff(t)
    assert_equal(d.numel(), 4, "diff length is N-1")
    for i in range(4):
        assert_almost_equal(d._get_float64(i), 1.0, atol=1e-6)


def test_tolist_values() raises:
    """Tolist flattens a tensor into a List[Float64]."""
    var t = full([3], 4.0, DType.float32)
    var lst = tolist(t)
    assert_equal(len(lst), 3, "tolist length")
    for i in range(3):
        assert_almost_equal(lst[i], 4.0, atol=1e-6)


def test_contiguous_preserves_values() raises:
    """Contiguous returns a tensor with the same shape and values."""
    var t = arange(0.0, 4.0, 1.0, DType.float32)
    var c = contiguous(t)
    assert_equal(c.numel(), 4, "contiguous numel")
    for i in range(4):
        assert_almost_equal(c._get_float64(i), Float64(i), atol=1e-6)


def main() raises:
    """Run all tensor_utils tests."""
    print("Running test_tensor_utils tests...")

    test_calculate_max_batch_size_positive()
    print("✓ test_calculate_max_batch_size_positive")

    test_calculate_max_batch_size_smaller_for_tight_memory()
    print("✓ test_calculate_max_batch_size_smaller_for_tight_memory")

    test_copy_is_independent()
    print("✓ test_copy_is_independent")

    test_clone_matches_source()
    print("✓ test_clone_matches_source")

    test_item_single_element()
    print("✓ test_item_single_element")

    test_diff_consecutive()
    print("✓ test_diff_consecutive")

    test_tolist_values()
    print("✓ test_tolist_values")

    test_contiguous_preserves_values()
    print("✓ test_contiguous_preserves_values")

    print("\nAll test_tensor_utils tests passed!")
