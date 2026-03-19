# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split per file convention. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for ExTensor multi-dimensional slicing with step support.

The *slices overload of __getitem__ now supports non-unit steps on all
dimensions, consistent with the 1D __getitem__(Slice) overload.  These
tests verify that step slicing produces correct shapes and values.
"""

from shared.core.extensor import ExTensor, zeros, arange
from tests.shared.conftest import assert_true, assert_equal


# ============================================================================
# Step support tests for __getitem__(*slices: Slice)
# ============================================================================


fn test_multidim_step2_first_dim() raises:
    """Step=2 on first dimension selects every other row."""
    var t = arange(0.0, 20.0, 1.0, DType.float32)
    var t2d = t.reshape([5, 4])

    # tensor[::2, :] — rows 0, 2, 4 -> shape [3, 4]
    var sliced = t2d[::2, :]
    var shape = sliced.shape()
    assert_equal(len(shape), 2)
    assert_equal(shape[0], 3)
    assert_equal(shape[1], 4)

    # Verify values via flat indexing on the result tensor:
    # row 0 (orig row 0) = [0,1,2,3], row 1 (orig row 2) = [8,9,10,11],
    # row 2 (orig row 4) = [16,17,18,19]
    assert_equal(Int(sliced[0]), 0)
    assert_equal(Int(sliced[3]), 3)
    assert_equal(Int(sliced[4]), 8)
    assert_equal(Int(sliced[7]), 11)
    assert_equal(Int(sliced[8]), 16)
    assert_equal(Int(sliced[11]), 19)
    print("PASS: test_multidim_step2_first_dim")


fn test_multidim_step2_second_dim() raises:
    """Step=2 on second dimension selects every other column."""
    var t = arange(0.0, 20.0, 1.0, DType.float32)
    var t2d = t.reshape([5, 4])

    # tensor[:, ::2] — columns 0, 2 -> shape [5, 2]
    var sliced = t2d[:, ::2]
    var shape = sliced.shape()
    assert_equal(len(shape), 2)
    assert_equal(shape[0], 5)
    assert_equal(shape[1], 2)

    # Verify values: row 0 cols 0,2 = [0,2]; row 1 cols 0,2 = [4,6]
    assert_equal(Int(sliced[0]), 0)
    assert_equal(Int(sliced[1]), 2)
    assert_equal(Int(sliced[2]), 4)
    assert_equal(Int(sliced[3]), 6)
    print("PASS: test_multidim_step2_second_dim")


fn test_multidim_negative_step() raises:
    """Negative step on first dimension reverses row order."""
    var t = arange(0.0, 20.0, 1.0, DType.float32)
    var t2d = t.reshape([5, 4])

    # tensor[::-1, :] — rows in reverse: 4, 3, 2, 1, 0 -> shape [5, 4]
    var sliced = t2d[::-1, :]
    var shape = sliced.shape()
    assert_equal(len(shape), 2)
    assert_equal(shape[0], 5)
    assert_equal(shape[1], 4)

    # First row of result = original row 4 = [16,17,18,19]
    assert_equal(Int(sliced[0]), 16)
    assert_equal(Int(sliced[3]), 19)
    # Last row of result = original row 0 = [0,1,2,3]
    assert_equal(Int(sliced[16]), 0)
    assert_equal(Int(sliced[19]), 3)
    print("PASS: test_multidim_negative_step")


fn test_multidim_step3_3d() raises:
    """Step=3 in 3D tensor selects every 3rd element along first dim."""
    var t = arange(0.0, 24.0, 1.0, DType.float32)
    var t3d = t.reshape([4, 3, 2])

    # tensor[::3, :, :] — indices 0, 3 along first dim -> shape [2, 3, 2]
    var sliced = t3d[::3, :, :]
    var shape = sliced.shape()
    assert_equal(len(shape), 3)
    assert_equal(shape[0], 2)
    assert_equal(shape[1], 3)
    assert_equal(shape[2], 2)
    print("PASS: test_multidim_step3_3d")


fn test_multidim_step1_does_not_raise() raises:
    """Explicit step=1 on all dimensions must NOT raise Error."""
    var t = arange(0.0, 20.0, 1.0, DType.float32)
    var t2d = t.reshape([5, 4])

    # tensor[::1, ::1] — step=1 is valid, should succeed
    var sliced = t2d[::1, ::1]
    var shape = sliced.shape()
    assert_equal(len(shape), 2)
    assert_equal(shape[0], 5)
    assert_equal(shape[1], 4)
    print("PASS: test_multidim_step1_does_not_raise")


fn test_multidim_no_step_does_not_raise() raises:
    """Omitted step (defaults to 1) must NOT raise Error."""
    var t = arange(0.0, 20.0, 1.0, DType.float32)
    var t2d = t.reshape([5, 4])

    # tensor[1:4, 1:3] — no step specified, should succeed
    var sliced = t2d[1:4, 1:3]
    var shape = sliced.shape()
    assert_equal(len(shape), 2)
    assert_equal(shape[0], 3)
    assert_equal(shape[1], 2)
    print("PASS: test_multidim_no_step_does_not_raise")


fn test_multidim_step0_raises() raises:
    """Step=0 must raise Error (invalid step)."""
    var t = arange(0.0, 20.0, 1.0, DType.float32)
    var t2d = t.reshape([5, 4])

    var raised = False
    try:
        var _ = t2d[::0, :]
    except:
        raised = True

    assert_true(raised, "Expected Error for step=0 on first dim")
    print("PASS: test_multidim_step0_raises")


fn main() raises:
    """Run all multi-dim step validation tests."""
    test_multidim_step2_first_dim()
    test_multidim_step2_second_dim()
    test_multidim_negative_step()
    test_multidim_step3_3d()
    test_multidim_step1_does_not_raise()
    test_multidim_no_step_does_not_raise()
    test_multidim_step0_raises()
    print("All multi-dim step validation tests passed!")
