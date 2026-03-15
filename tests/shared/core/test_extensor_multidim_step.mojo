# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split per file convention. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for ExTensor multi-dimensional slicing step validation (issue #4463).

The *slices overload of __getitem__ previously ignored the step field,
silently returning wrong results for e.g. tensor[::2, :].  These tests
verify that a non-unit step now raises an Error (fail-fast, option 1).
"""

from shared.core.extensor import ExTensor, zeros, arange
from tests.shared.conftest import assert_true, assert_equal


# ============================================================================
# Step-validation tests for __getitem__(*slices: Slice)
# ============================================================================


fn test_multidim_step2_first_dim_raises() raises:
    """Step=2 on first dimension must raise Error. Closes #4463."""
    var t = arange(0.0, 20.0, 1.0, DType.float32)
    var t2d = t.reshape([5, 4])

    var raised = False
    try:
        # tensor[::2, :] — step=2 should be rejected
        var _ = t2d[::2, :]
    except:
        raised = True

    assert_true(raised, "Expected Error for step=2 on first dim")
    print("PASS: test_multidim_step2_first_dim_raises")


fn test_multidim_step2_second_dim_raises() raises:
    """Step=2 on second dimension must raise Error. Closes #4463."""
    var t = arange(0.0, 20.0, 1.0, DType.float32)
    var t2d = t.reshape([5, 4])

    var raised = False
    try:
        # tensor[:, ::2] — step=2 on second dim should be rejected
        var _ = t2d[:, ::2]
    except:
        raised = True

    assert_true(raised, "Expected Error for step=2 on second dim")
    print("PASS: test_multidim_step2_second_dim_raises")


fn test_multidim_negative_step_raises() raises:
    """Negative step on any dimension must raise Error. Closes #4463."""
    var t = arange(0.0, 20.0, 1.0, DType.float32)
    var t2d = t.reshape([5, 4])

    var raised = False
    try:
        # tensor[::-1, :] — negative step should be rejected
        var _ = t2d[::-1, :]
    except:
        raised = True

    assert_true(raised, "Expected Error for negative step on first dim")
    print("PASS: test_multidim_negative_step_raises")


fn test_multidim_step3_3d_raises() raises:
    """Step=3 in 3D tensor must raise Error. Closes #4463."""
    var t = arange(0.0, 24.0, 1.0, DType.float32)
    var t3d = t.reshape([4, 3, 2])

    var raised = False
    try:
        var _ = t3d[::3, :, :]
    except:
        raised = True

    assert_true(raised, "Expected Error for step=3 on first dim of 3D tensor")
    print("PASS: test_multidim_step3_3d_raises")


fn test_multidim_step1_does_not_raise() raises:
    """Explicit step=1 on all dimensions must NOT raise Error. Closes #4463."""
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
    """Omitted step (defaults to 1) must NOT raise Error. Closes #4463."""
    var t = arange(0.0, 20.0, 1.0, DType.float32)
    var t2d = t.reshape([5, 4])

    # tensor[1:4, 1:3] — no step specified, should succeed
    var sliced = t2d[1:4, 1:3]
    var shape = sliced.shape()
    assert_equal(len(shape), 2)
    assert_equal(shape[0], 3)
    assert_equal(shape[1], 2)
    print("PASS: test_multidim_no_step_does_not_raise")


fn main() raises:
    """Run all multi-dim step validation tests."""
    test_multidim_step2_first_dim_raises()
    test_multidim_step2_second_dim_raises()
    test_multidim_negative_step_raises()
    test_multidim_step3_3d_raises()
    test_multidim_step1_does_not_raise()
    test_multidim_no_step_does_not_raise()
    print("All multi-dim step validation tests passed!")
