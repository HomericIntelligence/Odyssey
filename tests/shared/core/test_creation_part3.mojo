# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_creation.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for ExTensor creation operations - Part 3: from_array() placeholders, arange(), and eye().

Tests arange() and eye() creation functions, plus remaining from_array() placeholders.
Split from test_creation.mojo per ADR-009 (≤10 fn test_ per file).
"""

# Import ExTensor and creation operations
from shared.core import (
    ExTensor,
    arange,
    eye,
)

# Import test helpers
from tests.shared.conftest import (
    assert_true,
    assert_false,
    assert_equal_int,
    assert_equal_float,
    assert_close_float,
    assert_shape,
    assert_dtype,
    assert_numel,
    assert_dim,
    assert_value_at,
    assert_all_values,
    assert_all_close,
)


# ============================================================================
# Test from_array() (placeholders - not yet implemented, see #3013)
# ============================================================================


fn test_from_array_2d() raises:
    """Test creating tensor from 2D nested array.

    NOTE(#3013): from_array() is not yet implemented. See test_from_array_1d
    for details.
    """
    pass


fn test_from_array_3d() raises:
    """Test creating tensor from 3D nested array.

    NOTE(#3013): from_array() is not yet implemented. See test_from_array_1d
    for details.
    """
    pass


# ============================================================================
# Test arange()
# ============================================================================


fn test_arange_basic() raises:
    """Test arange with start, stop, step=1."""
    var t = arange(0.0, 10.0, 1.0, DType.float32)

    assert_numel(t, 10, "arange(0, 10, 1) should have 10 elements")
    assert_dim(t, 1, "arange should be 1D")
    assert_dtype(t, DType.float32, "arange should have float32 dtype")

    # Check values: [0, 1, 2, ..., 9]
    for i in range(10):
        assert_value_at(
            t, i, Float64(i), 1e-6, "arange value at index " + String(i)
        )


fn test_arange_step_2() raises:
    """Test arange with step > 1."""
    var t = arange(0.0, 10.0, 2.0, DType.float32)

    assert_numel(t, 5, "arange(0, 10, 2) should have 5 elements")
    assert_value_at(t, 0, 0.0, 1e-6, "arange[0]")
    assert_value_at(t, 1, 2.0, 1e-6, "arange[1]")
    assert_value_at(t, 2, 4.0, 1e-6, "arange[2]")
    assert_value_at(t, 3, 6.0, 1e-6, "arange[3]")
    assert_value_at(t, 4, 8.0, 1e-6, "arange[4]")


fn test_arange_step_fractional() raises:
    """Test arange with fractional step."""
    var t = arange(0.0, 1.0, 0.2, DType.float64)

    assert_numel(t, 5, "arange(0, 1, 0.2) should have 5 elements")
    assert_value_at(t, 0, 0.0, 1e-8, "arange fractional [0]")
    assert_value_at(t, 1, 0.2, 1e-8, "arange fractional [1]")
    assert_value_at(t, 2, 0.4, 1e-8, "arange fractional [2]")
    assert_value_at(t, 3, 0.6, 1e-8, "arange fractional [3]")
    assert_value_at(t, 4, 0.8, 1e-8, "arange fractional [4]")


fn test_arange_reverse() raises:
    """Test arange with negative step (reverse order)."""
    var t = arange(10.0, 0.0, -1.0, DType.float32)

    assert_numel(t, 10, "arange(10, 0, -1) should have 10 elements")
    # Check values: [10, 9, 8, ..., 1]
    for i in range(10):
        assert_value_at(t, i, Float64(10 - i), 1e-6, "arange reverse value")


fn test_arange_float() raises:
    """Test arange with float dtype."""
    var t = arange(1.5, 5.5, 1.0, DType.float64)

    assert_numel(t, 4, "arange(1.5, 5.5, 1.0) should have 4 elements")
    assert_dtype(t, DType.float64, "arange should have float64 dtype")
    assert_value_at(t, 0, 1.5, 1e-8, "arange float [0]")
    assert_value_at(t, 1, 2.5, 1e-8, "arange float [1]")
    assert_value_at(t, 2, 3.5, 1e-8, "arange float [2]")
    assert_value_at(t, 3, 4.5, 1e-8, "arange float [3]")


# ============================================================================
# Test eye() - square identity matrix
# ============================================================================


fn test_eye_square() raises:
    """Test creating square identity matrix."""
    var t = eye(5, 5, 0, DType.float32)

    assert_dim(t, 2, "eye should be 2D")
    assert_numel(t, 25, "eye(5,5) should have 25 elements")
    assert_dtype(t, DType.float32, "eye should have float32 dtype")

    # Check diagonal is 1, off-diagonal is 0
    for i in range(5):
        for j in range(5):
            var flat_idx = i * 5 + j
            if i == j:
                assert_value_at(
                    t, flat_idx, 1.0, 1e-8, "eye diagonal should be 1.0"
                )
            else:
                assert_value_at(
                    t, flat_idx, 0.0, 1e-8, "eye off-diagonal should be 0.0"
                )


# ============================================================================
# Main test runner
# ============================================================================


fn main() raises:
    """Run from_array() placeholder, arange(), and eye() (square) creation tests.
    """
    print(
        "Running ExTensor creation tests - Part 3: from_array placeholders,"
        " arange(), eye()..."
    )

    # from_array() placeholder tests
    test_from_array_2d()
    test_from_array_3d()

    # arange() tests
    test_arange_basic()
    test_arange_step_2()
    test_arange_step_fractional()
    test_arange_reverse()
    test_arange_float()

    # eye() tests
    test_eye_square()

    print("All Part 3 creation tests completed!")
