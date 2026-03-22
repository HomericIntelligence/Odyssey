# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_creation.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for AnyTensor creation operations - Part 4: eye() (rectangular/offset) and linspace().

Tests rectangular and offset-diagonal eye() and linspace() creation functions.
Split from test_creation.mojo per ADR-009 (≤10 fn test_ per file).
"""

# Import AnyTensor and creation operations
from shared.core.extensor import AnyTensor, eye, linspace

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
# Test eye() - rectangular and offset diagonal
# ============================================================================


fn test_eye_rectangular() raises:
    """Test creating rectangular identity matrix."""
    var t = eye(3, 5, 0, DType.float64)

    assert_dim(t, 2, "eye should be 2D")
    assert_numel(t, 15, "eye(3,5) should have 15 elements")
    assert_dtype(t, DType.float64, "eye should have float64 dtype")

    # Check diagonal is 1 where i==j, rest is 0
    for i in range(3):
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


fn test_eye_offset_diagonal() raises:
    """Test creating identity matrix with offset diagonal (k parameter)."""
    # Test k=1 (superdiagonal - ones above main diagonal)
    var t1 = eye(4, 4, 1, DType.float32)
    assert_dim(t1, 2, "eye should be 2D")
    assert_numel(t1, 16, "eye(4,4) should have 16 elements")

    for i in range(4):
        for j in range(4):
            var flat_idx = i * 4 + j
            if j == i + 1:  # superdiagonal
                assert_value_at(
                    t1, flat_idx, 1.0, 1e-8, "superdiagonal should be 1.0"
                )
            else:
                assert_value_at(
                    t1, flat_idx, 0.0, 1e-8, "non-superdiagonal should be 0.0"
                )

    # Test k=-1 (subdiagonal - ones below main diagonal)
    var t2 = eye(4, 4, -1, DType.float32)
    for i in range(4):
        for j in range(4):
            var flat_idx = i * 4 + j
            if j == i - 1:  # subdiagonal
                assert_value_at(
                    t2, flat_idx, 1.0, 1e-8, "subdiagonal should be 1.0"
                )
            else:
                assert_value_at(
                    t2, flat_idx, 0.0, 1e-8, "non-subdiagonal should be 0.0"
                )


# ============================================================================
# Test linspace()
# ============================================================================


fn test_linspace_basic() raises:
    """Test linspace with basic range."""
    var t = linspace(0.0, 10.0, 11, DType.float32)

    assert_numel(t, 11, "linspace(0, 10, 11) should have 11 elements")
    assert_dim(t, 1, "linspace should be 1D")
    assert_dtype(t, DType.float32, "linspace should have float32 dtype")

    # Check values: [0, 1, 2, ..., 10]
    for i in range(11):
        assert_value_at(
            t, i, Float64(i), 1e-6, "linspace value at index " + String(i)
        )


fn test_linspace_negative_range() raises:
    """Test linspace with negative start/stop."""
    var t = linspace(-5.0, 5.0, 11, DType.float64)

    assert_numel(t, 11, "linspace(-5, 5, 11) should have 11 elements")
    assert_dtype(t, DType.float64, "linspace should have float64 dtype")

    # Check values: [-5, -4, -3, ..., 5]
    for i in range(11):
        assert_value_at(t, i, Float64(-5 + i), 1e-8, "linspace negative value")


fn test_linspace_small_num() raises:
    """Test linspace with small number of points."""
    var t = linspace(0.0, 1.0, 2, DType.float32)

    assert_numel(t, 2, "linspace(0, 1, 2) should have 2 elements")
    assert_value_at(t, 0, 0.0, 1e-6, "linspace start should be 0.0")
    assert_value_at(t, 1, 1.0, 1e-6, "linspace end should be 1.0")


fn test_linspace_large_num() raises:
    """Test linspace with large number of points."""
    var t = linspace(0.0, 100.0, 101, DType.float64)

    assert_numel(t, 101, "linspace(0, 100, 101) should have 101 elements")
    # Spot-check a few values
    assert_value_at(t, 0, 0.0, 1e-8, "linspace start")
    assert_value_at(t, 50, 50.0, 1e-6, "linspace middle")
    assert_value_at(t, 100, 100.0, 1e-8, "linspace end")


# ============================================================================
# Main test runner
# ============================================================================


fn main() raises:
    """Run eye() (rectangular/offset) and linspace() creation tests."""
    print(
        "Running AnyTensor creation tests - Part 4: eye() rectangular/offset and"
        " linspace()..."
    )

    # eye() tests
    test_eye_rectangular()
    test_eye_offset_diagonal()

    # linspace() tests
    test_linspace_basic()
    test_linspace_negative_range()
    test_linspace_small_num()
    test_linspace_large_num()

    print("All Part 4 creation tests completed!")
