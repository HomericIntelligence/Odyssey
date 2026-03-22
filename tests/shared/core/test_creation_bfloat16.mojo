# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_creation.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for bfloat16 dtype support in AnyTensor factory functions.

Verifies that arange(), eye(), linspace(), and randn() correctly route
bfloat16 tensors to _set_float64 (floating-point path) rather than
_set_int64. Regression tests for issue #3906.

Split per ADR-009 (≤10 fn test_ per file).
"""

# Import AnyTensor and creation operations
from shared.core import (
    AnyTensor,
    arange,
    eye,
    linspace,
    randn,
)

# Import test helpers
from tests.shared.conftest import (
    assert_true,
    assert_equal_int,
    assert_dtype,
    assert_numel,
    assert_dim,
    assert_value_at,
)


# ============================================================================
# Test bfloat16 dtype guard in arange()
# ============================================================================


fn test_arange_bfloat16_dtype() raises:
    """Test arange() preserves bfloat16 dtype."""
    var t = arange(0.0, 5.0, 1.0, DType.bfloat16)

    assert_dtype(t, DType.bfloat16, "arange bfloat16 should preserve dtype")
    assert_numel(t, 5, "arange(0, 5, 1) bfloat16 should have 5 elements")
    assert_dim(t, 1, "arange bfloat16 should be 1D")


fn test_arange_bfloat16_values() raises:
    """Test arange() with bfloat16 stores float values (not silently truncated to int)."""
    # bfloat16 has ~2 decimal digits of precision; use integer-valued sequence
    var t = arange(0.0, 4.0, 1.0, DType.bfloat16)

    # Values must be floating-point: 0.0, 1.0, 2.0, 3.0
    # If routed to _set_int64, all values would be 0 (Int64 truncation)
    # bfloat16 tolerance: ~1e-2 for small integers
    assert_value_at(t, 0, 0.0, 1e-2, "arange bfloat16 [0] should be 0.0")
    assert_value_at(t, 1, 1.0, 1e-2, "arange bfloat16 [1] should be 1.0")
    assert_value_at(t, 2, 2.0, 1e-2, "arange bfloat16 [2] should be 2.0")
    assert_value_at(t, 3, 3.0, 1e-2, "arange bfloat16 [3] should be 3.0")


# ============================================================================
# Test bfloat16 dtype guard in eye()
# ============================================================================


fn test_eye_bfloat16_dtype() raises:
    """Test eye() preserves bfloat16 dtype."""
    var t = eye(3, 3, 0, DType.bfloat16)

    assert_dtype(t, DType.bfloat16, "eye bfloat16 should preserve dtype")
    assert_numel(t, 9, "eye(3,3) bfloat16 should have 9 elements")


fn test_eye_bfloat16_values() raises:
    """Test eye() with bfloat16 stores float values on diagonal."""
    var t = eye(3, 3, 0, DType.bfloat16)

    # Diagonal must be 1.0, off-diagonal must be 0.0
    # If _set_int64 were used instead of _set_float64, diagonal would be 0
    for i in range(3):
        for j in range(3):
            var flat_idx = i * 3 + j
            if i == j:
                assert_value_at(
                    t, flat_idx, 1.0, 1e-2, "eye bfloat16 diagonal should be 1.0"
                )
            else:
                assert_value_at(
                    t, flat_idx, 0.0, 1e-2, "eye bfloat16 off-diagonal should be 0.0"
                )


# ============================================================================
# Test bfloat16 dtype guard in linspace()
# ============================================================================


fn test_linspace_bfloat16_dtype() raises:
    """Test linspace() preserves bfloat16 dtype."""
    var t = linspace(0.0, 4.0, 5, DType.bfloat16)

    assert_dtype(t, DType.bfloat16, "linspace bfloat16 should preserve dtype")
    assert_numel(t, 5, "linspace(0, 4, 5) bfloat16 should have 5 elements")
    assert_dim(t, 1, "linspace bfloat16 should be 1D")


fn test_linspace_bfloat16_values() raises:
    """Test linspace() with bfloat16 stores float values (not silently truncated to int)."""
    var t = linspace(0.0, 4.0, 5, DType.bfloat16)

    # Values must be: 0.0, 1.0, 2.0, 3.0, 4.0
    assert_value_at(t, 0, 0.0, 1e-2, "linspace bfloat16 [0] should be 0.0")
    assert_value_at(t, 1, 1.0, 1e-2, "linspace bfloat16 [1] should be 1.0")
    assert_value_at(t, 2, 2.0, 1e-2, "linspace bfloat16 [2] should be 2.0")
    assert_value_at(t, 3, 3.0, 1e-2, "linspace bfloat16 [3] should be 3.0")
    assert_value_at(t, 4, 4.0, 1e-2, "linspace bfloat16 [4] should be 4.0")


# ============================================================================
# Test bfloat16 dtype guard in randn()
# ============================================================================


fn test_randn_bfloat16_dtype() raises:
    """Test randn() preserves bfloat16 dtype."""
    var t = randn([3, 4], DType.bfloat16)

    assert_dtype(t, DType.bfloat16, "randn bfloat16 should preserve dtype")
    assert_numel(t, 12, "randn([3,4]) bfloat16 should have 12 elements")


fn test_randn_bfloat16_nonzero() raises:
    """Test randn() with bfloat16 stores float values (not silently zeroed via int path)."""
    # Use a larger tensor for statistical confidence.
    # If routed to _set_int64, Box-Muller float values would be truncated to 0.
    var t = randn([50], DType.bfloat16, seed=42)

    var nonzero_count = 0
    for i in range(t.numel()):
        var val = t._get_float64(i)
        if val > 1e-3 or val < -1e-3:
            nonzero_count += 1

    # Most values from N(0,1) have |val| > 1e-3; require ≥40 of 50
    assert_true(
        nonzero_count >= 40,
        "randn bfloat16 should store non-zero floats, not int-truncated zeros",
    )


# ============================================================================
# Main test runner
# ============================================================================


fn main() raises:
    """Run bfloat16 dtype guard tests for factory functions."""
    print(
        "Running AnyTensor bfloat16 dtype guard tests (issue #3906)..."
    )

    # arange() bfloat16 tests
    test_arange_bfloat16_dtype()
    test_arange_bfloat16_values()

    # eye() bfloat16 tests
    test_eye_bfloat16_dtype()
    test_eye_bfloat16_values()

    # linspace() bfloat16 tests
    test_linspace_bfloat16_dtype()
    test_linspace_bfloat16_values()

    # randn() bfloat16 tests
    test_randn_bfloat16_dtype()
    test_randn_bfloat16_nonzero()

    print("All bfloat16 dtype guard tests passed!")
