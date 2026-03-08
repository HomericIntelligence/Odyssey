# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_shape.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for ExTensor shape manipulation: flatten_to_2d additional cases.

Split from test_shape.mojo per ADR-009 (≤10 fn test_ per file).
"""

# Import ExTensor and operations
from shared.core import (
    ExTensor,
    ones,
    flatten_to_2d,
)

# Import test helpers
from tests.shared.conftest import (
    assert_dtype,
    assert_numel,
    assert_dim,
    assert_value_at,
    assert_all_values,
)


# ============================================================================
# Test flatten_to_2d() - additional cases
# ============================================================================


fn test_flatten_to_2d_single_batch() raises:
    """Test flatten_to_2d with batch size 1."""
    var shape: List[Int] = [1, 64, 7, 7]
    var a = ones(shape, DType.float32)

    var b = flatten_to_2d(a)

    var out_shape = b.shape()
    if out_shape[0] != 1:
        raise Error("Batch dimension should be 1, got " + String(out_shape[0]))
    if out_shape[1] != 3136:
        raise Error(
            "Flattened dimension should be 3136 (64*7*7), got "
            + String(out_shape[1])
        )


fn test_flatten_to_2d_preserves_dtype() raises:
    """Test that flatten_to_2d preserves dtype."""
    var shape: List[Int] = [2, 3, 4, 4]
    var a = ones(shape, DType.float64)

    var b = flatten_to_2d(a)

    if b.dtype() != DType.float64:
        raise Error("flatten_to_2d should preserve dtype")


# ============================================================================
# Main test runner
# ============================================================================


fn main() raises:
    """Run shape manipulation tests part 4 (flatten_to_2d additional cases)."""
    print("Running ExTensor shape manipulation tests (part 4)...")

    # flatten_to_2d() additional tests
    print("  Testing flatten_to_2d() additional cases...")
    test_flatten_to_2d_single_batch()
    test_flatten_to_2d_preserves_dtype()

    print("All shape manipulation tests (part 4) completed!")
