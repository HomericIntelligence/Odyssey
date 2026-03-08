# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_shape.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for ExTensor shape manipulation: repeat, broadcast_to, permute, dtype preservation.

Split from test_shape.mojo per ADR-009 (≤10 fn test_ per file).
"""

# Import ExTensor and operations
from shared.core import (
    ExTensor,
    zeros,
    ones,
    full,
    arange,
    broadcast_to,
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
# Test repeat()
# ============================================================================


fn test_repeat_elements() raises:
    """Test repeating each element."""
    from shared.core import repeat

    var a = arange(0.0, 3.0, 1.0, DType.float32)  # [0, 1, 2]
    var b = repeat(a, 2)

    # Result: [0, 0, 1, 1, 2, 2] (6 elements)
    assert_numel(b, 6, "Repeated tensor should have 6 elements")


fn test_repeat_axis() raises:
    """Test repeating along specific axis."""
    from shared.core import repeat

    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var a = ones(shape, DType.float32)  # 2x3
    var b = repeat(a, 2, axis=0)

    # Result should be 4x3 (each row repeated twice)
    assert_numel(b, 12, "Should have 12 elements (4*3)")


# ============================================================================
# Test broadcast_to()
# ============================================================================


fn test_broadcast_to_compatible() raises:
    """Test broadcasting to compatible shape."""
    var a = arange(0.0, 3.0, 1.0, DType.float32)  # Shape (3,)
    var target_shape = List[Int]()
    target_shape.append(4)
    target_shape.append(3)
    var b = broadcast_to(a, target_shape)

    # Result should be 4x3 (broadcasting (3,) to (4,3))
    assert_dim(b, 2, "Broadcasted tensor should be 2D")
    assert_numel(b, 12, "Should have 12 elements")


fn test_broadcast_to_incompatible() raises:
    """Test that broadcasting to incompatible shape raises error."""
    var shape_orig = List[Int]()
    shape_orig.append(3)
    var a = arange(0.0, 3.0, 1.0, DType.float32)  # Shape (3,)
    var target_shape = List[Int]()
    target_shape.append(5)  # Incompatible: 3 != 5 and neither is 1

    var error_raised = False
    try:
        var b = broadcast_to(a, target_shape)
        _ = b  # Suppress unused warning
    except e:
        error_raised = True
        var error_msg = String(e)
        # Verify error message mentions broadcast compatibility
        if (
            "broadcast" not in error_msg.lower()
            and "compatible" not in error_msg.lower()
        ):
            raise Error("Error message should mention broadcast compatibility")

    if not error_raised:
        raise Error("broadcast_to with incompatible shape should raise error")


# ============================================================================
# Test permute()
# ============================================================================


fn test_permute_axes() raises:
    """Test permuting axes (similar to transpose with axes)."""
    from shared.core import permute

    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)
    var a = ones(shape, DType.float32)  # Shape (2, 3, 4)
    var dims = List[Int]()
    dims.append(2)
    dims.append(0)
    dims.append(1)
    var b = permute(a, dims)

    # Result should be (4, 2, 3)
    assert_dim(b, 3, "Should still be 3D")
    assert_numel(b, 24, "Should have same elements")


# ============================================================================
# Test dtype preservation
# ============================================================================


fn test_reshape_preserves_dtype() raises:
    """Test that reshape preserves dtype."""
    from shared.core import reshape

    var a = arange(0.0, 12.0, 1.0, DType.float64)
    var new_shape = List[Int]()
    new_shape.append(3)
    new_shape.append(4)
    var b = reshape(a, new_shape)

    assert_dtype(b, DType.float64, "Reshape should preserve dtype")


# ============================================================================
# Test flatten_to_2d() - basic
# ============================================================================


fn test_flatten_to_2d_basic() raises:
    """Test basic flatten_to_2d functionality."""
    from shared.core import flatten_to_2d

    # Create 4D tensor: (batch=2, channels=3, height=4, width=4)
    var shape: List[Int] = [2, 3, 4, 4]
    var a = ones(shape, DType.float32)

    var b = flatten_to_2d(a)

    # Should be (2, 48) where 48 = 3 * 4 * 4
    assert_dim(b, 2, "flatten_to_2d should produce 2D tensor")
    assert_numel(b, 96, "flatten_to_2d should preserve element count")

    var out_shape = b.shape()
    if out_shape[0] != 2:
        raise Error(
            "Batch dimension should be preserved (expected 2, got "
            + String(out_shape[0])
            + ")"
        )
    if out_shape[1] != 48:
        raise Error(
            "Flattened dimension should be 48 (3*4*4), got "
            + String(out_shape[1])
        )


# ============================================================================
# Main test runner
# ============================================================================


fn main() raises:
    """Run shape manipulation tests part 3 (repeat, broadcast_to, permute, dtype, flatten_to_2d)."""
    print("Running ExTensor shape manipulation tests (part 3)...")

    # repeat() tests
    print("  Testing repeat()...")
    test_repeat_elements()
    test_repeat_axis()

    # broadcast_to() tests
    print("  Testing broadcast_to()...")
    test_broadcast_to_compatible()
    test_broadcast_to_incompatible()

    # permute() tests
    print("  Testing permute()...")
    test_permute_axes()

    # Dtype preservation
    print("  Testing dtype preservation...")
    test_reshape_preserves_dtype()

    # flatten_to_2d() tests (basic)
    print("  Testing flatten_to_2d() basic...")
    test_flatten_to_2d_basic()

    print("All shape manipulation tests (part 3) completed!")
