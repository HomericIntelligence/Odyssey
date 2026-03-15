# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_shape.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for ExTensor shape manipulation: reshape, squeeze, unsqueeze, expand_dims, flatten.

Split from test_shape.mojo per ADR-009 (≤10 fn test_ per file).
"""

# Import ExTensor and operations
from shared.core.extensor import ExTensor, zeros, ones, full, arange
from shared.core.shape import reshape, squeeze, unsqueeze, expand_dims, flatten, ravel

# Import test helpers
from tests.shared.conftest import (
    assert_dtype,
    assert_numel,
    assert_dim,
    assert_value_at,
    assert_all_values,
)


# ============================================================================
# Test reshape()
# ============================================================================


fn test_reshape_valid() raises:
    """Test reshaping to compatible size."""
    var shape_orig = List[Int]()
    shape_orig.append(12)
    var a = arange(0.0, 12.0, 1.0, DType.float32)  # 12 elements
    var new_shape = List[Int]()
    new_shape.append(3)
    new_shape.append(4)
    var b = reshape(a, new_shape)

    assert_dim(b, 2, "Reshaped tensor should be 2D")
    assert_numel(b, 12, "Reshaped tensor should have same number of elements")


fn test_reshape_invalid_size() raises:
    """Test that reshape with incompatible size raises error."""
    var a = arange(0.0, 12.0, 1.0, DType.float32)  # 12 elements
    var new_shape = List[Int]()
    new_shape.append(3)
    new_shape.append(5)  # 15 elements, incompatible with 12

    var error_raised = False
    try:
        var b = reshape(a, new_shape)
        _ = b  # Suppress unused warning
    except e:
        error_raised = True
        var error_msg = String(e)
        # Verify error message mentions element count mismatch
        if (
            "element count mismatch" not in error_msg
            and "reshape" not in error_msg.lower()
        ):
            raise Error(
                "Error message should mention reshape or element count mismatch"
            )

    if not error_raised:
        raise Error("reshape with incompatible size should raise error")


fn test_reshape_infer_dimension() raises:
    """Test reshape with inferred dimension (-1)."""
    var shape = List[Int]()
    shape.append(12)
    var a = arange(0.0, 12.0, 1.0, DType.float32)
    var new_shape = List[Int]()
    new_shape.append(3)
    new_shape.append(-1)  # Infer: should be 4
    var b = reshape(a, new_shape)

    assert_dim(b, 2, "Should be 2D")
    assert_numel(b, 12, "Should have 12 elements")


# ============================================================================
# Test squeeze()
# ============================================================================


fn test_squeeze_all_dims() raises:
    """Test removing all size-1 dimensions."""
    var shape = List[Int]()
    shape.append(1)
    shape.append(3)
    shape.append(1)
    shape.append(4)
    var a = ones(shape, DType.float32)  # Shape (1, 3, 1, 4)
    var b = squeeze(a)

    # Result should be (3, 4)
    assert_dim(b, 2, "Should remove all size-1 dims")
    assert_numel(b, 12, "Should have 12 elements")


fn test_squeeze_specific_dim() raises:
    """Test removing specific size-1 dimension."""
    var shape = List[Int]()
    shape.append(1)
    shape.append(3)
    shape.append(4)
    var a = ones(shape, DType.float32)  # Shape (1, 3, 4)
    var b = squeeze(a, axis=0)

    # Result should be (3, 4)
    assert_dim(b, 2, "Should remove dim 0")


# ============================================================================
# Test unsqueeze() / expand_dims()
# ============================================================================


fn test_unsqueeze_add_dim() raises:
    """Test adding a size-1 dimension."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var a = ones(shape, DType.float32)  # Shape (3, 4)
    var b = unsqueeze(a, axis=0)

    # Result should be (1, 3, 4)
    assert_dim(b, 3, "Should add dimension")
    assert_numel(b, 12, "Should have same elements")


fn test_expand_dims_at_end() raises:
    """Test adding dimension at end."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var a = ones(shape, DType.float32)
    var b = expand_dims(a, axis=-1)

    # Result should be (3, 4, 1)
    assert_dim(b, 3, "Should add trailing dimension")


# ============================================================================
# Test flatten() / ravel()
# ============================================================================


fn test_flatten_c_order() raises:
    """Test flattening tensor to 1D (C order)."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var a = arange(0.0, 12.0, 1.0, DType.float32)
    var b = flatten(a)

    assert_dim(b, 1, "Flattened tensor should be 1D")
    assert_numel(b, 12, "Should have 12 elements")


fn test_ravel_view() raises:
    """Test ravel (should return view if possible)."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var a = ones(shape, DType.float32)
    var b = ravel(a)

    # Currently returns a 1D copy (view semantics deferred to future work)
    assert_dim(b, 1, "Ravel should be 1D")


# ============================================================================
# Main test runner
# ============================================================================


fn main() raises:
    """Run shape manipulation tests part 1 (reshape, squeeze, unsqueeze, flatten).
    """
    print("Running ExTensor shape manipulation tests (part 1)...")

    # reshape() tests
    print("  Testing reshape()...")
    test_reshape_valid()
    test_reshape_invalid_size()
    test_reshape_infer_dimension()

    # squeeze() tests
    print("  Testing squeeze()...")
    test_squeeze_all_dims()
    test_squeeze_specific_dim()

    # unsqueeze() / expand_dims() tests
    print("  Testing unsqueeze() / expand_dims()...")
    test_unsqueeze_add_dim()
    test_expand_dims_at_end()

    # flatten() / ravel() tests
    print("  Testing flatten() / ravel()...")
    test_flatten_c_order()
    test_ravel_view()

    print("All shape manipulation tests (part 1) completed!")
