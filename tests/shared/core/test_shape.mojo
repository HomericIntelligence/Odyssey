"""Tests for ExTensor shape manipulation operations.

Tests shape manipulation including reshape, squeeze, unsqueeze, expand_dims,
flatten, ravel, concatenate, stack, split, tile, repeat, broadcast_to, permute.
"""

# Import ExTensor and operations
from shared.core import (
    ExTensor,
    zeros,
    ones,
    full,
    arange,
    reshape,
    squeeze,
    unsqueeze,
    expand_dims,
    flatten,
    ravel,
    concatenate,
    stack,
    flatten_to_2d,
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
# Test concatenate()
# ============================================================================


fn test_concatenate_axis_0() raises:
    """Test concatenating along axis 0."""
    var shape_a = List[Int]()
    shape_a.append(2)
    shape_a.append(3)
    var shape_b = List[Int]()
    shape_b.append(3)
    shape_b.append(3)

    var a = ones(shape_a, DType.float32)  # 2x3
    var b = full(shape_b, 2.0, DType.float32)  # 3x3

    var tensors: List[ExTensor] = []
    tensors.append(a)
    tensors.append(b)
    var c = concatenate(tensors, axis=0)

    # Result should be 5x3 (2+3 rows, 3 cols)
    assert_dim(c, 2, "Concatenated tensor should be 2D")
    assert_numel(c, 15, "Should have 15 elements (5*3)")


fn test_concatenate_axis_1() raises:
    """Test concatenating along axis 1."""
    var shape_a = List[Int]()
    shape_a.append(3)
    shape_a.append(2)
    var shape_b = List[Int]()
    shape_b.append(3)
    shape_b.append(4)

    var a = ones(shape_a, DType.float32)  # 3x2
    var b = full(shape_b, 2.0, DType.float32)  # 3x4

    var tensors: List[ExTensor] = []
    tensors.append(a)
    tensors.append(b)
    var c = concatenate(tensors, axis=1)

    # Result should be 3x6 (3 rows, 2+4 cols)
    assert_numel(c, 18, "Should have 18 elements (3*6)")


# ============================================================================
# Test stack()
# ============================================================================


fn test_stack_new_axis() raises:
    """Test stacking tensors along new axis."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)

    var a = ones(shape, DType.float32)  # 2x3
    var b = full(shape, 2.0, DType.float32)  # 2x3

    var tensors: List[ExTensor] = []
    tensors.append(a)
    tensors.append(b)
    var c = stack(tensors, axis=0)

    # Result should be 2x2x3 (stacked along new axis 0)
    assert_dim(c, 3, "Stacked tensor should be 3D")
    assert_numel(c, 12, "Should have 12 elements (2*2*3)")


fn test_stack_axis_1() raises:
    """Test stacking along axis 1."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)

    var a = ones(shape, DType.float32)
    var b = full(shape, 2.0, DType.float32)

    var tensors: List[ExTensor] = []
    tensors.append(a)
    tensors.append(b)
    var c = stack(tensors, axis=1)

    # Result should be 2x2x3 (stacked along axis 1)
    assert_dim(c, 3, "Should be 3D")


# ============================================================================
# Test split()
# ============================================================================


fn test_split_equal() raises:
    """Test splitting into equal parts."""
    from shared.core import split

    var a = arange(0.0, 12.0, 1.0, DType.float32)
    var parts = split(a, 3)

    # Should give 3 tensors of size 4 each
    if len(parts) != 3:
        raise Error("Should split into 3 parts")
    for i in range(3):
        assert_numel(parts[i], 4, "Each part should have 4 elements")


fn test_split_unequal() raises:
    """Test splitting into unequal parts."""
    from shared.core import split_with_indices

    var a = arange(0.0, 10.0, 1.0, DType.float32)
    var indices = List[Int]()
    indices.append(3)
    indices.append(7)
    var parts = split_with_indices(a, indices)

    # Should give 3 tensors of sizes 3, 4, 3
    if len(parts) != 3:
        raise Error("Should split into 3 parts")
    assert_numel(parts[0], 3, "First part should have 3 elements")
    assert_numel(parts[1], 4, "Second part should have 4 elements")
    assert_numel(parts[2], 3, "Third part should have 3 elements")


# ============================================================================
# Test tile()
# ============================================================================


fn test_tile_1d() raises:
    """Test tiling 1D tensor."""
    from shared.core import tile

    var a = arange(0.0, 3.0, 1.0, DType.float32)  # [0, 1, 2]
    var reps = List[Int]()
    reps.append(3)
    var b = tile(a, reps)

    # Result: [0, 1, 2, 0, 1, 2, 0, 1, 2] (9 elements)
    assert_numel(b, 9, "Tiled tensor should have 9 elements")


fn test_tile_multidim() raises:
    """Test tiling with multi-dimensional repetitions."""
    from shared.core import tile

    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var a = ones(shape, DType.float32)  # 2x3
    var reps = List[Int]()
    reps.append(2)
    reps.append(3)
    var b = tile(a, reps)

    # Result should be 4x9 (2*2 rows, 3*3 cols)
    assert_numel(b, 36, "Should have 36 elements (4*9)")


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
    var a = arange(0.0, 12.0, 1.0, DType.float64)
    var new_shape = List[Int]()
    new_shape.append(3)
    new_shape.append(4)
    var b = reshape(a, new_shape)

    assert_dtype(b, DType.float64, "Reshape should preserve dtype")


# ============================================================================
# Test flatten_to_2d()
# ============================================================================


fn test_flatten_to_2d_basic() raises:
    """Test basic flatten_to_2d functionality."""
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
    """Run all shape manipulation tests."""
    print("Running ExTensor shape manipulation tests...")

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

    # concatenate() tests
    print("  Testing concatenate()...")
    test_concatenate_axis_0()
    test_concatenate_axis_1()

    # stack() tests
    print("  Testing stack()...")
    test_stack_new_axis()
    test_stack_axis_1()

    # split() tests
    print("  Testing split()...")
    test_split_equal()
    test_split_unequal()

    # tile() tests
    print("  Testing tile()...")
    test_tile_1d()
    test_tile_multidim()

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

    # flatten_to_2d() tests
    print("  Testing flatten_to_2d()...")
    test_flatten_to_2d_basic()
    test_flatten_to_2d_single_batch()
    test_flatten_to_2d_preserves_dtype()

    print("All shape manipulation tests completed!")
