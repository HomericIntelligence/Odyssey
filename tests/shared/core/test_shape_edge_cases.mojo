"""Tests for reshape edge cases.

Tests edge cases for reshape operations including:
- Reshape to/from scalars
- Reshape with empty tensors
- Reshape between dimensions
"""


from shared.tensor.any_tensor import AnyTensor, zeros, ones, full, arange
from shared.core.shape import reshape, squeeze, unsqueeze, concatenate, stack
from tests.shared.conftest import (
    assert_dtype,
    assert_numel,
    assert_dim,
    assert_value_at,
    assert_all_values,
    assert_shape,
)


fn test_reshape_to_scalar() raises:
    """Reshape [1] to [] (scalar)."""
    var shape_from = List[Int]()
    shape_from.append(1)
    var t = full(shape_from, 42.0, DType.float32)

    var shape_to = List[Int]()
    var result = reshape(t, shape_to)

    assert_dim(result, 0, "Result should be 0D scalar")
    assert_numel(result, 1, "Scalar should have 1 element")
    assert_value_at(result, 0, 42.0, 1e-6, "Value should be preserved")


fn test_reshape_from_scalar() raises:
    """Reshape [] to [1, 1, 1]."""
    var shape_from = List[Int]()
    var t = full(shape_from, 42.0, DType.float32)

    var shape_to = List[Int]()
    shape_to.append(1)
    shape_to.append(1)
    shape_to.append(1)
    var result = reshape(t, shape_to)

    assert_dim(result, 3, "Result should be 3D")
    assert_numel(result, 1, "Should have 1 element")
    assert_value_at(result, 0, 42.0, 1e-6, "Value should be preserved")


fn test_reshape_1d_to_2d() raises:
    """Reshape [6] to [2, 3]."""
    var shape_from = List[Int]()
    shape_from.append(6)
    var t = arange(0, 6, 1, DType.float32)

    var shape_to = List[Int]()
    shape_to.append(2)
    shape_to.append(3)
    var result = reshape(t, shape_to)

    assert_dim(result, 2, "Result should be 2D")
    assert_numel(result, 6, "Should have 6 elements")


fn test_reshape_2d_to_1d() raises:
    """Reshape [2, 3] to [6]."""
    var shape_from = List[Int]()
    shape_from.append(2)
    shape_from.append(3)
    var t = ones(shape_from, DType.float32)

    var shape_to = List[Int]()
    shape_to.append(6)
    var result = reshape(t, shape_to)

    assert_dim(result, 1, "Result should be 1D")
    assert_numel(result, 6, "Should have 6 elements")


fn test_reshape_preserve_size() raises:
    """Reshape [2, 3, 4] to [8, 3]."""
    var shape_from = List[Int]()
    shape_from.append(2)
    shape_from.append(3)
    shape_from.append(4)
    var t = ones(shape_from, DType.float32)

    var shape_to = List[Int]()
    shape_to.append(8)
    shape_to.append(3)
    var result = reshape(t, shape_to)

    assert_numel(result, 24, "Should preserve element count (2*3*4=24)")


fn test_reshape_empty_tensor() raises:
    """Reshape [0] to [0, 3] (empty tensor)."""
    var shape_from = List[Int]()
    shape_from.append(0)
    var t = zeros(shape_from, DType.float32)

    var shape_to = List[Int]()
    shape_to.append(0)
    shape_to.append(3)
    var result = reshape(t, shape_to)

    assert_dim(result, 2, "Result should be 2D")
    assert_numel(result, 0, "Should have 0 elements")


fn test_squeeze_size_one_dim() raises:
    """Squeeze [1, 3, 1, 4] -> [3, 4]."""
    var shape = List[Int]()
    shape.append(1)
    shape.append(3)
    shape.append(1)
    shape.append(4)
    var t = ones(shape, DType.float32)

    var result = squeeze(t)

    assert_dim(result, 2, "Result should be 2D")
    assert_numel(result, 12, "Should have 3*4=12 elements")


fn test_squeeze_all_size_one() raises:
    """Squeeze [1, 1, 1] -> []."""
    var shape = List[Int]()
    shape.append(1)
    shape.append(1)
    shape.append(1)
    var t = full(shape, 42.0, DType.float32)

    var result = squeeze(t)

    assert_dim(result, 0, "Result should be 0D scalar")
    assert_numel(result, 1, "Should have 1 element")
    assert_value_at(result, 0, 42.0, 1e-6, "Value should be preserved")


fn test_squeeze_no_size_one() raises:
    """Squeeze [2, 3, 4] with no size-1 dims."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    shape.append(4)
    var t = ones(shape, DType.float32)

    var result = squeeze(t)

    # Should remain unchanged if no size-1 dimensions
    assert_dim(result, 3, "Result should remain 3D")
    assert_numel(result, 24, "Should preserve elements")


fn test_squeeze_1d_with_size_one() raises:
    """Squeeze [1] -> []."""
    var shape = List[Int]()
    shape.append(1)
    var t = full(shape, 99.0, DType.float32)

    var result = squeeze(t)

    assert_dim(result, 0, "Result should be 0D scalar")
    assert_value_at(result, 0, 99.0, 1e-6, "Value preserved")


fn test_unsqueeze_scalar() raises:
    """Unsqueeze [] at dim 0 -> [1]."""
    var shape = List[Int]()
    var t = full(shape, 42.0, DType.float32)

    var result = unsqueeze(t, 0)

    assert_dim(result, 1, "Result should be 1D")
    assert_numel(result, 1, "Should have 1 element")


fn test_unsqueeze_1d() raises:
    """Unsqueeze [3] at dim 0 -> [1, 3]."""
    var shape = List[Int]()
    shape.append(3)
    var t = full(shape, 2.0, DType.float32)

    var result = unsqueeze(t, 0)

    assert_dim(result, 2, "Result should be 2D")
    assert_numel(result, 3, "Should have 3 elements")


fn test_unsqueeze_1d_at_end() raises:
    """Unsqueeze [3] at dim 1 -> [3, 1]."""
    var shape = List[Int]()
    shape.append(3)
    var t = ones(shape, DType.float32)

    var result = unsqueeze(t, 1)

    assert_dim(result, 2, "Result should be 2D")
    assert_numel(result, 3, "Should have 3 elements")


fn test_unsqueeze_2d() raises:
    """Unsqueeze [2, 3] at dim 1 -> [2, 1, 3]."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var t = ones(shape, DType.float32)

    var result = unsqueeze(t, 1)

    assert_dim(result, 3, "Result should be 3D")
    assert_numel(result, 6, "Should have 6 elements")


fn test_concatenate_single_tensor() raises:
    """Concatenate single tensor returns copy."""
    var shape = List[Int]()
    shape.append(3)
    shape.append(4)
    var t = ones(shape, DType.float32)

    var tensors = List[AnyTensor]()
    tensors.append(t)
    var result = concatenate(tensors, 0)

    assert_dim(result, 2, "Result should be 2D")
    assert_numel(result, 12, "Should have 12 elements")


fn test_concatenate_along_axis_0() raises:
    """Concatenate [2,3] + [2,3] along axis 0 -> [4,3]."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)

    var t1 = ones(shape, DType.float32)
    var t2 = full(shape, 2.0, DType.float32)

    var tensors = List[AnyTensor]()
    tensors.append(t1)
    tensors.append(t2)
    var result = concatenate(tensors, 0)

    assert_dim(result, 2, "Result should be 2D")
    # Result should be [4, 3]


fn test_concatenate_along_axis_1() raises:
    """Concatenate [2,3] + [2,4] along axis 1 -> [2,7]."""
    var shape_a = List[Int]()
    shape_a.append(2)
    shape_a.append(3)

    var shape_b = List[Int]()
    shape_b.append(2)
    shape_b.append(4)

    var t1 = ones(shape_a, DType.float32)
    var t2 = full(shape_b, 2.0, DType.float32)

    var tensors = List[AnyTensor]()
    tensors.append(t1)
    tensors.append(t2)
    var result = concatenate(tensors, 1)

    assert_dim(result, 2, "Result should be 2D")
    # Result should be [2, 7]


fn test_concatenate_with_empty() raises:
    """Concatenate [3, 0] + [3, 4] along axis 1 -> [3, 4]."""
    var shape_a = List[Int]()
    shape_a.append(3)
    shape_a.append(0)

    var shape_b = List[Int]()
    shape_b.append(3)
    shape_b.append(4)

    var t1 = zeros(shape_a, DType.float32)
    var t2 = ones(shape_b, DType.float32)

    var tensors = List[AnyTensor]()
    tensors.append(t1)
    tensors.append(t2)
    var result = concatenate(tensors, 1)

    assert_dim(result, 2, "Result should be 2D")


fn test_concatenate_1d_tensors() raises:
    """Concatenate 1D tensors."""
    var shape = List[Int]()
    shape.append(3)

    var t1 = full(shape, 1.0, DType.float32)
    var t2 = full(shape, 2.0, DType.float32)

    var tensors = List[AnyTensor]()
    tensors.append(t1)
    tensors.append(t2)
    var result = concatenate(tensors, 0)

    assert_dim(result, 1, "Result should be 1D")
    assert_numel(result, 6, "Should have 6 elements")


fn test_stack_single_tensor() raises:
    """Stack single tensor adds dimension."""
    var shape = List[Int]()
    shape.append(3)
    var t = ones(shape, DType.float32)

    var tensors = List[AnyTensor]()
    tensors.append(t)
    var result = stack(tensors, 0)

    # Stack should add dimension: [3] -> [1, 3]
    assert_dim(result, 2, "Result should be 2D after stacking")


fn test_stack_2d_tensors() raises:
    """Stack 2D tensors."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)

    var t1 = ones(shape, DType.float32)
    var t2 = full(shape, 2.0, DType.float32)

    var tensors = List[AnyTensor]()
    tensors.append(t1)
    tensors.append(t2)
    var result = stack(tensors, 0)

    # [2,3] stacked at axis 0 -> [2, 2, 3]
    assert_dim(result, 3, "Result should be 3D")


fn test_stack_along_different_axis() raises:
    """Stack along different axis."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)

    var t1 = ones(shape, DType.float32)
    var t2 = full(shape, 2.0, DType.float32)

    var tensors = List[AnyTensor]()
    tensors.append(t1)
    tensors.append(t2)
    var result = stack(tensors, 1)

    # [2,3] stacked at axis 1 -> [2, 2, 3]
    assert_dim(result, 3, "Result should be 3D")


fn test_reshape_preserves_numel() raises:
    """Reshape must preserve total number of elements."""
    var shape_from = List[Int]()
    shape_from.append(3)
    shape_from.append(4)
    shape_from.append(5)
    var t = ones(shape_from, DType.float32)

    var shape_to = List[Int]()
    shape_to.append(6)
    shape_to.append(10)
    var result = reshape(t, shape_to)

    assert_numel(result, 60, "Should preserve 3*4*5=60 elements")


fn test_squeeze_preserves_numel() raises:
    """Squeeze must preserve total number of elements."""
    var shape = List[Int]()
    shape.append(1)
    shape.append(5)
    shape.append(1)
    var t = ones(shape, DType.float32)

    var result = squeeze(t)

    assert_numel(result, 5, "Should preserve 5 elements")


fn test_unsqueeze_preserves_numel() raises:
    """Unsqueeze must preserve total number of elements."""
    var shape = List[Int]()
    shape.append(3)
    var t = ones(shape, DType.float32)

    var result = unsqueeze(t, 0)

    assert_numel(result, 3, "Should preserve 3 elements")


fn main() raises:
    """Run all test_shape_edge_cases tests."""
    print("Running test_shape_edge_cases tests...")

    test_reshape_to_scalar()
    print("✓ test_reshape_to_scalar")

    test_reshape_from_scalar()
    print("✓ test_reshape_from_scalar")

    test_reshape_1d_to_2d()
    print("✓ test_reshape_1d_to_2d")

    test_reshape_2d_to_1d()
    print("✓ test_reshape_2d_to_1d")

    test_reshape_preserve_size()
    print("✓ test_reshape_preserve_size")

    test_reshape_empty_tensor()
    print("✓ test_reshape_empty_tensor")

    test_squeeze_size_one_dim()
    print("✓ test_squeeze_size_one_dim")

    test_squeeze_all_size_one()
    print("✓ test_squeeze_all_size_one")

    test_squeeze_no_size_one()
    print("✓ test_squeeze_no_size_one")

    test_squeeze_1d_with_size_one()
    print("✓ test_squeeze_1d_with_size_one")

    test_unsqueeze_scalar()
    print("✓ test_unsqueeze_scalar")

    test_unsqueeze_1d()
    print("✓ test_unsqueeze_1d")

    test_unsqueeze_1d_at_end()
    print("✓ test_unsqueeze_1d_at_end")

    test_unsqueeze_2d()
    print("✓ test_unsqueeze_2d")

    test_concatenate_single_tensor()
    print("✓ test_concatenate_single_tensor")

    test_concatenate_along_axis_0()
    print("✓ test_concatenate_along_axis_0")

    test_concatenate_along_axis_1()
    print("✓ test_concatenate_along_axis_1")

    test_concatenate_with_empty()
    print("✓ test_concatenate_with_empty")

    test_concatenate_1d_tensors()
    print("✓ test_concatenate_1d_tensors")

    test_stack_single_tensor()
    print("✓ test_stack_single_tensor")

    test_stack_2d_tensors()
    print("✓ test_stack_2d_tensors")

    test_stack_along_different_axis()
    print("✓ test_stack_along_different_axis")

    test_reshape_preserves_numel()
    print("✓ test_reshape_preserves_numel")

    test_squeeze_preserves_numel()
    print("✓ test_squeeze_preserves_numel")

    test_unsqueeze_preserves_numel()
    print("✓ test_unsqueeze_preserves_numel")

    print("\nAll test_shape_edge_cases tests passed!")
