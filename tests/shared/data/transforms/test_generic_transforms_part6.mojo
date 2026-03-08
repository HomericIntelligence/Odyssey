# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_generic_transforms.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for generic data transformation utilities - Part 6.

Tests integration scenarios and edge cases.
"""

from shared.core.extensor import ExTensor
from tests.shared.conftest import (
    assert_true,
    assert_equal,
    assert_false,
    assert_almost_equal,
    assert_greater,
    assert_less,
    TestFixtures,
)
from shared.data.transforms import Transform
from shared.data.generic_transforms import (
    IdentityTransform,
    LambdaTransform,
    ConditionalTransform,
    ClampTransform,
    DebugTransform,
    BatchTransform,
    ToFloat32,
    ToInt32,
    SequentialTransform,
    AnyTransform,
)

# Type comptime for test convenience
comptime Tensor = ExTensor


# ============================================================================
# Integration Tests
# ============================================================================


fn test_integration_preprocessing_pipeline() raises:
    """Test typical preprocessing pipeline."""
    var values = List[Float32]()
    values.append(-5.0)
    values.append(0.0)
    values.append(5.0)
    values.append(10.0)
    values.append(15.0)
    var data = ExTensor(values^)

    fn normalize(value: Float32) -> Float32:
        # Scale to [0, 1]
        return (value + 5.0) / 20.0

    var transforms: List[AnyTransform] = []
    transforms.append(AnyTransform(LambdaTransform(normalize)))  # Normalize
    transforms.append(AnyTransform(ClampTransform(0.0, 1.0)))  # Ensure bounds
    transforms.append(AnyTransform(DebugTransform("pipeline")))  # Debug

    var pipeline = SequentialTransform(transforms^)
    var result = pipeline(data)

    # Check values are properly normalized and clamped
    assert_almost_equal(result[0], 0.0)
    assert_almost_equal(result[1], 0.25)
    assert_almost_equal(result[2], 0.5)
    assert_almost_equal(result[3], 0.75)
    assert_almost_equal(result[4], 1.0)


fn test_integration_conditional_augmentation() raises:
    """Test conditional augmentation pipeline."""
    var large_values = List[Float32]()
    large_values.append(1.0)
    large_values.append(2.0)
    large_values.append(3.0)
    large_values.append(4.0)
    var large_data = ExTensor(large_values^)
    var small_values = List[Float32]()
    small_values.append(1.0)
    small_values.append(2.0)
    var small_data = ExTensor(small_values^)

    fn is_large_enough(tensor: ExTensor) raises -> Bool:
        return tensor.num_elements() >= 3

    fn augment(value: Float32) -> Float32:
        return value * 1.5

    var base_transform = LambdaTransform(augment)
    var conditional = ConditionalTransform[LambdaTransform](
        is_large_enough, base_transform^
    )

    var result_large = conditional(large_data)
    var result_small = conditional(small_data)

    # Large should be augmented
    assert_almost_equal(result_large[0], 1.5)
    assert_almost_equal(result_large[1], 3.0)

    # Small should NOT be augmented
    assert_almost_equal(result_small[0], 1.0)
    assert_almost_equal(result_small[1], 2.0)


fn test_integration_batch_preprocessing() raises:
    """Test batch preprocessing pipeline."""
    var values1 = List[Float32]()
    values1.append(100.0)
    values1.append(200.0)
    var values2 = List[Float32]()
    values2.append(300.0)
    values2.append(400.0)

    var batch: List[Tensor] = []
    batch.append(ExTensor(values1^))
    batch.append(ExTensor(values2^))

    fn scale_down(value: Float32) -> Float32:
        return value / 100.0

    var transforms: List[AnyTransform] = []
    transforms.append(AnyTransform(LambdaTransform(scale_down)))
    transforms.append(AnyTransform(ClampTransform(0.0, 5.0)))

    var pipeline = SequentialTransform(transforms^)
    var batch_transform = BatchTransform(AnyTransform(pipeline^))

    var results = batch_transform(batch)

    # First batch item
    assert_almost_equal(results[0][0], 1.0)
    assert_almost_equal(results[0][1], 2.0)

    # Second batch item
    assert_almost_equal(results[1][0], 3.0)
    assert_almost_equal(results[1][1], 4.0)


fn test_integration_type_conversion_pipeline() raises:
    """Test type conversion in pipeline."""
    var values = List[Float32]()
    values.append(1.9)
    values.append(2.5)
    values.append(3.1)
    var data = ExTensor(values^)

    var transforms: List[AnyTransform] = []
    transforms.append(AnyTransform(ToInt32()))  # Convert to int (truncate)
    transforms.append(AnyTransform(ToFloat32()))  # Convert back to float
    transforms.append(AnyTransform(ClampTransform(0.0, 3.0)))  # Clamp result

    var pipeline = SequentialTransform(transforms^)
    var result = pipeline(data)

    # Should truncate then clamp
    assert_almost_equal(result[0], 1.0)
    assert_almost_equal(result[1], 2.0)
    assert_almost_equal(result[2], 3.0)


# ============================================================================
# Edge Case Tests
# ============================================================================


fn test_edge_case_very_large_values() raises:
    """Test transforms with very large values."""
    var values = List[Float32]()
    values.append(1e6)
    values.append(1e7)
    values.append(1e8)
    var data = ExTensor(values^)
    var clamp = ClampTransform(0.0, 1e9)

    var result = clamp(data)

    assert_almost_equal(result[0], 1e6)
    assert_almost_equal(result[1], 1e7)
    assert_almost_equal(result[2], 1e8)


fn test_edge_case_very_small_values() raises:
    """Test transforms with very small values."""
    var values = List[Float32]()
    values.append(1e-6)
    values.append(1e-7)
    values.append(1e-8)
    var data = ExTensor(values^)
    var clamp = ClampTransform(1e-9, 1.0)

    var result = clamp(data)

    assert_almost_equal(result[0], 1e-6, tolerance=1e-9)
    assert_almost_equal(result[1], 1e-7, tolerance=1e-9)
    assert_almost_equal(result[2], 1e-8, tolerance=1e-9)


fn test_edge_case_all_zeros() raises:
    """Test transforms with all zeros."""
    var values = List[Float32]()
    values.append(0.0)
    values.append(0.0)
    values.append(0.0)
    var data = ExTensor(values^)

    fn add_one(value: Float32) -> Float32:
        return value + 1.0

    var transform = LambdaTransform(add_one)
    var result = transform(data)

    assert_almost_equal(result[0], 1.0)
    assert_almost_equal(result[1], 1.0)
    assert_almost_equal(result[2], 1.0)


fn test_edge_case_single_element() raises:
    """Test transforms with single element."""
    var values = List[Float32]()
    values.append(42.0)
    var data = ExTensor(values^)

    var transforms: List[AnyTransform] = []
    transforms.append(AnyTransform(ClampTransform(0.0, 100.0)))
    transforms.append(AnyTransform(DebugTransform("single")))

    var pipeline = SequentialTransform(transforms^)
    var result = pipeline(data)

    assert_equal(result.num_elements(), 1)
    assert_almost_equal(result[0], 42.0)


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run integration and edge case transform tests."""
    print("Running generic transform tests - Part 6...")
    print()

    # Integration tests
    print("Testing integration scenarios...")
    test_integration_preprocessing_pipeline()
    test_integration_conditional_augmentation()
    test_integration_batch_preprocessing()
    test_integration_type_conversion_pipeline()
    print("  ✓ 4 integration tests passed")

    # Edge case tests
    print("Testing edge cases...")
    test_edge_case_very_large_values()
    test_edge_case_very_small_values()
    test_edge_case_all_zeros()
    test_edge_case_single_element()
    print("  ✓ 4 edge case tests passed")

    print()
    print("✓ All 8 Part 6 generic transform tests passed!")
