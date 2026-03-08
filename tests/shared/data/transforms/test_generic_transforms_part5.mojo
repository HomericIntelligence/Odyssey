# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_generic_transforms.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for generic data transformation utilities - Part 5.

Tests second part of sequential transforms and batch transforms.
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
# Sequential Composition Tests (second part)
# ============================================================================


fn test_sequential_with_clamp() raises:
    """Test sequential including clamp transform."""
    var values = List[Float32]()
    values.append(0.5)
    values.append(1.5)
    values.append(2.5)
    var data = ExTensor(values^)

    fn double_fn(value: Float32) -> Float32:
        return value * 2.0

    var transforms: List[AnyTransform] = []
    transforms.append(
        AnyTransform(LambdaTransform(double_fn))
    )  # Double: 1.0, 3.0, 5.0
    transforms.append(AnyTransform(ClampTransform(0.0, 4.0)))  # Clamp to [0, 4]

    var sequential = SequentialTransform(transforms^)
    var result = sequential(data)

    assert_almost_equal(result[0], 1.0)  # 0.5*2 = 1.0
    assert_almost_equal(result[1], 3.0)  # 1.5*2 = 3.0
    assert_almost_equal(result[2], 4.0)  # 2.5*2 = 5.0, clamped to 4.0


fn test_sequential_deterministic() raises:
    """Test sequential produces same result on repeated calls."""
    var values = List[Float32]()
    values.append(1.0)
    values.append(2.0)
    values.append(3.0)
    var data = ExTensor(values^)

    fn triple(value: Float32) -> Float32:
        return value * 3.0

    var transforms: List[AnyTransform] = []
    transforms.append(AnyTransform(LambdaTransform(triple)))
    transforms.append(AnyTransform(ClampTransform(0.0, 5.0)))

    var sequential = SequentialTransform(transforms^)

    var result1 = sequential(data)
    var result2 = sequential(data)

    # Should be identical
    for i in range(result1.num_elements()):
        assert_almost_equal(result1[i], result2[i])


# ============================================================================
# Batch Transform Tests
# ============================================================================


fn test_batch_transform_basic() raises:
    """Test batch transform applies to multiple tensors."""
    var values1 = List[Float32]()
    values1.append(1.0)
    values1.append(2.0)
    var values2 = List[Float32]()
    values2.append(3.0)
    values2.append(4.0)
    var values3 = List[Float32]()
    values3.append(5.0)
    values3.append(6.0)

    var tensors: List[Tensor] = []
    tensors.append(ExTensor(values1^))
    tensors.append(ExTensor(values2^))
    tensors.append(ExTensor(values3^))

    fn double_fn(value: Float32) -> Float32:
        return value * 2.0

    var base_transform = LambdaTransform(double_fn)
    var batch = BatchTransform(AnyTransform(base_transform^))

    var results = batch(tensors)

    # Check first tensor
    assert_almost_equal(results[0][0], 2.0)
    assert_almost_equal(results[0][1], 4.0)

    # Check second tensor
    assert_almost_equal(results[1][0], 6.0)
    assert_almost_equal(results[1][1], 8.0)

    # Check third tensor
    assert_almost_equal(results[2][0], 10.0)
    assert_almost_equal(results[2][1], 12.0)


fn test_batch_transform_empty_list() raises:
    """Test batch transform with empty list."""
    var tensors: List[Tensor] = []

    fn double_fn(value: Float32) -> Float32:
        return value * 2.0

    var base_transform = LambdaTransform(double_fn)
    var batch = BatchTransform(AnyTransform(base_transform^))

    var results = batch(tensors)

    assert_equal(len(results), 0)


fn test_batch_transform_single_tensor() raises:
    """Test batch transform with single tensor."""
    var values = List[Float32]()
    values.append(1.0)
    values.append(2.0)
    values.append(3.0)

    var tensors: List[Tensor] = []
    tensors.append(ExTensor(values^))

    fn add_ten(value: Float32) -> Float32:
        return value + 10.0

    var base_transform = LambdaTransform(add_ten)
    var batch = BatchTransform(AnyTransform(base_transform^))

    var results = batch(tensors)

    assert_equal(len(results), 1)
    assert_almost_equal(results[0][0], 11.0)
    assert_almost_equal(results[0][1], 12.0)
    assert_almost_equal(results[0][2], 13.0)


fn test_batch_transform_different_sizes() raises:
    """Test batch transform with different sized tensors."""
    var values1 = List[Float32]()
    values1.append(1.0)
    var values2 = List[Float32]()
    values2.append(2.0)
    values2.append(3.0)
    var values3 = List[Float32]()
    values3.append(4.0)
    values3.append(5.0)
    values3.append(6.0)

    var tensors: List[Tensor] = []
    tensors.append(ExTensor(values1^))
    tensors.append(ExTensor(values2^))
    tensors.append(ExTensor(values3^))

    fn double_fn(value: Float32) -> Float32:
        return value * 2.0

    var base_transform = LambdaTransform(double_fn)
    var batch = BatchTransform(AnyTransform(base_transform^))

    var results = batch(tensors)

    # First tensor (size 1)
    assert_equal(results[0].num_elements(), 1)
    assert_almost_equal(results[0][0], 2.0)

    # Second tensor (size 2)
    assert_equal(results[1].num_elements(), 2)
    assert_almost_equal(results[1][0], 4.0)
    assert_almost_equal(results[1][1], 6.0)

    # Third tensor (size 3)
    assert_equal(results[2].num_elements(), 3)
    assert_almost_equal(results[2][0], 8.0)
    assert_almost_equal(results[2][1], 10.0)
    assert_almost_equal(results[2][2], 12.0)


fn test_batch_transform_with_clamp() raises:
    """Test batch transform with clamp."""
    var values1 = List[Float32]()
    values1.append(0.0)
    values1.append(5.0)
    values1.append(10.0)
    var values2 = List[Float32]()
    values2.append(15.0)
    values2.append(20.0)
    values2.append(25.0)

    var tensors: List[Tensor] = []
    tensors.append(ExTensor(values1^))
    tensors.append(ExTensor(values2^))

    var base_transform = ClampTransform(2.0, 18.0)
    var batch = BatchTransform(AnyTransform(base_transform^))

    var results = batch(tensors)

    # First tensor
    assert_almost_equal(results[0][0], 2.0)  # Clamped
    assert_almost_equal(results[0][1], 5.0)  # Unchanged
    assert_almost_equal(results[0][2], 10.0)  # Unchanged

    # Second tensor
    assert_almost_equal(results[1][0], 15.0)  # Unchanged
    assert_almost_equal(results[1][1], 18.0)  # Clamped
    assert_almost_equal(results[1][2], 18.0)  # Clamped


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run sequential (second part) and batch transform tests."""
    print("Running generic transform tests - Part 5...")
    print()

    # Sequential tests (second part)
    print("Testing SequentialTransform (second part)...")
    test_sequential_with_clamp()
    test_sequential_deterministic()
    print("  ✓ 2 sequential tests passed")

    # Batch tests
    print("Testing BatchTransform...")
    test_batch_transform_basic()
    test_batch_transform_empty_list()
    test_batch_transform_single_tensor()
    test_batch_transform_different_sizes()
    test_batch_transform_with_clamp()
    print("  ✓ 5 batch tests passed")

    print()
    print("✓ All 7 Part 5 generic transform tests passed!")
