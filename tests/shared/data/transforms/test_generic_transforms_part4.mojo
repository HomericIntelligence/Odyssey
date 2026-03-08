# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_generic_transforms.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for generic data transformation utilities - Part 4.

Tests type conversion transforms and first part of sequential transforms.
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
# Type Conversion Tests
# ============================================================================


fn test_to_float32_preserves_values() raises:
    """Test ToFloat32 preserves float values."""
    var values = List[Float32]()
    values.append(1.5)
    values.append(2.5)
    values.append(3.5)
    var data = ExTensor(values^)
    var converter = ToFloat32()

    var result = converter(data)

    assert_almost_equal(result[0], 1.5)
    assert_almost_equal(result[1], 2.5)
    assert_almost_equal(result[2], 3.5)


fn test_to_int32_truncates() raises:
    """Test ToInt32 truncates float values."""
    var values = List[Float32]()
    values.append(1.9)
    values.append(2.5)
    values.append(3.1)
    var data = ExTensor(values^)
    var converter = ToInt32()

    var result = converter(data)

    assert_equal(Int(result[0]), 1)
    assert_equal(Int(result[1]), 2)
    assert_equal(Int(result[2]), 3)


fn test_to_int32_negative() raises:
    """Test ToInt32 handles negative values."""
    var values = List[Float32]()
    values.append(-1.9)
    values.append(-2.5)
    values.append(-3.1)
    var data = ExTensor(values^)
    var converter = ToInt32()

    var result = converter(data)

    assert_equal(Int(result[0]), -1)
    assert_equal(Int(result[1]), -2)
    assert_equal(Int(result[2]), -3)


fn test_to_int32_zero() raises:
    """Test ToInt32 handles zero."""
    var values = List[Float32]()
    values.append(0.0)
    values.append(0.1)
    values.append(-0.1)
    var data = ExTensor(values^)
    var converter = ToInt32()

    var result = converter(data)

    assert_equal(Int(result[0]), 0)
    assert_equal(Int(result[1]), 0)
    assert_equal(Int(result[2]), 0)


# ============================================================================
# Sequential Composition Tests (first part)
# ============================================================================


fn test_sequential_basic() raises:
    """Test sequential application of transforms."""
    var values = List[Float32]()
    values.append(1.0)
    values.append(2.0)
    values.append(3.0)
    var data = ExTensor(values^)

    fn double_fn(value: Float32) -> Float32:
        return value * 2.0

    fn add_one(value: Float32) -> Float32:
        return value + 1.0

    var transforms: List[AnyTransform] = []
    transforms.append(AnyTransform(LambdaTransform(double_fn)))
    transforms.append(AnyTransform(LambdaTransform(add_one)))

    var sequential = SequentialTransform(transforms^)
    var result = sequential(data)

    # Should apply double then add_one: (1*2)+1=3, (2*2)+1=5, (3*2)+1=7
    assert_almost_equal(result[0], 3.0)
    assert_almost_equal(result[1], 5.0)
    assert_almost_equal(result[2], 7.0)


fn test_sequential_single_transform() raises:
    """Test sequential with single transform."""
    var values = List[Float32]()
    values.append(1.0)
    values.append(2.0)
    values.append(3.0)
    var data = ExTensor(values^)

    fn double_fn(value: Float32) -> Float32:
        return value * 2.0

    var transforms: List[AnyTransform] = []
    transforms.append(AnyTransform(LambdaTransform(double_fn)))

    var sequential = SequentialTransform(transforms^)
    var result = sequential(data)

    assert_almost_equal(result[0], 2.0)
    assert_almost_equal(result[1], 4.0)
    assert_almost_equal(result[2], 6.0)


fn test_sequential_empty() raises:
    """Test sequential with no transforms."""
    var values = List[Float32]()
    values.append(1.0)
    values.append(2.0)
    values.append(3.0)
    var data = ExTensor(values^)

    var transforms: List[AnyTransform] = []
    var sequential = SequentialTransform(transforms^)

    var result = sequential(data)

    # Should act like identity
    assert_almost_equal(result[0], 1.0)
    assert_almost_equal(result[1], 2.0)
    assert_almost_equal(result[2], 3.0)


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run type conversion and sequential (first part) transform tests."""
    print("Running generic transform tests - Part 4...")
    print()

    # Type conversion tests
    print("Testing type conversions...")
    test_to_float32_preserves_values()
    test_to_int32_truncates()
    test_to_int32_negative()
    test_to_int32_zero()
    print("  ✓ 4 type conversion tests passed")

    # Sequential tests (first part)
    print("Testing SequentialTransform (first part)...")
    test_sequential_basic()
    test_sequential_single_transform()
    test_sequential_empty()
    print("  ✓ 3 sequential tests passed")

    print()
    print("✓ All 7 Part 4 generic transform tests passed!")
