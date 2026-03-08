# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_generic_transforms.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for generic data transformation utilities - Part 2.

Tests conditional transforms and the first half of clamp transforms.
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
# Conditional Transform Tests
# ============================================================================


fn test_conditional_always_apply() raises:
    """Test conditional transform with always-true predicate."""
    var values = List[Float32]()
    values.append(1.0)
    values.append(2.0)
    values.append(3.0)
    var data = ExTensor(values^)

    fn always_true(tensor: ExTensor) raises -> Bool:
        return True

    fn double_fn(value: Float32) -> Float32:
        return value * 2.0

    var base_transform = LambdaTransform(double_fn)
    var conditional = ConditionalTransform[LambdaTransform](
        always_true, base_transform^
    )

    var result = conditional(data)

    # Should apply transform
    assert_almost_equal(result[0], 2.0)
    assert_almost_equal(result[1], 4.0)
    assert_almost_equal(result[2], 6.0)


fn test_conditional_never_apply() raises:
    """Test conditional transform with always-false predicate."""
    var values = List[Float32]()
    values.append(1.0)
    values.append(2.0)
    values.append(3.0)
    var data = ExTensor(values^)

    fn always_false(tensor: ExTensor) raises -> Bool:
        return False

    fn double_fn(value: Float32) -> Float32:
        return value * 2.0

    var base_transform = LambdaTransform(double_fn)
    var conditional = ConditionalTransform[LambdaTransform](
        always_false, base_transform^
    )

    var result = conditional(data)

    # Should NOT apply transform (return original)
    assert_almost_equal(result[0], 1.0)
    assert_almost_equal(result[1], 2.0)
    assert_almost_equal(result[2], 3.0)


fn test_conditional_based_on_size() raises:
    """Test conditional transform based on tensor size."""
    var small_values = List[Float32]()
    small_values.append(1.0)
    small_values.append(2.0)
    var small_data = ExTensor(small_values^)
    var large_values = List[Float32]()
    large_values.append(1.0)
    large_values.append(2.0)
    large_values.append(3.0)
    large_values.append(4.0)
    var large_data = ExTensor(large_values^)

    fn is_large(tensor: ExTensor) raises -> Bool:
        return tensor.num_elements() > 3

    fn double_fn(value: Float32) -> Float32:
        return value * 2.0

    var base_transform = LambdaTransform(double_fn)
    var conditional = ConditionalTransform[LambdaTransform](
        is_large, base_transform^
    )

    var result_small = conditional(small_data)
    var result_large = conditional(large_data)

    # Small should not be transformed
    assert_almost_equal(result_small[0], 1.0)
    assert_almost_equal(result_small[1], 2.0)

    # Large should be transformed
    assert_almost_equal(result_large[0], 2.0)
    assert_almost_equal(result_large[1], 4.0)


fn test_conditional_based_on_values() raises:
    """Test conditional transform based on tensor values."""
    var positive_values = List[Float32]()
    positive_values.append(1.0)
    positive_values.append(2.0)
    positive_values.append(3.0)
    var positive_data = ExTensor(positive_values^)
    var mixed_values = List[Float32]()
    mixed_values.append(-1.0)
    mixed_values.append(2.0)
    mixed_values.append(3.0)
    var mixed_data = ExTensor(mixed_values^)

    fn all_positive(tensor: ExTensor) raises -> Bool:
        for i in range(tensor.num_elements()):
            if tensor[i] < 0.0:
                return False
        return True

    fn double_fn(value: Float32) -> Float32:
        return value * 2.0

    var base_transform = LambdaTransform(double_fn)
    var conditional = ConditionalTransform[LambdaTransform](
        all_positive, base_transform^
    )

    var result_positive = conditional(positive_data)
    var result_mixed = conditional(mixed_data)

    # Positive should be transformed
    assert_almost_equal(result_positive[0], 2.0)

    # Mixed should not be transformed
    assert_almost_equal(result_mixed[0], -1.0)


# ============================================================================
# Clamp Transform Tests (first half)
# ============================================================================


fn test_clamp_basic() raises:
    """Test clamp limits values to range."""
    var values = List[Float32]()
    values.append(0.0)
    values.append(0.5)
    values.append(1.0)
    values.append(1.5)
    values.append(2.0)
    var data = ExTensor(values^)
    var clamp = ClampTransform(0.3, 1.2)

    var result = clamp(data)

    assert_almost_equal(result[0], 0.3)  # Clamped to min
    assert_almost_equal(result[1], 0.5)  # Unchanged
    assert_almost_equal(result[2], 1.0)  # Unchanged
    assert_almost_equal(result[3], 1.2)  # Clamped to max
    assert_almost_equal(result[4], 1.2)  # Clamped to max


fn test_clamp_all_below_min() raises:
    """Test clamp when all values below minimum."""
    var values = List[Float32]()
    values.append(-5.0)
    values.append(-2.0)
    values.append(0.0)
    var data = ExTensor(values^)
    var clamp = ClampTransform(1.0, 10.0)

    var result = clamp(data)

    assert_almost_equal(result[0], 1.0)
    assert_almost_equal(result[1], 1.0)
    assert_almost_equal(result[2], 1.0)


fn test_clamp_all_above_max() raises:
    """Test clamp when all values above maximum."""
    var values = List[Float32]()
    values.append(15.0)
    values.append(20.0)
    values.append(25.0)
    var data = ExTensor(values^)
    var clamp = ClampTransform(1.0, 10.0)

    var result = clamp(data)

    assert_almost_equal(result[0], 10.0)
    assert_almost_equal(result[1], 10.0)
    assert_almost_equal(result[2], 10.0)


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run conditional and clamp (first half) transform tests."""
    print("Running generic transform tests - Part 2...")
    print()

    # Conditional tests
    print("Testing ConditionalTransform...")
    test_conditional_always_apply()
    test_conditional_never_apply()
    test_conditional_based_on_size()
    test_conditional_based_on_values()
    print("  ✓ 4 conditional tests passed")

    # Clamp tests (first half)
    print("Testing ClampTransform (first half)...")
    test_clamp_basic()
    test_clamp_all_below_min()
    test_clamp_all_above_max()
    print("  ✓ 3 clamp tests passed")

    print()
    print("✓ All 7 Part 2 generic transform tests passed!")
