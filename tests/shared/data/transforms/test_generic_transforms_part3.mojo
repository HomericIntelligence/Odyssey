# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_generic_transforms.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for generic data transformation utilities - Part 3.

Tests second half of clamp transforms and debug transforms.
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
# Clamp Transform Tests (second half)
# ============================================================================


fn test_clamp_all_in_range() raises:
    """Test clamp when all values already in range."""
    var values = List[Float32]()
    values.append(2.0)
    values.append(5.0)
    values.append(8.0)
    var data = ExTensor(values^)
    var clamp = ClampTransform(1.0, 10.0)

    var result = clamp(data)

    assert_almost_equal(result[0], 2.0)
    assert_almost_equal(result[1], 5.0)
    assert_almost_equal(result[2], 8.0)


fn test_clamp_negative_range() raises:
    """Test clamp with negative range."""
    var values = List[Float32]()
    values.append(-10.0)
    values.append(-5.0)
    values.append(0.0)
    values.append(5.0)
    var data = ExTensor(values^)
    var clamp = ClampTransform(-8.0, -2.0)

    var result = clamp(data)

    assert_almost_equal(result[0], -8.0)  # Clamped to min
    assert_almost_equal(result[1], -5.0)  # Unchanged
    assert_almost_equal(result[2], -2.0)  # Clamped to max
    assert_almost_equal(result[3], -2.0)  # Clamped to max


fn test_clamp_zero_crossing() raises:
    """Test clamp range crossing zero."""
    var values = List[Float32]()
    values.append(-5.0)
    values.append(-1.0)
    values.append(0.0)
    values.append(1.0)
    values.append(5.0)
    var data = ExTensor(values^)
    var clamp = ClampTransform(-2.0, 2.0)

    var result = clamp(data)

    assert_almost_equal(result[0], -2.0)
    assert_almost_equal(result[1], -1.0)
    assert_almost_equal(result[2], 0.0)
    assert_almost_equal(result[3], 1.0)
    assert_almost_equal(result[4], 2.0)


# ============================================================================
# Debug Transform Tests
# ============================================================================


fn test_debug_passthrough() raises:
    """Test debug transform passes data through unchanged."""
    var values = List[Float32]()
    values.append(1.0)
    values.append(2.0)
    values.append(3.0)
    var data = ExTensor(values^)
    var debug = DebugTransform("test")

    var result = debug(data)

    # Should pass through unchanged
    assert_equal(result.num_elements(), 3)
    assert_almost_equal(result[0], 1.0)
    assert_almost_equal(result[1], 2.0)
    assert_almost_equal(result[2], 3.0)


fn test_debug_with_empty_tensor() raises:
    """Test debug transform with empty tensor."""
    var values = List[Float32]()
    var data = ExTensor(values^)
    var debug = DebugTransform("empty_test")

    var result = debug(data)

    assert_equal(result.num_elements(), 0)


fn test_debug_with_large_tensor() raises:
    """Test debug transform with large tensor."""
    var values = List[Float32]()
    for i in range(100):
        values.append(Float32(i))

    var data = ExTensor(values^)
    var debug = DebugTransform("large_test")

    var result = debug(data)

    # Should pass through unchanged
    assert_equal(result.num_elements(), 100)
    for i in range(100):
        var expected = Float32(i)
        assert_almost_equal(Float32(result[i]), expected)


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run clamp (second half) and debug transform tests."""
    print("Running generic transform tests - Part 3...")
    print()

    # Clamp tests (second half)
    print("Testing ClampTransform (second half)...")
    test_clamp_all_in_range()
    test_clamp_negative_range()
    test_clamp_zero_crossing()
    print("  ✓ 3 clamp tests passed")

    # Debug tests
    print("Testing DebugTransform...")
    test_debug_passthrough()
    test_debug_with_empty_tensor()
    test_debug_with_large_tensor()
    print("  ✓ 3 debug tests passed")

    print()
    print("✓ All 6 Part 3 generic transform tests passed!")
