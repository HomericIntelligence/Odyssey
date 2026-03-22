# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_utils.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for argsort utility functions in shared.core.utils (Part 3 of 3).

This module tests argsort functions including:
- argsort (sorted array, reverse sorted, negative values, multidimensional)
"""

from tests.shared.conftest import (
    assert_equal,
    assert_equal_int,
    assert_shape,
    assert_true,
    assert_close_float,
)
from shared.core.extensor import AnyTensor, zeros, ones, arange
from shared.core.utils import (
    argsort,
)


# ============================================================================
# Argsort Tests
# ============================================================================


fn test_argsort_sorted_array() raises:
    """Test argsort on already sorted array."""
    var t = arange(0.0, 5.0, 1.0, DType.float32)
    var indices = argsort(t, descending=False)
    assert_equal_int(indices[0], 0)
    assert_equal_int(indices[1], 1)
    assert_equal_int(indices[2], 2)
    assert_equal_int(indices[3], 3)
    assert_equal_int(indices[4], 4)


fn test_argsort_reverse_sorted() raises:
    """Test argsort on reverse sorted array."""
    var shape = List[Int]()
    shape.append(5)
    var t = zeros(shape, DType.float32)
    t._data.bitcast[Float32]()[0] = 5.0
    t._data.bitcast[Float32]()[1] = 4.0
    t._data.bitcast[Float32]()[2] = 3.0
    t._data.bitcast[Float32]()[3] = 2.0
    t._data.bitcast[Float32]()[4] = 1.0

    var indices = argsort(t, descending=False)
    assert_equal_int(indices[0], 4)
    assert_equal_int(indices[1], 3)
    assert_equal_int(indices[2], 2)
    assert_equal_int(indices[3], 1)
    assert_equal_int(indices[4], 0)


fn test_argsort_negative_values() raises:
    """Test argsort with negative values."""
    var shape = List[Int]()
    shape.append(5)
    var t = zeros(shape, DType.float32)
    t._data.bitcast[Float32]()[0] = -5.0
    t._data.bitcast[Float32]()[1] = 2.0
    t._data.bitcast[Float32]()[2] = -1.0
    t._data.bitcast[Float32]()[3] = 0.0
    t._data.bitcast[Float32]()[4] = 3.0

    var indices = argsort(t, descending=False)
    assert_equal_int(indices[0], 0)  # -5
    assert_equal_int(indices[1], 2)  # -1
    assert_equal_int(indices[2], 3)  # 0
    assert_equal_int(indices[3], 1)  # 2
    assert_equal_int(indices[4], 4)  # 3


fn test_argsort_multidimensional() raises:
    """Test argsort on multi-dimensional tensor (flattens)."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var t = zeros(shape, DType.float32)
    t._data.bitcast[Float32]()[0] = 5.0
    t._data.bitcast[Float32]()[1] = 1.0
    t._data.bitcast[Float32]()[2] = 3.0
    t._data.bitcast[Float32]()[3] = 2.0
    t._data.bitcast[Float32]()[4] = 4.0
    t._data.bitcast[Float32]()[5] = 0.0

    var indices = argsort(t, descending=False)
    assert_equal_int(indices[0], 5)  # 0
    assert_equal_int(indices[1], 1)  # 1
    assert_equal_int(indices[2], 3)  # 2
    assert_equal_int(indices[3], 2)  # 3
    assert_equal_int(indices[4], 4)  # 4
    assert_equal_int(indices[5], 0)  # 5


fn main() raises:
    """Run all tests."""
    print("=" * 60)
    print("Running shared.core.utils tests (Part 3)")
    print("=" * 60)

    # Argsort tests
    print("\n=== Argsort ===")
    test_argsort_sorted_array()
    print("✓ test_argsort_sorted_array")
    test_argsort_reverse_sorted()
    print("✓ test_argsort_reverse_sorted")
    test_argsort_negative_values()
    print("✓ test_argsort_negative_values")
    test_argsort_multidimensional()
    print("✓ test_argsort_multidimensional")

    print("\n" + "=" * 60)
    print("All 4 utils tests (Part 3) passed!")
    print("=" * 60)
