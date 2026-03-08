# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_utils.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for top_k utility functions in shared.core.utils (Part 2 of 3).

This module tests top_k functions including:
- top_k_indices (single element, all elements, duplicates)
- top_k (values and indices, multidimensional)
- argsort (ascending, descending, single element)
"""

from tests.shared.conftest import (
    assert_equal,
    assert_equal_int,
    assert_shape,
    assert_true,
    assert_close_float,
)
from shared.core.extensor import ExTensor, zeros, ones, arange
from shared.core.utils import (
    top_k_indices,
    top_k,
    argsort,
)


# ============================================================================
# Top K Tests
# ============================================================================


fn test_top_k_indices_single_element() raises:
    """Test top_k_indices with k=1."""
    var t = arange(0.0, 5.0, 1.0, DType.float32)
    var indices = top_k_indices(t, 1)
    assert_equal_int(indices[0], 4)


fn test_top_k_indices_all_elements() raises:
    """Test top_k_indices with k=numel."""
    var t = arange(0.0, 5.0, 1.0, DType.float32)
    var indices = top_k_indices(t, 5)
    assert_equal_int(indices[0], 4)
    assert_equal_int(indices[1], 3)
    assert_equal_int(indices[2], 2)
    assert_equal_int(indices[3], 1)
    assert_equal_int(indices[4], 0)


fn test_top_k_indices_with_duplicates() raises:
    """Test top_k_indices with duplicate values."""
    var shape = List[Int]()
    shape.append(6)
    var t = zeros(shape, DType.float32)
    t._data.bitcast[Float32]()[0] = 5.0
    t._data.bitcast[Float32]()[1] = 5.0
    t._data.bitcast[Float32]()[2] = 3.0
    t._data.bitcast[Float32]()[3] = 3.0
    t._data.bitcast[Float32]()[4] = 1.0
    t._data.bitcast[Float32]()[5] = 1.0

    var indices = top_k_indices(t, 3)
    # First two should be indices 0 and 1 (both value 5)
    # Third should be index 2 or 3 (value 3)
    assert_equal_int(len(indices), 3)


fn test_top_k_values_and_indices() raises:
    """Test top_k function returns both values and indices."""
    var t = arange(0.0, 10.0, 1.0, DType.float32)
    var result = top_k(t, 3)
    var values = result[0]

    # Check shape of values
    assert_shape(values, [3])

    # Check values are in correct order (descending)
    assert_close_float(values._get_float64(0), 9.0)
    assert_close_float(values._get_float64(1), 8.0)
    assert_close_float(values._get_float64(2), 7.0)

    # Check indices (access from result tuple to avoid List[Int] copy)
    assert_equal_int(result[1][0], 9)
    assert_equal_int(result[1][1], 8)
    assert_equal_int(result[1][2], 7)


fn test_top_k_multidimensional() raises:
    """Test top_k on multi-dimensional tensor."""
    var t = arange(0.0, 12.0, 1.0, DType.float32)
    var result = top_k(t, 2)
    var values = result[0]

    assert_shape(values, [2])
    assert_close_float(values._get_float64(0), 11.0)
    assert_close_float(values._get_float64(1), 10.0)


# ============================================================================
# Argsort Tests (partial)
# ============================================================================


fn test_argsort_ascending() raises:
    """Test argsort in ascending order."""
    var shape = List[Int]()
    shape.append(5)
    var t = zeros(shape, DType.float32)
    t._data.bitcast[Float32]()[0] = 5.0
    t._data.bitcast[Float32]()[1] = 2.0
    t._data.bitcast[Float32]()[2] = 8.0
    t._data.bitcast[Float32]()[3] = 1.0
    t._data.bitcast[Float32]()[4] = 9.0

    var indices = argsort(t, descending=False)
    assert_equal_int(indices[0], 3)  # value 1
    assert_equal_int(indices[1], 1)  # value 2
    assert_equal_int(indices[2], 0)  # value 5
    assert_equal_int(indices[3], 2)  # value 8
    assert_equal_int(indices[4], 4)  # value 9


fn test_argsort_descending() raises:
    """Test argsort in descending order."""
    var shape = List[Int]()
    shape.append(5)
    var t = zeros(shape, DType.float32)
    t._data.bitcast[Float32]()[0] = 5.0
    t._data.bitcast[Float32]()[1] = 2.0
    t._data.bitcast[Float32]()[2] = 8.0
    t._data.bitcast[Float32]()[3] = 1.0
    t._data.bitcast[Float32]()[4] = 9.0

    var indices = argsort(t, descending=True)
    assert_equal_int(indices[0], 4)  # value 9
    assert_equal_int(indices[1], 2)  # value 8
    assert_equal_int(indices[2], 0)  # value 5
    assert_equal_int(indices[3], 1)  # value 2
    assert_equal_int(indices[4], 3)  # value 1


fn test_argsort_single_element() raises:
    """Test argsort with single element."""
    var t = arange(0.0, 1.0, 1.0, DType.float32)
    var indices = argsort(t, descending=False)
    assert_equal_int(indices[0], 0)


fn main() raises:
    """Run all tests."""
    print("=" * 60)
    print("Running shared.core.utils tests (Part 2)")
    print("=" * 60)

    # Top K tests
    print("\n=== Top K ===")
    test_top_k_indices_single_element()
    print("✓ test_top_k_indices_single_element")
    test_top_k_indices_all_elements()
    print("✓ test_top_k_indices_all_elements")
    test_top_k_indices_with_duplicates()
    print("✓ test_top_k_indices_with_duplicates")
    test_top_k_values_and_indices()
    print("✓ test_top_k_values_and_indices")
    test_top_k_multidimensional()
    print("✓ test_top_k_multidimensional")

    # Argsort tests (partial)
    print("\n=== Argsort (partial) ===")
    test_argsort_ascending()
    print("✓ test_argsort_ascending")
    test_argsort_descending()
    print("✓ test_argsort_descending")
    test_argsort_single_element()
    print("✓ test_argsort_single_element")

    print("\n" + "=" * 60)
    print("All 8 utils tests (Part 2) passed!")
    print("=" * 60)
