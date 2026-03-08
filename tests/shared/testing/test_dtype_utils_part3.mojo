# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_dtype_utils.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for dtype_utils module - Part 3: integration / iteration tests

Tests the DType iteration utilities:
- dtype lists are independent
- iterate all dtypes without errors
- iterate float dtypes only
"""

from shared.testing.dtype_utils import (
    get_test_dtypes,
    get_float_dtypes,
    dtype_to_string,
)
from shared.testing.assertions import (
    assert_equal_int,
    assert_true,
)


fn test_dtype_lists_are_independent() raises:
    """Test that returned lists are independent."""
    var dtypes1 = get_test_dtypes()
    var dtypes2 = get_test_dtypes()

    # Both should have same content
    assert_equal_int(
        dtypes1.__len__(), dtypes2.__len__(), "Lists should have same length"
    )

    # Both should contain float32
    var found1 = False
    var found2 = False
    for dtype in dtypes1:
        if dtype == DType.float32:
            found1 = True
    for dtype in dtypes2:
        if dtype == DType.float32:
            found2 = True

    assert_true(found1 and found2, "Both lists should contain float32")


fn test_iterate_all_dtypes() raises:
    """Test that we can iterate over all dtypes without errors."""
    var dtypes = get_test_dtypes()
    var count = 0
    for dtype in dtypes:
        # Just verify we can access each dtype
        var name = dtype_to_string(dtype)
        count += 1

    assert_equal_int(count, 4, "Should iterate through 4 dtypes")


fn test_iterate_float_dtypes_only() raises:
    """Test that we can iterate over float dtypes."""
    var dtypes = get_float_dtypes()
    var count = 0
    for dtype in dtypes:
        var name = dtype_to_string(dtype)
        count += 1

    assert_equal_int(count, 3, "Should iterate through 3 float dtypes")


fn main() raises:
    print("Testing dtype_utils module (part 3)...")

    test_dtype_lists_are_independent()
    print("✓ test_dtype_lists_are_independent")

    test_iterate_all_dtypes()
    print("✓ test_iterate_all_dtypes")

    test_iterate_float_dtypes_only()
    print("✓ test_iterate_float_dtypes_only")

    print("\n✅ All dtype_utils part 3 tests passed!")
