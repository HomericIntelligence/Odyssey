# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_dtype_utils.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for dtype_utils module - Part 1: get_test_dtypes tests

Tests the DType iteration utilities:
- get_test_dtypes() returns all dtypes (count, float32, float16, bfloat16, int8)
- get_float_dtypes() count check
"""

from shared.testing.dtype_utils import (
    get_test_dtypes,
    get_float_dtypes,
)
from shared.testing.assertions import (
    assert_equal_int,
    assert_true,
)


fn test_get_test_dtypes_not_empty() raises:
    """Test that get_test_dtypes returns a non-empty list."""
    var dtypes = get_test_dtypes()
    assert_equal_int(dtypes.__len__(), 4, "Should have 4 dtypes")


fn test_get_test_dtypes_contains_float32() raises:
    """Test that get_test_dtypes includes float32."""
    var dtypes = get_test_dtypes()
    var found = False
    for dtype in dtypes:
        if dtype == DType.float32:
            found = True
            break
    assert_true(found, "Should contain float32")


fn test_get_test_dtypes_contains_float16() raises:
    """Test that get_test_dtypes includes float16."""
    var dtypes = get_test_dtypes()
    var found = False
    for dtype in dtypes:
        if dtype == DType.float16:
            found = True
            break
    assert_true(found, "Should contain float16")


fn test_get_test_dtypes_contains_bfloat16() raises:
    """Test that get_test_dtypes includes bfloat16."""
    var dtypes = get_test_dtypes()
    var found = False
    for dtype in dtypes:
        if dtype == DType.bfloat16:
            found = True
            break
    assert_true(found, "Should contain bfloat16")


fn test_get_test_dtypes_contains_int8() raises:
    """Test that get_test_dtypes includes int8."""
    var dtypes = get_test_dtypes()
    var found = False
    for dtype in dtypes:
        if dtype == DType.int8:
            found = True
            break
    assert_true(found, "Should contain int8")


fn test_get_float_dtypes_count() raises:
    """Test that get_float_dtypes returns 3 items (no int8)."""
    var dtypes = get_float_dtypes()
    assert_equal_int(dtypes.__len__(), 3, "Should have 3 float dtypes")


fn main() raises:
    print("Testing dtype_utils module (part 1)...")

    test_get_test_dtypes_not_empty()
    print("✓ test_get_test_dtypes_not_empty")

    test_get_test_dtypes_contains_float32()
    print("✓ test_get_test_dtypes_contains_float32")

    test_get_test_dtypes_contains_float16()
    print("✓ test_get_test_dtypes_contains_float16")

    test_get_test_dtypes_contains_bfloat16()
    print("✓ test_get_test_dtypes_contains_bfloat16")

    test_get_test_dtypes_contains_int8()
    print("✓ test_get_test_dtypes_contains_int8")

    test_get_float_dtypes_count()
    print("✓ test_get_float_dtypes_count")

    print("\n✅ All dtype_utils part 1 tests passed!")
