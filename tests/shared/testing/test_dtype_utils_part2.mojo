# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_dtype_utils.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for dtype_utils module - Part 2: get_float_dtypes, get_precision_dtypes, get_float32_only, dtype_to_string

Tests the DType iteration utilities:
- get_float_dtypes() no int8 check
- get_precision_dtypes() count
- get_float32_only() single dtype and value
- dtype_to_string() conversions for float16, float32, bfloat16, int8
"""

from shared.testing.dtype_utils import (
    get_float_dtypes,
    get_precision_dtypes,
    get_float32_only,
    dtype_to_string,
)
from shared.testing.assertions import (
    assert_equal_int,
    assert_true,
)


fn test_get_float_dtypes_no_int8() raises:
    """Test that get_float_dtypes excludes int8."""
    var dtypes = get_float_dtypes()
    for dtype in dtypes:
        assert_true(
            dtype != DType.int8, "get_float_dtypes should not include int8"
        )


fn test_get_precision_dtypes_count() raises:
    """Test that get_precision_dtypes returns 4 items."""
    var dtypes = get_precision_dtypes()
    assert_equal_int(dtypes.__len__(), 4, "Should have 4 precision dtypes")


fn test_get_float32_only_single_dtype() raises:
    """Test that get_float32_only returns exactly one dtype."""
    var dtypes = get_float32_only()
    assert_equal_int(dtypes.__len__(), 1, "Should have exactly 1 dtype")


fn test_get_float32_only_is_float32() raises:
    """Test that get_float32_only returns float32."""
    var dtypes = get_float32_only()
    assert_true(dtypes[0] == DType.float32, "Should be float32")


fn test_dtype_to_string_float16() raises:
    """Test dtype_to_string converts float16 correctly."""
    var result = dtype_to_string(DType.float16)
    assert_true(result == "float16", "Should convert to 'float16'")


fn test_dtype_to_string_float32() raises:
    """Test dtype_to_string converts float32 correctly."""
    var result = dtype_to_string(DType.float32)
    assert_true(result == "float32", "Should convert to 'float32'")


fn test_dtype_to_string_bfloat16() raises:
    """Test dtype_to_string converts bfloat16 correctly."""
    var result = dtype_to_string(DType.bfloat16)
    assert_true(result == "bfloat16", "Should convert to 'bfloat16'")


fn test_dtype_to_string_int8() raises:
    """Test dtype_to_string converts int8 correctly."""
    var result = dtype_to_string(DType.int8)
    assert_true(result == "int8", "Should convert to 'int8'")


fn main() raises:
    print("Testing dtype_utils module (part 2)...")

    test_get_float_dtypes_no_int8()
    print("✓ test_get_float_dtypes_no_int8")

    test_get_precision_dtypes_count()
    print("✓ test_get_precision_dtypes_count")

    test_get_float32_only_single_dtype()
    print("✓ test_get_float32_only_single_dtype")

    test_get_float32_only_is_float32()
    print("✓ test_get_float32_only_is_float32")

    test_dtype_to_string_float16()
    print("✓ test_dtype_to_string_float16")

    test_dtype_to_string_float32()
    print("✓ test_dtype_to_string_float32")

    test_dtype_to_string_bfloat16()
    print("✓ test_dtype_to_string_bfloat16")

    test_dtype_to_string_int8()
    print("✓ test_dtype_to_string_int8")

    print("\n✅ All dtype_utils part 2 tests passed!")
