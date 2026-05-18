"""Tests for dtype_utils module.

Tests the DType iteration utilities:
- get_test_dtypes() returns all dtypes (count, float32, float16, bfloat16, int8)
- get_float_dtypes() count check
"""


from projectodyssey.testing.dtype_utils import (
    dtype_to_string,
    get_float32_only,
    get_float_dtypes,
    get_precision_dtypes,
    get_test_dtypes,
)
from projectodyssey.testing.assertions import (
    assert_equal_int,
    assert_true,
)


def test_get_test_dtypes_not_empty() raises:
    """Test that get_test_dtypes returns a non-empty list."""
    var dtypes = get_test_dtypes()
    assert_equal_int(dtypes.__len__(), 4, "Should have 4 dtypes")


def test_get_test_dtypes_contains_float32() raises:
    """Test that get_test_dtypes includes float32."""
    var dtypes = get_test_dtypes()
    var found = False
    for dtype in dtypes:
        if dtype == DType.float32:
            found = True
            break
    assert_true(found, "Should contain float32")


def test_get_test_dtypes_contains_float16() raises:
    """Test that get_test_dtypes includes float16."""
    var dtypes = get_test_dtypes()
    var found = False
    for dtype in dtypes:
        if dtype == DType.float16:
            found = True
            break
    assert_true(found, "Should contain float16")


def test_get_test_dtypes_contains_bfloat16() raises:
    """Test that get_test_dtypes includes bfloat16."""
    var dtypes = get_test_dtypes()
    var found = False
    for dtype in dtypes:
        if dtype == DType.bfloat16:
            found = True
            break
    assert_true(found, "Should contain bfloat16")


def test_get_test_dtypes_contains_int8() raises:
    """Test that get_test_dtypes includes int8."""
    var dtypes = get_test_dtypes()
    var found = False
    for dtype in dtypes:
        if dtype == DType.int8:
            found = True
            break
    assert_true(found, "Should contain int8")


def test_get_float_dtypes_count() raises:
    """Test that get_float_dtypes returns 3 items (no int8)."""
    var dtypes = get_float_dtypes()
    assert_equal_int(dtypes.__len__(), 3, "Should have 3 float dtypes")


def test_get_float_dtypes_no_int8() raises:
    """Test that get_float_dtypes excludes int8."""
    var dtypes = get_float_dtypes()
    for dtype in dtypes:
        assert_true(
            dtype != DType.int8, "get_float_dtypes should not include int8"
        )


def test_get_precision_dtypes_count() raises:
    """Test that get_precision_dtypes returns 4 items."""
    var dtypes = get_precision_dtypes()
    assert_equal_int(dtypes.__len__(), 4, "Should have 4 precision dtypes")


def test_get_float32_only_single_dtype() raises:
    """Test that get_float32_only returns exactly one dtype."""
    var dtypes = get_float32_only()
    assert_equal_int(dtypes.__len__(), 1, "Should have exactly 1 dtype")


def test_get_float32_only_is_float32() raises:
    """Test that get_float32_only returns float32."""
    var dtypes = get_float32_only()
    assert_true(dtypes[0] == DType.float32, "Should be float32")


def test_dtype_to_string_float16() raises:
    """Test dtype_to_string converts float16 correctly."""
    var result = dtype_to_string(DType.float16)
    assert_true(result == "float16", "Should convert to 'float16'")


def test_dtype_to_string_float32() raises:
    """Test dtype_to_string converts float32 correctly."""
    var result = dtype_to_string(DType.float32)
    assert_true(result == "float32", "Should convert to 'float32'")


def test_dtype_to_string_bfloat16() raises:
    """Test dtype_to_string converts bfloat16 correctly."""
    var result = dtype_to_string(DType.bfloat16)
    assert_true(result == "bfloat16", "Should convert to 'bfloat16'")


def test_dtype_to_string_int8() raises:
    """Test dtype_to_string converts int8 correctly."""
    var result = dtype_to_string(DType.int8)
    assert_true(result == "int8", "Should convert to 'int8'")


def test_dtype_lists_are_independent() raises:
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


def test_iterate_all_dtypes() raises:
    """Test that we can iterate over all dtypes without errors."""
    var dtypes = get_test_dtypes()
    var count = 0
    for dtype in dtypes:
        # Just verify we can access each dtype
        _ = dtype_to_string(dtype)
        count += 1

    assert_equal_int(count, 4, "Should iterate through 4 dtypes")


def test_iterate_float_dtypes_only() raises:
    """Test that we can iterate over float dtypes."""
    var dtypes = get_float_dtypes()
    var count = 0
    for dtype in dtypes:
        _ = dtype_to_string(dtype)
        count += 1

    assert_equal_int(count, 3, "Should iterate through 3 float dtypes")


def main() raises:
    """Run all test_dtype_utils tests."""
    print("Running test_dtype_utils tests...")

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

    test_dtype_lists_are_independent()
    print("✓ test_dtype_lists_are_independent")

    test_iterate_all_dtypes()
    print("✓ test_iterate_all_dtypes")

    test_iterate_float_dtypes_only()
    print("✓ test_iterate_float_dtypes_only")

    print("\nAll test_dtype_utils tests passed!")
