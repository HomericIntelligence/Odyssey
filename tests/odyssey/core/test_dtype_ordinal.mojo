"""Unit tests for dtype_ordinal module.

Tests DType to ordinal conversion and dtype name formatting.
"""

from std.testing import assert_equal, assert_true, assert_false
from odyssey.base.dtype_ordinal import (
    dtype_to_ordinal,
    format_dtype_name,
    DTYPE_FLOAT16,
    DTYPE_FLOAT32,
    DTYPE_FLOAT64,
    DTYPE_INT8,
    DTYPE_INT16,
    DTYPE_INT32,
    DTYPE_INT64,
    DTYPE_UINT8,
    DTYPE_UINT16,
    DTYPE_UINT32,
    DTYPE_UINT64,
    DTYPE_BOOL,
    DTYPE_UNSUPPORTED,
    SUPPORTED_DTYPE_COUNT,
)


def test_dtype_to_ordinal_all_supported() raises:
    """Test that all supported dtypes map to correct ordinals."""
    assert_equal(dtype_to_ordinal(DType.float16), DTYPE_FLOAT16)
    assert_equal(dtype_to_ordinal(DType.float32), DTYPE_FLOAT32)
    assert_equal(dtype_to_ordinal(DType.float64), DTYPE_FLOAT64)
    assert_equal(dtype_to_ordinal(DType.int8), DTYPE_INT8)
    assert_equal(dtype_to_ordinal(DType.int16), DTYPE_INT16)
    assert_equal(dtype_to_ordinal(DType.int32), DTYPE_INT32)
    assert_equal(dtype_to_ordinal(DType.int64), DTYPE_INT64)
    assert_equal(dtype_to_ordinal(DType.uint8), DTYPE_UINT8)
    assert_equal(dtype_to_ordinal(DType.uint16), DTYPE_UINT16)
    assert_equal(dtype_to_ordinal(DType.uint32), DTYPE_UINT32)
    assert_equal(dtype_to_ordinal(DType.uint64), DTYPE_UINT64)
    assert_equal(dtype_to_ordinal(DType.bool), DTYPE_BOOL)
    print("✓ All supported dtypes map to correct ordinals")


def test_dtype_to_ordinal_bool() raises:
    """Test that DType.bool maps to DTYPE_BOOL ordinal."""
    assert_equal(dtype_to_ordinal(DType.bool), DTYPE_BOOL)
    print("✓ DType.bool maps to DTYPE_BOOL ordinal")


def test_dtype_to_ordinal_unsupported() raises:
    """Test that unsupported dtypes return DTYPE_UNSUPPORTED."""
    # Test bfloat16 (not currently supported)
    assert_equal(dtype_to_ordinal(DType.bfloat16), DTYPE_UNSUPPORTED)
    print("✓ Unsupported dtypes return DTYPE_UNSUPPORTED")


def test_ordinal_uniqueness() raises:
    """Test that all ordinals are unique (no collisions)."""
    var ordinals = List[Int]()
    ordinals.append(DTYPE_FLOAT16)
    ordinals.append(DTYPE_FLOAT32)
    ordinals.append(DTYPE_FLOAT64)
    ordinals.append(DTYPE_INT8)
    ordinals.append(DTYPE_INT16)
    ordinals.append(DTYPE_INT32)
    ordinals.append(DTYPE_INT64)
    ordinals.append(DTYPE_UINT8)
    ordinals.append(DTYPE_UINT16)
    ordinals.append(DTYPE_UINT32)
    ordinals.append(DTYPE_UINT64)
    ordinals.append(DTYPE_BOOL)

    # Check uniqueness by comparing all pairs
    for i in range(len(ordinals)):
        for j in range(i + 1, len(ordinals)):
            if ordinals[i] == ordinals[j]:
                raise Error("Ordinal collision detected")

    print("✓ All ordinals are unique")


def test_ordinal_count() raises:
    """Test that SUPPORTED_DTYPE_COUNT matches actual count."""
    assert_equal(SUPPORTED_DTYPE_COUNT, 12)
    print("✓ SUPPORTED_DTYPE_COUNT is correct")


def test_format_dtype_name_all_supported() raises:
    """Test that all supported dtypes have correct names."""
    assert_equal(format_dtype_name(DType.float16), "float16")
    assert_equal(format_dtype_name(DType.float32), "float32")
    assert_equal(format_dtype_name(DType.float64), "float64")
    assert_equal(format_dtype_name(DType.int8), "int8")
    assert_equal(format_dtype_name(DType.int16), "int16")
    assert_equal(format_dtype_name(DType.int32), "int32")
    assert_equal(format_dtype_name(DType.int64), "int64")
    assert_equal(format_dtype_name(DType.uint8), "uint8")
    assert_equal(format_dtype_name(DType.uint16), "uint16")
    assert_equal(format_dtype_name(DType.uint32), "uint32")
    assert_equal(format_dtype_name(DType.uint64), "uint64")
    assert_equal(format_dtype_name(DType.bool), "bool")
    print("✓ All supported dtypes format correctly")


def test_format_dtype_name_unsupported() raises:
    """Test that unsupported dtypes return 'unknown'."""
    assert_equal(format_dtype_name(DType.bfloat16), "unknown")
    print("✓ Unsupported dtypes format as 'unknown'")


def test_ordinal_values_sequential() raises:
    """Test that ordinal values are sequential from 0."""
    assert_equal(DTYPE_FLOAT16, 0)
    assert_equal(DTYPE_FLOAT32, 1)
    assert_equal(DTYPE_FLOAT64, 2)
    assert_equal(DTYPE_INT8, 3)
    assert_equal(DTYPE_INT16, 4)
    assert_equal(DTYPE_INT32, 5)
    assert_equal(DTYPE_INT64, 6)
    assert_equal(DTYPE_UINT8, 7)
    assert_equal(DTYPE_UINT16, 8)
    assert_equal(DTYPE_UINT32, 9)
    assert_equal(DTYPE_UINT64, 10)
    assert_equal(DTYPE_BOOL, 11)
    assert_equal(DTYPE_UNSUPPORTED, -1)
    print("✓ Ordinal values are sequential")


def main() raises:
    """Run all dtype_ordinal tests."""
    print("Running dtype_ordinal tests...")
    print()

    test_dtype_to_ordinal_all_supported()
    test_dtype_to_ordinal_bool()
    test_dtype_to_ordinal_unsupported()
    test_ordinal_uniqueness()
    test_ordinal_count()
    test_format_dtype_name_all_supported()
    test_format_dtype_name_unsupported()
    test_ordinal_values_sequential()

    print()
    print("All dtype_ordinal tests passed!")
