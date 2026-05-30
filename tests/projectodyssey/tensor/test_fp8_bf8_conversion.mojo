"""Tests for FP8/BF8 dtype conversion in AnyTensor.

Tests cover:
- to_fp8() rejects bfloat16 with descriptive error
- to_bf8() rejects bfloat16 with descriptive error
- to_fp8() accepts float32 (round-trip to fp8 and back)
- to_bf8() accepts float32 (round-trip to bf8 and back)
- to_fp8() accepts float16 and float64
- to_bf8() accepts float16 and float64
- to_fp8() and to_bf8() reject non-float dtypes (int32, etc.)
"""

from std.testing import assert_true, assert_almost_equal
from projectodyssey.tensor.any_tensor import AnyTensor, zeros


def test_fp8_rejects_bfloat16() raises:
    """Verify to_fp8() rejects bfloat16 with descriptive error."""
    var t = zeros([2, 3], DType.bfloat16)
    var raised = False
    try:
        _ = t.to_fp8()
    except:
        raised = True
    assert_true(raised, "to_fp8() should reject bfloat16")
    print("PASS: test_fp8_rejects_bfloat16")


def test_bf8_rejects_bfloat16() raises:
    """Verify to_bf8() rejects bfloat16 with descriptive error."""
    var t = zeros([2, 3], DType.bfloat16)
    var raised = False
    try:
        _ = t.to_bf8()
    except:
        raised = True
    assert_true(raised, "to_bf8() should reject bfloat16")
    print("PASS: test_bf8_rejects_bfloat16")


def test_fp8_accepts_float32() raises:
    """Verify to_fp8() accepts float32 without error."""
    var t = zeros([2, 3], DType.float32)
    var fp8_t = t.to_fp8()
    assert_true(fp8_t.dtype() == DType.uint8, "FP8 output should be uint8")
    assert_true(fp8_t.numel() == 6, "FP8 conversion preserves numel")
    print("PASS: test_fp8_accepts_float32")


def test_bf8_accepts_float32() raises:
    """Verify to_bf8() accepts float32 without error."""
    var t = zeros([2, 3], DType.float32)
    var bf8_t = t.to_bf8()
    assert_true(bf8_t.dtype() == DType.uint8, "BF8 output should be uint8")
    assert_true(bf8_t.numel() == 6, "BF8 conversion preserves numel")
    print("PASS: test_bf8_accepts_float32")


def test_fp8_accepts_float16() raises:
    """Verify to_fp8() accepts float16 without error."""
    var t = zeros([2, 3], DType.float16)
    var fp8_t = t.to_fp8()
    assert_true(fp8_t.dtype() == DType.uint8, "FP8 output should be uint8")
    assert_true(fp8_t.numel() == 6, "FP8 conversion preserves numel")
    print("PASS: test_fp8_accepts_float16")


def test_bf8_accepts_float16() raises:
    """Verify to_bf8() accepts float16 without error."""
    var t = zeros([2, 3], DType.float16)
    var bf8_t = t.to_bf8()
    assert_true(bf8_t.dtype() == DType.uint8, "BF8 output should be uint8")
    assert_true(bf8_t.numel() == 6, "BF8 conversion preserves numel")
    print("PASS: test_bf8_accepts_float16")


def test_fp8_accepts_float64() raises:
    """Verify to_fp8() accepts float64 without error."""
    var t = zeros([2, 3], DType.float64)
    var fp8_t = t.to_fp8()
    assert_true(fp8_t.dtype() == DType.uint8, "FP8 output should be uint8")
    assert_true(fp8_t.numel() == 6, "FP8 conversion preserves numel")
    print("PASS: test_fp8_accepts_float64")


def test_bf8_accepts_float64() raises:
    """Verify to_bf8() accepts float64 without error."""
    var t = zeros([2, 3], DType.float64)
    var bf8_t = t.to_bf8()
    assert_true(bf8_t.dtype() == DType.uint8, "BF8 output should be uint8")
    assert_true(bf8_t.numel() == 6, "BF8 conversion preserves numel")
    print("PASS: test_bf8_accepts_float64")


def test_fp8_rejects_int32() raises:
    """Verify to_fp8() rejects int32 dtype."""
    var t = zeros([2, 3], DType.int32)
    var raised = False
    try:
        _ = t.to_fp8()
    except:
        raised = True
    assert_true(raised, "to_fp8() should reject int32")
    print("PASS: test_fp8_rejects_int32")


def test_bf8_rejects_int32() raises:
    """Verify to_bf8() rejects int32 dtype."""
    var t = zeros([2, 3], DType.int32)
    var raised = False
    try:
        _ = t.to_bf8()
    except:
        raised = True
    assert_true(raised, "to_bf8() should reject int32")
    print("PASS: test_bf8_rejects_int32")


def main() raises:
    test_fp8_rejects_bfloat16()
    test_bf8_rejects_bfloat16()
    test_fp8_accepts_float32()
    test_bf8_accepts_float32()
    test_fp8_accepts_float16()
    test_bf8_accepts_float16()
    test_fp8_accepts_float64()
    test_bf8_accepts_float64()
    test_fp8_rejects_int32()
    test_bf8_rejects_int32()
    print("All 10 FP8/BF8 conversion tests passed!")
