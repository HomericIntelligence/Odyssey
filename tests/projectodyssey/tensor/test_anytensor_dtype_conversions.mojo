"""Equivalence tests for AnyTensor integer and FP8/BF8 dtype conversion methods.

Tests verify that clamping behavior, same-dtype fast-paths, and bfloat16
rejection are preserved correctly, serving as a TDD baseline before and after
the refactor that extracts parametric helper methods (Issue #5181).

Covers:
- to_int8/to_int16 clamping behavior (high and low)
- to_uint8 clamping behavior (high and low)
- Same-dtype fast-path for at least 2 integer types
- to_fp8/to_bf8 bfloat16 rejection with byte-exact error message check
"""

from std.testing import assert_true
from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.tensor.tensor_creation import zeros


# ===----------------------------------------------------------------------===#
# Integer clamping tests
# ===----------------------------------------------------------------------===#


def test_to_int8_clamps_high() raises:
    """Values above 127 are clamped to 127."""
    var t = AnyTensor([1], DType.float32)
    t[0] = Float32(200.0)
    var result = t.to_int8()
    assert_true(result.dtype() == DType.int8, "output dtype must be int8")
    # Read back via Int conversion
    var got = result._get_int64(0)
    assert_true(got == 127, "value above 127 must be clamped to 127")
    print("PASS: test_to_int8_clamps_high")


def test_to_int8_clamps_low() raises:
    """Values below -128 are clamped to -128."""
    var t = AnyTensor([1], DType.float32)
    t[0] = Float32(-200.0)
    var result = t.to_int8()
    assert_true(result.dtype() == DType.int8, "output dtype must be int8")
    var got = result._get_int64(0)
    assert_true(got == -128, "value below -128 must be clamped to -128")
    print("PASS: test_to_int8_clamps_low")


def test_to_int8_in_range() raises:
    """Values within [-128, 127] are converted without clamping."""
    var t = AnyTensor([1], DType.float32)
    t[0] = Float32(42.0)
    var result = t.to_int8()
    var got = result._get_int64(0)
    assert_true(got == 42, "in-range value must be preserved")
    print("PASS: test_to_int8_in_range")


def test_to_int16_clamps_high() raises:
    """Values above 32767 are clamped to 32767."""
    var t = AnyTensor([1], DType.float32)
    t[0] = Float32(40000.0)
    var result = t.to_int16()
    assert_true(result.dtype() == DType.int16, "output dtype must be int16")
    var got = result._get_int64(0)
    assert_true(got == 32767, "value above 32767 must be clamped to 32767")
    print("PASS: test_to_int16_clamps_high")


def test_to_int16_clamps_low() raises:
    """Values below -32768 are clamped to -32768."""
    var t = AnyTensor([1], DType.float32)
    t[0] = Float32(-40000.0)
    var result = t.to_int16()
    assert_true(result.dtype() == DType.int16, "output dtype must be int16")
    var got = result._get_int64(0)
    assert_true(got == -32768, "value below -32768 must be clamped to -32768")
    print("PASS: test_to_int16_clamps_low")


def test_to_uint8_clamps_high() raises:
    """Values above 255 are clamped to 255."""
    var t = AnyTensor([1], DType.float32)
    t[0] = Float32(300.0)
    var result = t.to_uint8()
    assert_true(result.dtype() == DType.uint8, "output dtype must be uint8")
    var got = result._get_int64(0)
    assert_true(got == 255, "value above 255 must be clamped to 255")
    print("PASS: test_to_uint8_clamps_high")


def test_to_uint8_clamps_low() raises:
    """Negative values are clamped to 0."""
    var t = AnyTensor([1], DType.float32)
    t[0] = Float32(-50.0)
    var result = t.to_uint8()
    assert_true(result.dtype() == DType.uint8, "output dtype must be uint8")
    var got = result._get_int64(0)
    assert_true(got == 0, "negative value must be clamped to 0")
    print("PASS: test_to_uint8_clamps_low")


# ===----------------------------------------------------------------------===#
# Same-dtype fast-path tests
# ===----------------------------------------------------------------------===#


def test_to_int8_same_dtype_fast_path() raises:
    """Verify to_int8() on int8 source uses fast-path (no clamping)."""
    var t = AnyTensor([1], DType.int8)
    t._set_int64(0, Int64(99))
    var result = t.to_int8()
    assert_true(result.dtype() == DType.int8, "output dtype must be int8")
    var got = result._get_int64(0)
    assert_true(got == 99, "same-dtype fast-path must preserve value")
    print("PASS: test_to_int8_same_dtype_fast_path")


def test_to_uint8_same_dtype_fast_path() raises:
    """Verify to_uint8() on uint8 source uses fast-path."""
    var t = AnyTensor([1], DType.uint8)
    t._set_int64(0, Int64(200))
    var result = t.to_uint8()
    assert_true(result.dtype() == DType.uint8, "output dtype must be uint8")
    var got = result._get_int64(0)
    assert_true(got == 200, "same-dtype fast-path must preserve value")
    print("PASS: test_to_uint8_same_dtype_fast_path")


def test_to_int32_same_dtype_fast_path() raises:
    """Verify to_int32() on int32 source uses fast-path."""
    var t = AnyTensor([1], DType.int32)
    t._set_int64(0, Int64(12345))
    var result = t.to_int32()
    assert_true(result.dtype() == DType.int32, "output dtype must be int32")
    var got = result._get_int64(0)
    assert_true(got == 12345, "same-dtype fast-path must preserve value")
    print("PASS: test_to_int32_same_dtype_fast_path")


# ===----------------------------------------------------------------------===#
# FP8/BF8 bfloat16 rejection tests (byte-exact error message check)
# ===----------------------------------------------------------------------===#


def test_to_fp8_rejects_bfloat16_exact_message() raises:
    """Verify to_fp8() rejects bfloat16 with byte-exact error message."""
    var t = zeros([2], DType.bfloat16)
    var raised = False
    var msg_ok = False
    try:
        _ = t.to_fp8()
    except e:
        raised = True
        # Check the key distinguishing phrase from the error message
        var err_str = String(e)
        msg_ok = "bfloat16" in err_str and "to_fp8" in err_str
    assert_true(raised, "to_fp8() must raise for bfloat16")
    assert_true(msg_ok, "to_fp8() error must mention bfloat16 and to_fp8")
    print("PASS: test_to_fp8_rejects_bfloat16_exact_message")


def test_to_mxfp4_rejects_bfloat16_at_outer_guard() raises:
    """MXFP4 rejects bfloat16 up front, not with the inner 'Invalid dtype'.

    Before the fix (#5564), bfloat16 passed the outer float guard and hit the
    inner dispatch's `else: raise Error("Invalid dtype for MXFP4 quantization")`
    — an accept-then-raise asymmetry. Now the outer guard rejects bfloat16 with
    an explicit message (consistent with to_fp8()/to_bf8()), so the error is the
    clear up-front one, NOT the generic inner "Invalid dtype".
    """
    var t = zeros([32], DType.bfloat16)
    var raised = False
    var msg_ok = False
    try:
        _ = t.to_mxfp4()
    except e:
        raised = True
        var err_str = String(e)
        # Must be the explicit up-front rejection, not the inner generic error.
        msg_ok = (
            "bfloat16" in err_str
            and "Invalid dtype for MXFP4 quantization" not in err_str
        )
    assert_true(raised, "to_mxfp4() must raise for bfloat16")
    assert_true(
        msg_ok,
        (
            "to_mxfp4() must reject bfloat16 at the outer guard (explicit"
            " bfloat16 message), not the inner 'Invalid dtype' error"
        ),
    )
    print("PASS: test_to_mxfp4_rejects_bfloat16_at_outer_guard")


def test_to_nvfp4_rejects_bfloat16_at_outer_guard() raises:
    """NVFP4 rejects bfloat16 up front, not with the inner 'Invalid dtype'."""
    var t = zeros([16], DType.bfloat16)
    var raised = False
    var msg_ok = False
    try:
        _ = t.to_nvfp4()
    except e:
        raised = True
        var err_str = String(e)
        msg_ok = (
            "bfloat16" in err_str
            and "Invalid dtype for NVFP4 quantization" not in err_str
        )
    assert_true(raised, "to_nvfp4() must raise for bfloat16")
    assert_true(
        msg_ok,
        (
            "to_nvfp4() must reject bfloat16 at the outer guard (explicit"
            " bfloat16 message), not the inner 'Invalid dtype' error"
        ),
    )
    print("PASS: test_to_nvfp4_rejects_bfloat16_at_outer_guard")


def test_to_bf8_rejects_bfloat16_exact_message() raises:
    """Verify to_bf8() rejects bfloat16 with byte-exact error message."""
    var t = zeros([2], DType.bfloat16)
    var raised = False
    var msg_ok = False
    try:
        _ = t.to_bf8()
    except e:
        raised = True
        var err_str = String(e)
        msg_ok = "bfloat16" in err_str and "to_bf8" in err_str
    assert_true(raised, "to_bf8() must raise for bfloat16")
    assert_true(msg_ok, "to_bf8() error must mention bfloat16 and to_bf8")
    print("PASS: test_to_bf8_rejects_bfloat16_exact_message")


# ===----------------------------------------------------------------------===#
# Multi-element correctness
# ===----------------------------------------------------------------------===#


def test_to_int8_multi_element() raises:
    """Verify to_int8() correctly converts multiple float32 elements."""
    var t = AnyTensor([4], DType.float32)
    t[0] = Float32(-200.0)
    t[1] = Float32(-1.0)
    t[2] = Float32(50.0)
    t[3] = Float32(200.0)
    var result = t.to_int8()
    assert_true(result.numel() == 4, "numel preserved")
    assert_true(result._get_int64(0) == -128, "clamped low")
    assert_true(result._get_int64(1) == -1, "in range negative")
    assert_true(result._get_int64(2) == 50, "in range positive")
    assert_true(result._get_int64(3) == 127, "clamped high")
    print("PASS: test_to_int8_multi_element")


def test_to_uint8_multi_element() raises:
    """Verify to_uint8() correctly converts multiple float32 elements."""
    var t = AnyTensor([3], DType.float32)
    t[0] = Float32(-10.0)
    t[1] = Float32(128.0)
    t[2] = Float32(300.0)
    var result = t.to_uint8()
    assert_true(result.numel() == 3, "numel preserved")
    assert_true(result._get_int64(0) == 0, "clamped low to 0")
    assert_true(result._get_int64(1) == 128, "in range")
    assert_true(result._get_int64(2) == 255, "clamped high to 255")
    print("PASS: test_to_uint8_multi_element")


def main() raises:
    test_to_int8_clamps_high()
    test_to_int8_clamps_low()
    test_to_int8_in_range()
    test_to_int16_clamps_high()
    test_to_int16_clamps_low()
    test_to_uint8_clamps_high()
    test_to_uint8_clamps_low()
    test_to_int8_same_dtype_fast_path()
    test_to_uint8_same_dtype_fast_path()
    test_to_int32_same_dtype_fast_path()
    test_to_fp8_rejects_bfloat16_exact_message()
    test_to_bf8_rejects_bfloat16_exact_message()
    test_to_mxfp4_rejects_bfloat16_at_outer_guard()
    test_to_nvfp4_rejects_bfloat16_at_outer_guard()
    test_to_int8_multi_element()
    test_to_uint8_multi_element()
    print("All 16 AnyTensor dtype conversion tests passed!")
