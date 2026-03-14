"""Tests for Mojo's built-in unsigned integer types (UInt8, UInt16, UInt32, UInt64).

These tests verify the behavior of Mojo's native unsigned integer builtins,
including arithmetic, bitwise operations, comparisons, boundary values,
overflow and underflow wrapping behavior, and conversions.

Wrapping Semantics:
    Unsigned integers use modular (wrap-around) arithmetic on overflow and underflow:

    - Overflow: max_value + 1 wraps to 0. For example UInt8(255) + 1 == UInt8(0).
    - Underflow: 0 - 1 wraps to max_value. For example UInt8(0) - 1 == UInt8(255).

    This matches C/C++ unsigned integer behavior and is defined (not undefined behavior).
    Test functions `test_*_overflow` and `test_*_underflow` validate this contract for
    each unsigned type. See also: `test_uint8_boundary_values` for edge-case coverage.
"""


fn test_uint8_construction() raises:
    """Test UInt8 construction from literals and zero value."""
    var zero: UInt8 = 0
    var one: UInt8 = 1
    var max_val: UInt8 = 255

    if zero != 0:
        raise Error("UInt8 zero construction failed")
    if one != 1:
        raise Error("UInt8 one construction failed")
    if max_val != 255:
        raise Error("UInt8 max value construction failed")


fn test_uint16_construction() raises:
    """Test UInt16 construction from literals and boundary values."""
    var zero: UInt16 = 0
    var one: UInt16 = 1
    var max_val: UInt16 = 65535

    if zero != 0:
        raise Error("UInt16 zero construction failed")
    if one != 1:
        raise Error("UInt16 one construction failed")
    if max_val != 65535:
        raise Error("UInt16 max value construction failed")


fn test_uint32_construction() raises:
    """Test UInt32 construction from literals and boundary values."""
    var zero: UInt32 = 0
    var one: UInt32 = 1
    var max_val: UInt32 = 4294967295

    if zero != 0:
        raise Error("UInt32 zero construction failed")
    if one != 1:
        raise Error("UInt32 one construction failed")
    if max_val != 4294967295:
        raise Error("UInt32 max value construction failed")


fn test_uint64_construction() raises:
    """Test UInt64 construction from literals and boundary values."""
    var zero: UInt64 = 0
    var one: UInt64 = 1
    var large: UInt64 = 18446744073709551615

    if zero != 0:
        raise Error("UInt64 zero construction failed")
    if one != 1:
        raise Error("UInt64 one construction failed")
    if large != 18446744073709551615:
        raise Error("UInt64 max value construction failed")


fn test_uint8_arithmetic() raises:
    """Test UInt8 addition, subtraction, multiplication, and division."""
    var a: UInt8 = 10
    var b: UInt8 = 3

    if a + b != 13:
        raise Error("UInt8 addition failed")
    if a - b != 7:
        raise Error("UInt8 subtraction failed")
    if a * b != 30:
        raise Error("UInt8 multiplication failed")
    if a // b != 3:
        raise Error("UInt8 integer division failed")
    if a % b != 1:
        raise Error("UInt8 modulo failed")


fn test_uint16_arithmetic() raises:
    """Test UInt16 arithmetic operations."""
    var a: UInt16 = 1000
    var b: UInt16 = 7

    if a + b != 1007:
        raise Error("UInt16 addition failed")
    if a - b != 993:
        raise Error("UInt16 subtraction failed")
    if a * b != 7000:
        raise Error("UInt16 multiplication failed")
    if a // b != 142:
        raise Error("UInt16 integer division failed")
    if a % b != 6:
        raise Error("UInt16 modulo failed")


fn test_uint32_arithmetic() raises:
    """Test UInt32 arithmetic operations."""
    var a: UInt32 = 100000
    var b: UInt32 = 3

    if a + b != 100003:
        raise Error("UInt32 addition failed")
    if a - b != 99997:
        raise Error("UInt32 subtraction failed")
    if a * b != 300000:
        raise Error("UInt32 multiplication failed")
    if a // b != 33333:
        raise Error("UInt32 integer division failed")
    if a % b != 1:
        raise Error("UInt32 modulo failed")


fn test_uint64_arithmetic() raises:
    """Test UInt64 arithmetic operations with large values."""
    var a: UInt64 = 10000000000
    var b: UInt64 = 3

    if a + b != 10000000003:
        raise Error("UInt64 addition failed")
    if a - b != 9999999997:
        raise Error("UInt64 subtraction failed")
    if a * b != 30000000000:
        raise Error("UInt64 multiplication failed")
    if a // b != 3333333333:
        raise Error("UInt64 integer division failed")
    if a % b != 1:
        raise Error("UInt64 modulo failed")


fn test_uint8_bitwise() raises:
    """Test UInt8 bitwise operations."""
    var a: UInt8 = 0b10110100  # 180
    var b: UInt8 = 0b01101100  # 108

    if a & b != 0b00100100:
        raise Error("UInt8 AND failed")
    if a | b != 0b11111100:
        raise Error("UInt8 OR failed")
    if a ^ b != 0b11011000:
        raise Error("UInt8 XOR failed")
    if a << 1 != 0b01101000:
        raise Error("UInt8 left shift failed")
    if a >> 1 != 0b01011010:
        raise Error("UInt8 right shift failed")


fn test_uint32_bitwise() raises:
    """Test UInt32 bitwise operations."""
    var a: UInt32 = 0xFF00FF00
    var b: UInt32 = 0x00FF00FF

    if a & b != 0:
        raise Error("UInt32 AND with complement failed")
    if a | b != 0xFFFFFFFF:
        raise Error("UInt32 OR with complement failed")
    if a ^ b != 0xFFFFFFFF:
        raise Error("UInt32 XOR with complement failed")
    if a >> 8 != 0x00FF00FF:
        raise Error("UInt32 right shift by 8 failed")
    if b << 8 != 0xFF00FF00:
        raise Error("UInt32 left shift by 8 failed")


fn test_uint8_comparisons() raises:
    """Test UInt8 comparison operators."""
    var a: UInt8 = 10
    var b: UInt8 = 20
    var c: UInt8 = 10

    if a == b:
        raise Error("UInt8 == should be false for different values")
    if not (a == c):
        raise Error("UInt8 == should be true for equal values")
    if not (a != b):
        raise Error("UInt8 != failed")
    if not (a < b):
        raise Error("UInt8 < failed")
    if not (a <= b):
        raise Error("UInt8 <= failed (less)")
    if not (a <= c):
        raise Error("UInt8 <= failed (equal)")
    if not (b > a):
        raise Error("UInt8 > failed")
    if not (b >= a):
        raise Error("UInt8 >= failed (greater)")
    if not (c >= a):
        raise Error("UInt8 >= failed (equal)")


fn test_uint32_comparisons() raises:
    """Test UInt32 comparison operators with larger values."""
    var small: UInt32 = 0
    var large: UInt32 = 4294967295

    if not (small < large):
        raise Error("UInt32 min < max comparison failed")
    if not (large > small):
        raise Error("UInt32 max > min comparison failed")
    if not (small == 0):
        raise Error("UInt32 zero equality failed")
    if not (large == 4294967295):
        raise Error("UInt32 max equality failed")


fn test_uint_widening_conversion() raises:
    """Test widening conversions between unsigned integer types."""
    var u8: UInt8 = 200
    var u16: UInt16 = u8.cast[DType.uint16]()
    var u32: UInt32 = u16.cast[DType.uint32]()
    var u64: UInt64 = u32.cast[DType.uint64]()

    if u16 != 200:
        raise Error("UInt8 -> UInt16 widening conversion failed")
    if u32 != 200:
        raise Error("UInt16 -> UInt32 widening conversion failed")
    if u64 != 200:
        raise Error("UInt32 -> UInt64 widening conversion failed")


fn test_uint_to_int_conversion() raises:
    """Test conversion between unsigned integers and Int."""
    var u8: UInt8 = 255
    var u32: UInt32 = 1000
    var u64: UInt64 = 9999

    var i_from_u8 = Int(u8)
    var i_from_u32 = Int(u32)
    var i_from_u64 = Int(u64)

    if i_from_u8 != 255:
        raise Error("UInt8 -> Int conversion failed")
    if i_from_u32 != 1000:
        raise Error("UInt32 -> Int conversion failed")
    if i_from_u64 != 9999:
        raise Error("UInt64 -> Int conversion failed")


fn test_uint_from_int_conversion() raises:
    """Test constructing unsigned integers from Int values."""
    var i: Int = 42
    var u8: UInt8 = i
    var u16: UInt16 = i
    var u32: UInt32 = i
    var u64: UInt64 = i

    if u8 != 42:
        raise Error("Int -> UInt8 conversion failed")
    if u16 != 42:
        raise Error("Int -> UInt16 conversion failed")
    if u32 != 42:
        raise Error("Int -> UInt32 conversion failed")
    if u64 != 42:
        raise Error("Int -> UInt64 conversion failed")


fn test_uint8_zero_operations() raises:
    """Test UInt8 operations involving zero."""
    var zero: UInt8 = 0
    var val: UInt8 = 42

    if zero + val != 42:
        raise Error("UInt8 zero addition failed")
    if val - zero != 42:
        raise Error("UInt8 zero subtraction failed")
    if zero * val != 0:
        raise Error("UInt8 zero multiplication failed")
    if zero & val != 0:
        raise Error("UInt8 zero AND failed")
    if zero | val != 42:
        raise Error("UInt8 zero OR failed")
    if zero ^ val != 42:
        raise Error("UInt8 zero XOR failed")


fn test_uint64_large_values() raises:
    """Test UInt64 operations with large values."""
    var billion: UInt64 = 1000000000
    var trillion: UInt64 = billion * billion

    if trillion != 1000000000000000000:
        raise Error("UInt64 large multiplication failed")

    var half: UInt64 = trillion // 2
    if half != 500000000000000000:
        raise Error("UInt64 large division failed")


fn test_uint_type_min_max_operations() raises:
    """Test arithmetic at boundary values."""
    # UInt8 max - 1 operations
    var u8_near_max: UInt8 = 254
    if u8_near_max + 1 != 255:
        raise Error("UInt8 near-max addition failed")

    # UInt16 max - 1 operations
    var u16_near_max: UInt16 = 65534
    if u16_near_max + 1 != 65535:
        raise Error("UInt16 near-max addition failed")

    # UInt32 subtraction from max
    var u32_max: UInt32 = 4294967295
    if u32_max - 1 != 4294967294:
        raise Error("UInt32 max - 1 failed")


fn test_uint_narrowing_conversion() raises:
    """Test narrowing conversions that truncate via modulo 2^N semantics.

    When casting a UInt64 value > 255 to UInt8, the result is the low 8 bits
    of the original value, equivalent to value % 256.
    """
    # 256 % 256 = 0
    var v256: UInt64 = 256
    if v256.cast[DType.uint8]() != 0:
        raise Error("UInt64(256).cast[DType.uint8]() should be 0")

    # 257 % 256 = 1
    var v257: UInt64 = 257
    if v257.cast[DType.uint8]() != 1:
        raise Error("UInt64(257).cast[DType.uint8]() should be 1")

    # 511 % 256 = 255
    var v511: UInt64 = 511
    if v511.cast[DType.uint8]() != 255:
        raise Error("UInt64(511).cast[DType.uint8]() should be 255")

    # 512 % 256 = 0
    var v512: UInt64 = 512
    if v512.cast[DType.uint8]() != 0:
        raise Error("UInt64(512).cast[DType.uint8]() should be 0")

    # 255 fits exactly — no truncation
    var v255: UInt64 = 255
    if v255.cast[DType.uint8]() != 255:
        raise Error("UInt64(255).cast[DType.uint8]() should be 255")

    # 0 is a no-op
    var v0: UInt64 = 0
    if v0.cast[DType.uint8]() != 0:
        raise Error("UInt64(0).cast[DType.uint8]() should be 0")


fn test_uint8_overflow_wrap() raises:
    """Test UInt8 addition wraps from 255 to 0."""
    var result: UInt8 = UInt8(255) + UInt8(1)
    if result != 0:
        raise Error(
            "UInt8 overflow wrap failed: expected 0, got " + String(result)
        )


fn test_uint8_underflow_wrap() raises:
    """Test UInt8 subtraction wraps from 0 to 255."""
    var result: UInt8 = UInt8(0) - UInt8(1)
    if result != 255:
        raise Error(
            "UInt8 underflow wrap failed: expected 255, got " + String(result)
        )


fn test_uint16_overflow_wrap() raises:
    """Test UInt16 addition wraps from 65535 to 0."""
    var result: UInt16 = UInt16(65535) + UInt16(1)
    if result != 0:
        raise Error(
            "UInt16 overflow wrap failed: expected 0, got " + String(result)
        )


fn test_uint16_underflow_wrap() raises:
    """Test UInt16 subtraction wraps from 0 to 65535."""
    var result: UInt16 = UInt16(0) - UInt16(1)
    if result != 65535:
        raise Error(
            "UInt16 underflow wrap failed: expected 65535, got "
            + String(result)
        )


fn test_uint32_overflow_wrap() raises:
    """Test UInt32 addition wraps from 4294967295 to 0."""
    var result: UInt32 = UInt32(4294967295) + UInt32(1)
    if result != 0:
        raise Error(
            "UInt32 overflow wrap failed: expected 0, got " + String(result)
        )


fn test_uint32_underflow_wrap() raises:
    """Test UInt32 subtraction wraps from 0 to 4294967295."""
    var result: UInt32 = UInt32(0) - UInt32(1)
    if result != 4294967295:
        raise Error(
            "UInt32 underflow wrap failed: expected 4294967295, got "
            + String(result)
        )


fn test_uint64_overflow_wrap() raises:
    """Test UInt64 addition wraps from max to 0."""
    var result: UInt64 = UInt64(18446744073709551615) + UInt64(1)
    if result != 0:
        raise Error(
            "UInt64 overflow wrap failed: expected 0, got " + String(result)
        )


fn test_uint64_underflow_wrap() raises:
    """Test UInt64 subtraction wraps from 0 to max."""
    var result: UInt64 = UInt64(0) - UInt64(1)
    if result != 18446744073709551615:
        raise Error(
            "UInt64 underflow wrap failed: expected 18446744073709551615, got "
            + String(result)
        )


fn test_uint8_overflow_wrap_add_chain() raises:
    """Test UInt8 overflow wraps mid-range: 250 + 10 == 4."""
    var result: UInt8 = UInt8(250) + UInt8(10)
    if result != 4:
        raise Error(
            "UInt8 mid-range overflow wrap failed: expected 4, got "
            + String(result)
        )


fn test_uint16_overflow_wrap_add_chain() raises:
    """Test UInt16 overflow wraps mid-range: 65530 + 10 == 4."""
    var result: UInt16 = UInt16(65530) + UInt16(10)
    if result != 4:
        raise Error(
            "UInt16 mid-range overflow wrap failed: expected 4, got "
            + String(result)
        )


fn test_uint32_overflow_wrap_add_chain() raises:
    """Test UInt32 overflow wraps mid-range: 4294967290 + 10 == 4."""
    var result: UInt32 = UInt32(4294967290) + UInt32(10)
    if result != 4:
        raise Error(
            "UInt32 mid-range overflow wrap failed: expected 4, got "
            + String(result)
        )


fn test_uint64_overflow_wrap_add_chain() raises:
    """Test UInt64 overflow wraps mid-range: max-5 + 10 == 4."""
    var result: UInt64 = UInt64(18446744073709551610) + UInt64(10)
    if result != 4:
        raise Error(
            "UInt64 mid-range overflow wrap failed: expected 4, got "
            + String(result)
        )


fn test_uint8_overflow_wrap_multiply() raises:
    """Test UInt8 multiplication overflow wraps: 128 * 2 == 0 (2^8 mod 256)."""
    var result: UInt8 = UInt8(128) * UInt8(2)
    if result != 0:
        raise Error(
            "UInt8 multiply overflow wrap failed: expected 0, got "
            + String(result)
        )


fn main():
    """Main test runner for unsigned integer type tests."""
    try:
        test_uint8_construction()
        print("OK test_uint8_construction")
    except e:
        print("FAIL test_uint8_construction:", e)

    try:
        test_uint16_construction()
        print("OK test_uint16_construction")
    except e:
        print("FAIL test_uint16_construction:", e)

    try:
        test_uint32_construction()
        print("OK test_uint32_construction")
    except e:
        print("FAIL test_uint32_construction:", e)

    try:
        test_uint64_construction()
        print("OK test_uint64_construction")
    except e:
        print("FAIL test_uint64_construction:", e)

    try:
        test_uint8_arithmetic()
        print("OK test_uint8_arithmetic")
    except e:
        print("FAIL test_uint8_arithmetic:", e)

    try:
        test_uint16_arithmetic()
        print("OK test_uint16_arithmetic")
    except e:
        print("FAIL test_uint16_arithmetic:", e)

    try:
        test_uint32_arithmetic()
        print("OK test_uint32_arithmetic")
    except e:
        print("FAIL test_uint32_arithmetic:", e)

    try:
        test_uint64_arithmetic()
        print("OK test_uint64_arithmetic")
    except e:
        print("FAIL test_uint64_arithmetic:", e)

    try:
        test_uint8_bitwise()
        print("OK test_uint8_bitwise")
    except e:
        print("FAIL test_uint8_bitwise:", e)

    try:
        test_uint32_bitwise()
        print("OK test_uint32_bitwise")
    except e:
        print("FAIL test_uint32_bitwise:", e)

    try:
        test_uint8_comparisons()
        print("OK test_uint8_comparisons")
    except e:
        print("FAIL test_uint8_comparisons:", e)

    try:
        test_uint32_comparisons()
        print("OK test_uint32_comparisons")
    except e:
        print("FAIL test_uint32_comparisons:", e)

    try:
        test_uint_widening_conversion()
        print("OK test_uint_widening_conversion")
    except e:
        print("FAIL test_uint_widening_conversion:", e)

    try:
        test_uint_narrowing_conversion()
        print("OK test_uint_narrowing_conversion")
    except e:
        print("FAIL test_uint_narrowing_conversion:", e)

    try:
        test_uint_to_int_conversion()
        print("OK test_uint_to_int_conversion")
    except e:
        print("FAIL test_uint_to_int_conversion:", e)

    try:
        test_uint_from_int_conversion()
        print("OK test_uint_from_int_conversion")
    except e:
        print("FAIL test_uint_from_int_conversion:", e)

    try:
        test_uint8_zero_operations()
        print("OK test_uint8_zero_operations")
    except e:
        print("FAIL test_uint8_zero_operations:", e)

    try:
        test_uint64_large_values()
        print("OK test_uint64_large_values")
    except e:
        print("FAIL test_uint64_large_values:", e)

    try:
        test_uint_type_min_max_operations()
        print("OK test_uint_type_min_max_operations")
    except e:
        print("FAIL test_uint_type_min_max_operations:", e)

    try:
        test_uint8_overflow_wrap()
        print("OK test_uint8_overflow_wrap")
    except e:
        print("FAIL test_uint8_overflow_wrap:", e)

    try:
        test_uint8_underflow_wrap()
        print("OK test_uint8_underflow_wrap")
    except e:
        print("FAIL test_uint8_underflow_wrap:", e)

    try:
        test_uint16_overflow_wrap()
        print("OK test_uint16_overflow_wrap")
    except e:
        print("FAIL test_uint16_overflow_wrap:", e)

    try:
        test_uint16_underflow_wrap()
        print("OK test_uint16_underflow_wrap")
    except e:
        print("FAIL test_uint16_underflow_wrap:", e)

    try:
        test_uint32_overflow_wrap()
        print("OK test_uint32_overflow_wrap")
    except e:
        print("FAIL test_uint32_overflow_wrap:", e)

    try:
        test_uint32_underflow_wrap()
        print("OK test_uint32_underflow_wrap")
    except e:
        print("FAIL test_uint32_underflow_wrap:", e)

    try:
        test_uint64_overflow_wrap()
        print("OK test_uint64_overflow_wrap")
    except e:
        print("FAIL test_uint64_overflow_wrap:", e)

    try:
        test_uint64_underflow_wrap()
        print("OK test_uint64_underflow_wrap")
    except e:
        print("FAIL test_uint64_underflow_wrap:", e)

    try:
        test_uint8_overflow_wrap_add_chain()
        print("OK test_uint8_overflow_wrap_add_chain")
    except e:
        print("FAIL test_uint8_overflow_wrap_add_chain:", e)

    try:
        test_uint16_overflow_wrap_add_chain()
        print("OK test_uint16_overflow_wrap_add_chain")
    except e:
        print("FAIL test_uint16_overflow_wrap_add_chain:", e)

    try:
        test_uint32_overflow_wrap_add_chain()
        print("OK test_uint32_overflow_wrap_add_chain")
    except e:
        print("FAIL test_uint32_overflow_wrap_add_chain:", e)

    try:
        test_uint64_overflow_wrap_add_chain()
        print("OK test_uint64_overflow_wrap_add_chain")
    except e:
        print("FAIL test_uint64_overflow_wrap_add_chain:", e)

    try:
        test_uint8_overflow_wrap_multiply()
        print("OK test_uint8_overflow_wrap_multiply")
    except e:
        print("FAIL test_uint8_overflow_wrap_multiply:", e)

    print("\n=== Unsigned Integer Type Tests Complete ===")
