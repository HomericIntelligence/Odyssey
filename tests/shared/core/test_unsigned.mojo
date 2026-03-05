"""Tests for Mojo's built-in unsigned integer types (UInt8, UInt16, UInt32, UInt64).

These tests verify the behavior of Mojo's native unsigned integer builtins,
including arithmetic, bitwise operations, comparisons, boundary values,
and conversions.
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

    print("\n=== Unsigned Integer Type Tests Complete ===")
