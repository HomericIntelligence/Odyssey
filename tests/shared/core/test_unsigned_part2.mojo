# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_unsigned.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for unsigned integer wrapping and narrowing (Part 2).

Note: Split from monolithic test file due to Mojo 0.26.1 heap corruption
bug that occurs after ~15 cumulative tests. See Issue #2942.

Tests cover:
- Multiply overflow wrapping for UInt16/32/64 (#3673)
- Subtract underflow wrapping for UInt16/32/64 (#3673)
- Narrowing conversions to UInt16/32 (#3675)
- Boundary arithmetic overflow (#3676)
"""


fn test_uint16_multiply_overflow() raises:
    """Test UInt16 multiplication overflow wraps. Closes #3673."""
    # 256 * 256 = 65536 = 0 mod 2^16
    var result: UInt16 = UInt16(256) * UInt16(256)
    if result != 0:
        raise Error(
            "UInt16 multiply overflow: expected 0, got " + String(result)
        )

    # 1000 * 100 = 100000, mod 65536 = 34464
    var result2: UInt16 = UInt16(1000) * UInt16(100)
    if result2 != 34464:
        raise Error(
            "UInt16 multiply overflow: expected 34464, got " + String(result2)
        )


fn test_uint32_multiply_overflow() raises:
    """Test UInt32 multiplication overflow wraps. Closes #3673."""
    # 65536 * 65536 = 2^32 = 0 mod 2^32
    var result: UInt32 = UInt32(65536) * UInt32(65536)
    if result != 0:
        raise Error(
            "UInt32 multiply overflow: expected 0, got " + String(result)
        )


fn test_uint64_multiply_overflow() raises:
    """Test UInt64 multiplication overflow wraps. Closes #3673."""
    # 2^32 * 2^32 = 2^64 = 0 mod 2^64
    var result: UInt64 = UInt64(4294967296) * UInt64(4294967296)
    if result != 0:
        raise Error(
            "UInt64 multiply overflow: expected 0, got " + String(result)
        )


fn test_uint16_subtract_underflow() raises:
    """Test UInt16 subtraction underflow wrapping. Closes #3673."""
    # 5 - 10 = 65531 (wraps)
    var result: UInt16 = UInt16(5) - UInt16(10)
    if result != 65531:
        raise Error(
            "UInt16 subtract underflow: expected 65531, got " + String(result)
        )


fn test_uint32_subtract_underflow() raises:
    """Test UInt32 subtraction underflow wrapping. Closes #3673."""
    var result: UInt32 = UInt32(5) - UInt32(10)
    if result != 4294967291:
        raise Error(
            "UInt32 subtract underflow: expected 4294967291, got "
            + String(result)
        )


fn test_uint64_subtract_underflow() raises:
    """Test UInt64 subtraction underflow wrapping. Closes #3673."""
    var result: UInt64 = UInt64(5) - UInt64(10)
    if result != 18446744073709551611:
        raise Error(
            "UInt64 subtract underflow: expected 18446744073709551611, got "
            + String(result)
        )


fn test_narrowing_to_uint16() raises:
    """Test narrowing conversion truncates to UInt16. Closes #3675."""
    # 65536 -> 0 (truncate to 16 bits)
    var val: UInt16 = UInt16(UInt32(65536))
    if val != 0:
        raise Error("Narrowing 65536 to UInt16: expected 0, got " + String(val))

    # 65537 -> 1
    var val2: UInt16 = UInt16(UInt32(65537))
    if val2 != 1:
        raise Error(
            "Narrowing 65537 to UInt16: expected 1, got " + String(val2)
        )


fn test_narrowing_to_uint32() raises:
    """Test narrowing conversion truncates to UInt32. Closes #3675."""
    # 2^32 -> 0
    var val: UInt32 = UInt32(UInt64(4294967296))
    if val != 0:
        raise Error("Narrowing 2^32 to UInt32: expected 0, got " + String(val))

    # 2^32 + 1 -> 1
    var val2: UInt32 = UInt32(UInt64(4294967297))
    if val2 != 1:
        raise Error(
            "Narrowing 2^32+1 to UInt32: expected 1, got " + String(val2)
        )


fn test_uint8_boundary_arithmetic() raises:
    """Test UInt8 arithmetic at boundary values. Closes #3676."""
    # 255 + 1 == 0
    var r1: UInt8 = UInt8(255) + UInt8(1)
    if r1 != 0:
        raise Error("UInt8 255+1: expected 0, got " + String(r1))

    # 0 - 1 == 255
    var r2: UInt8 = UInt8(0) - UInt8(1)
    if r2 != 255:
        raise Error("UInt8 0-1: expected 255, got " + String(r2))

    # 255 * 255 == 1 (mod 256: 65025 mod 256 = 1)
    var r3: UInt8 = UInt8(255) * UInt8(255)
    if r3 != 1:
        raise Error("UInt8 255*255: expected 1, got " + String(r3))


fn main():
    """Run unsigned integer wrapping and narrowing tests (Part 2)."""
    try:
        test_uint16_multiply_overflow()
        print("OK test_uint16_multiply_overflow")
    except e:
        print("FAIL test_uint16_multiply_overflow:", e)

    try:
        test_uint32_multiply_overflow()
        print("OK test_uint32_multiply_overflow")
    except e:
        print("FAIL test_uint32_multiply_overflow:", e)

    try:
        test_uint64_multiply_overflow()
        print("OK test_uint64_multiply_overflow")
    except e:
        print("FAIL test_uint64_multiply_overflow:", e)

    try:
        test_uint16_subtract_underflow()
        print("OK test_uint16_subtract_underflow")
    except e:
        print("FAIL test_uint16_subtract_underflow:", e)

    try:
        test_uint32_subtract_underflow()
        print("OK test_uint32_subtract_underflow")
    except e:
        print("FAIL test_uint32_subtract_underflow:", e)

    try:
        test_uint64_subtract_underflow()
        print("OK test_uint64_subtract_underflow")
    except e:
        print("FAIL test_uint64_subtract_underflow:", e)

    try:
        test_narrowing_to_uint16()
        print("OK test_narrowing_to_uint16")
    except e:
        print("FAIL test_narrowing_to_uint16:", e)

    try:
        test_narrowing_to_uint32()
        print("OK test_narrowing_to_uint32")
    except e:
        print("FAIL test_narrowing_to_uint32:", e)

    try:
        test_uint8_boundary_arithmetic()
        print("OK test_uint8_boundary_arithmetic")
    except e:
        print("FAIL test_uint8_boundary_arithmetic:", e)

    print("\n=== Unsigned Part 2 Tests Complete ===")
