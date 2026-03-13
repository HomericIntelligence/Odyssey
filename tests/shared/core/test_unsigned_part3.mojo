# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_unsigned.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Tests for unsigned integer bitwise ops and overflow edge cases (Part 3).

Note: Split from monolithic test file due to Mojo 0.26.1 heap corruption
bug that occurs after ~15 cumulative tests. See Issue #2942.

Tests cover:
- UInt16 bitwise operations (#3892)
- UInt64 bitwise operations (#3893)
- UInt16 overflow boundary arithmetic (#3894)
- UInt32 overflow boundary arithmetic (#3895)
- UInt64 overflow boundary arithmetic (#3896)
"""


fn test_uint16_bitwise_and() raises:
    """Test UInt16 bitwise AND. Closes #3892."""
    var a: UInt16 = 0xFF0F
    var b: UInt16 = 0xF0FF
    var result = a & b
    if result != 0xF00F:
        raise Error("UInt16 AND: expected 0xF00F, got " + String(result))


fn test_uint16_bitwise_or_xor() raises:
    """Test UInt16 bitwise OR and XOR. Closes #3892."""
    var a: UInt16 = 0xFF00
    var b: UInt16 = 0x00FF

    var or_result = a | b
    if or_result != 0xFFFF:
        raise Error("UInt16 OR: expected 0xFFFF, got " + String(or_result))

    var xor_result = a ^ b
    if xor_result != 0xFFFF:
        raise Error("UInt16 XOR: expected 0xFFFF, got " + String(xor_result))

    # XOR with same value = 0
    var xor_same = a ^ a
    if xor_same != 0:
        raise Error("UInt16 XOR self: expected 0, got " + String(xor_same))


fn test_uint64_bitwise_and() raises:
    """Test UInt64 bitwise AND. Closes #3893."""
    var a: UInt64 = 0xFFFFFFFF00000000
    var b: UInt64 = 0x00000000FFFFFFFF
    var result = a & b
    if result != 0:
        raise Error("UInt64 AND: expected 0, got " + String(result))


fn test_uint64_bitwise_or_xor() raises:
    """Test UInt64 bitwise OR and XOR. Closes #3893."""
    var a: UInt64 = 0xFFFFFFFF00000000
    var b: UInt64 = 0x00000000FFFFFFFF
    var or_result = a | b
    if or_result != 0xFFFFFFFFFFFFFFFF:
        raise Error("UInt64 OR: expected max, got " + String(or_result))

    var xor_result = a ^ b
    if xor_result != 0xFFFFFFFFFFFFFFFF:
        raise Error("UInt64 XOR: expected max, got " + String(xor_result))


fn test_uint16_shift_operations() raises:
    """Test UInt16 shift left/right. Closes #3892."""
    var val: UInt16 = 1
    var shifted_left = val << 15
    if shifted_left != 32768:
        raise Error(
            "UInt16 shift left 15: expected 32768, got " + String(shifted_left)
        )

    var shifted_right = shifted_left >> 15
    if shifted_right != 1:
        raise Error(
            "UInt16 shift right 15: expected 1, got " + String(shifted_right)
        )


fn test_uint16_overflow_boundary() raises:
    """Test UInt16 overflow at exact boundary. Closes #3894."""
    # 65535 + 1 = 0
    var r1: UInt16 = UInt16(65535) + UInt16(1)
    if r1 != 0:
        raise Error("UInt16 65535+1: expected 0, got " + String(r1))

    # 65534 + 2 = 0
    var r2: UInt16 = UInt16(65534) + UInt16(2)
    if r2 != 0:
        raise Error("UInt16 65534+2: expected 0, got " + String(r2))


fn test_uint32_overflow_boundary() raises:
    """Test UInt32 overflow at exact boundary. Closes #3895."""
    var r1: UInt32 = UInt32(4294967295) + UInt32(1)
    if r1 != 0:
        raise Error("UInt32 max+1: expected 0, got " + String(r1))

    var r2: UInt32 = UInt32(4294967294) + UInt32(2)
    if r2 != 0:
        raise Error("UInt32 max-1+2: expected 0, got " + String(r2))


fn test_uint64_overflow_boundary() raises:
    """Test UInt64 overflow at exact boundary. Closes #3896."""
    var r1: UInt64 = UInt64(18446744073709551615) + UInt64(1)
    if r1 != 0:
        raise Error("UInt64 max+1: expected 0, got " + String(r1))


fn test_uint_not_operations() raises:
    """Test bitwise NOT for various unsigned types. Closes #3892, #3893."""
    var u8: UInt8 = ~UInt8(0)
    if u8 != 255:
        raise Error("NOT UInt8(0): expected 255, got " + String(u8))

    var u16: UInt16 = ~UInt16(0)
    if u16 != 65535:
        raise Error("NOT UInt16(0): expected 65535, got " + String(u16))

    var u64: UInt64 = ~UInt64(0)
    if u64 != 18446744073709551615:
        raise Error("NOT UInt64(0): expected max, got " + String(u64))


fn main():
    """Run unsigned integer bitwise and overflow boundary tests (Part 3)."""
    try:
        test_uint16_bitwise_and()
        print("OK test_uint16_bitwise_and")
    except e:
        print("FAIL test_uint16_bitwise_and:", e)

    try:
        test_uint16_bitwise_or_xor()
        print("OK test_uint16_bitwise_or_xor")
    except e:
        print("FAIL test_uint16_bitwise_or_xor:", e)

    try:
        test_uint64_bitwise_and()
        print("OK test_uint64_bitwise_and")
    except e:
        print("FAIL test_uint64_bitwise_and:", e)

    try:
        test_uint64_bitwise_or_xor()
        print("OK test_uint64_bitwise_or_xor")
    except e:
        print("FAIL test_uint64_bitwise_or_xor:", e)

    try:
        test_uint16_shift_operations()
        print("OK test_uint16_shift_operations")
    except e:
        print("FAIL test_uint16_shift_operations:", e)

    try:
        test_uint16_overflow_boundary()
        print("OK test_uint16_overflow_boundary")
    except e:
        print("FAIL test_uint16_overflow_boundary:", e)

    try:
        test_uint32_overflow_boundary()
        print("OK test_uint32_overflow_boundary")
    except e:
        print("FAIL test_uint32_overflow_boundary:", e)

    try:
        test_uint64_overflow_boundary()
        print("OK test_uint64_overflow_boundary")
    except e:
        print("FAIL test_uint64_overflow_boundary:", e)

    try:
        test_uint_not_operations()
        print("OK test_uint_not_operations")
    except e:
        print("FAIL test_uint_not_operations:", e)

    print("\n=== Unsigned Part 3 Tests Complete ===")
