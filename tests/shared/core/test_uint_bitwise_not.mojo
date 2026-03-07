"""Tests for the bitwise NOT (~) operator on unsigned integer types.

Covers UInt8, UInt16, UInt32, and UInt64 with boundary values (zero and max),
an alternating-bit mid-range value, and double-inversion identity.

Follow-up from #3081 (issue #3293).
"""


fn test_uint8_not_zero() raises:
    """~UInt8(0) should equal 255 (all bits set)."""
    var result: UInt8 = ~UInt8(0)
    if result != 255:
        raise Error("~UInt8(0) expected 255, got " + String(result))


fn test_uint8_not_max() raises:
    """~UInt8(255) should equal 0 (all bits cleared)."""
    var result: UInt8 = ~UInt8(255)
    if result != 0:
        raise Error("~UInt8(255) expected 0, got " + String(result))


fn test_uint8_not_alternating() raises:
    """~UInt8(0b10101010) should equal 0b01010101 (85)."""
    var result: UInt8 = ~UInt8(0b10101010)
    if result != 85:
        raise Error("~UInt8(0b10101010) expected 85, got " + String(result))


fn test_uint8_double_inversion() raises:
    """~~UInt8(42) should equal 42 (double complement identity)."""
    var val: UInt8 = 42
    if ~~val != val:
        raise Error("~~UInt8(42) expected 42")


fn test_uint16_not_zero() raises:
    """~UInt16(0) should equal 65535 (all bits set)."""
    var result: UInt16 = ~UInt16(0)
    if result != 65535:
        raise Error("~UInt16(0) expected 65535, got " + String(result))


fn test_uint16_not_max() raises:
    """~UInt16(65535) should equal 0 (all bits cleared)."""
    var result: UInt16 = ~UInt16(65535)
    if result != 0:
        raise Error("~UInt16(65535) expected 0, got " + String(result))


fn test_uint16_not_alternating() raises:
    """~UInt16(0xAAAA) should equal 0x5555 (21845)."""
    var result: UInt16 = ~UInt16(0xAAAA)
    if result != 0x5555:
        raise Error("~UInt16(0xAAAA) expected 21845, got " + String(result))


fn test_uint16_double_inversion() raises:
    """~~UInt16(1000) should equal 1000 (double complement identity)."""
    var val: UInt16 = 1000
    if ~~val != val:
        raise Error("~~UInt16(1000) expected 1000")


fn test_uint32_not_zero() raises:
    """~UInt32(0) should equal 4294967295 (all bits set)."""
    var result: UInt32 = ~UInt32(0)
    if result != 4294967295:
        raise Error("~UInt32(0) expected 4294967295, got " + String(result))


fn test_uint32_not_max() raises:
    """~UInt32(4294967295) should equal 0 (all bits cleared)."""
    var result: UInt32 = ~UInt32(4294967295)
    if result != 0:
        raise Error("~UInt32(4294967295) expected 0, got " + String(result))


fn test_uint32_not_alternating() raises:
    """~UInt32(0xAAAAAAAA) should equal 0x55555555 (1431655765)."""
    var result: UInt32 = ~UInt32(0xAAAAAAAA)
    if result != 0x55555555:
        raise Error(
            "~UInt32(0xAAAAAAAA) expected 1431655765, got " + String(result)
        )


fn test_uint32_double_inversion() raises:
    """~~UInt32(12345) should equal 12345 (double complement identity)."""
    var val: UInt32 = 12345
    if ~~val != val:
        raise Error("~~UInt32(12345) expected 12345")


fn test_uint64_not_zero() raises:
    """~UInt64(0) should equal 18446744073709551615 (all bits set)."""
    var result: UInt64 = ~UInt64(0)
    if result != 18446744073709551615:
        raise Error(
            "~UInt64(0) expected 18446744073709551615, got " + String(result)
        )


fn test_uint64_not_max() raises:
    """~UInt64(18446744073709551615) should equal 0 (all bits cleared)."""
    var result: UInt64 = ~UInt64(18446744073709551615)
    if result != 0:
        raise Error(
            "~UInt64(18446744073709551615) expected 0, got " + String(result)
        )


fn test_uint64_not_alternating() raises:
    """~UInt64(0xAAAAAAAAAAAAAAAA) should equal 0x5555555555555555."""
    var result: UInt64 = ~UInt64(0xAAAAAAAAAAAAAAAA)
    if result != 0x5555555555555555:
        raise Error(
            "~UInt64(0xAAAAAAAAAAAAAAAA) expected 0x5555555555555555, got "
            + String(result)
        )


fn test_uint64_double_inversion() raises:
    """~~UInt64(999999) should equal 999999 (double complement identity)."""
    var val: UInt64 = 999999
    if ~~val != val:
        raise Error("~~UInt64(999999) expected 999999")


fn main():
    """Main test runner for UInt bitwise NOT operator tests."""
    try:
        test_uint8_not_zero()
        print("OK test_uint8_not_zero")
    except e:
        print("FAIL test_uint8_not_zero:", e)

    try:
        test_uint8_not_max()
        print("OK test_uint8_not_max")
    except e:
        print("FAIL test_uint8_not_max:", e)

    try:
        test_uint8_not_alternating()
        print("OK test_uint8_not_alternating")
    except e:
        print("FAIL test_uint8_not_alternating:", e)

    try:
        test_uint8_double_inversion()
        print("OK test_uint8_double_inversion")
    except e:
        print("FAIL test_uint8_double_inversion:", e)

    try:
        test_uint16_not_zero()
        print("OK test_uint16_not_zero")
    except e:
        print("FAIL test_uint16_not_zero:", e)

    try:
        test_uint16_not_max()
        print("OK test_uint16_not_max")
    except e:
        print("FAIL test_uint16_not_max:", e)

    try:
        test_uint16_not_alternating()
        print("OK test_uint16_not_alternating")
    except e:
        print("FAIL test_uint16_not_alternating:", e)

    try:
        test_uint16_double_inversion()
        print("OK test_uint16_double_inversion")
    except e:
        print("FAIL test_uint16_double_inversion:", e)

    try:
        test_uint32_not_zero()
        print("OK test_uint32_not_zero")
    except e:
        print("FAIL test_uint32_not_zero:", e)

    try:
        test_uint32_not_max()
        print("OK test_uint32_not_max")
    except e:
        print("FAIL test_uint32_not_max:", e)

    try:
        test_uint32_not_alternating()
        print("OK test_uint32_not_alternating")
    except e:
        print("FAIL test_uint32_not_alternating:", e)

    try:
        test_uint32_double_inversion()
        print("OK test_uint32_double_inversion")
    except e:
        print("FAIL test_uint32_double_inversion:", e)

    try:
        test_uint64_not_zero()
        print("OK test_uint64_not_zero")
    except e:
        print("FAIL test_uint64_not_zero:", e)

    try:
        test_uint64_not_max()
        print("OK test_uint64_not_max")
    except e:
        print("FAIL test_uint64_not_max:", e)

    try:
        test_uint64_not_alternating()
        print("OK test_uint64_not_alternating")
    except e:
        print("FAIL test_uint64_not_alternating:", e)

    try:
        test_uint64_double_inversion()
        print("OK test_uint64_double_inversion")
    except e:
        print("FAIL test_uint64_double_inversion:", e)

    print("\n=== UInt Bitwise NOT Tests Complete ===")
