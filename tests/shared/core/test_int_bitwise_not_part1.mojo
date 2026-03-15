"""Tests for the bitwise NOT (~) operator on signed integer types (part 1).

Covers Int8 and Int16 with signed-specific boundary values using two's complement
semantics: ~x == -x - 1.

Follow-up from #3293 (issue #3896).

Note: Split from part2 due to Mojo 0.26.1 heap corruption bug that occurs after
~15 cumulative tests. See ADR-009 and Issue #2942.
"""


fn test_int8_not_zero() raises:
    """~Int8(0) should equal -1 (two's complement: ~0 == -1)."""
    var result: Int8 = ~Int8(0)
    if result != -1:
        raise Error("~Int8(0) expected -1, got " + String(result))


fn test_int8_not_neg_one() raises:
    """~Int8(-1) should equal 0 (two's complement: ~(-1) == 0)."""
    var result: Int8 = ~Int8(-1)
    if result != 0:
        raise Error("~Int8(-1) expected 0, got " + String(result))


fn test_int8_not_max() raises:
    """~Int8(127) should equal -128 (two's complement boundary: ~MAX == MIN)."""
    var result: Int8 = ~Int8(127)
    if result != -128:
        raise Error("~Int8(127) expected -128, got " + String(result))


fn test_int8_double_inversion() raises:
    """~~Int8(42) should equal 42 (double complement identity)."""
    var val: Int8 = 42
    if ~~val != val:
        raise Error("~~Int8(42) expected 42")


fn test_int16_not_zero() raises:
    """~Int16(0) should equal -1 (two's complement: ~0 == -1)."""
    var result: Int16 = ~Int16(0)
    if result != -1:
        raise Error("~Int16(0) expected -1, got " + String(result))


fn test_int16_not_neg_one() raises:
    """~Int16(-1) should equal 0 (two's complement: ~(-1) == 0)."""
    var result: Int16 = ~Int16(-1)
    if result != 0:
        raise Error("~Int16(-1) expected 0, got " + String(result))


fn test_int16_not_max() raises:
    """~Int16(32767) should equal -32768 (two's complement boundary: ~MAX == MIN)."""
    var result: Int16 = ~Int16(32767)
    if result != -32768:
        raise Error("~Int16(32767) expected -32768, got " + String(result))


fn test_int16_double_inversion() raises:
    """~~Int16(1000) should equal 1000 (double complement identity)."""
    var val: Int16 = 1000
    if ~~val != val:
        raise Error("~~Int16(1000) expected 1000")


fn main():
    """Main test runner for Int8 and Int16 bitwise NOT operator tests."""
    try:
        test_int8_not_zero()
        print("OK test_int8_not_zero")
    except e:
        print("FAIL test_int8_not_zero:", e)

    try:
        test_int8_not_neg_one()
        print("OK test_int8_not_neg_one")
    except e:
        print("FAIL test_int8_not_neg_one:", e)

    try:
        test_int8_not_max()
        print("OK test_int8_not_max")
    except e:
        print("FAIL test_int8_not_max:", e)

    try:
        test_int8_double_inversion()
        print("OK test_int8_double_inversion")
    except e:
        print("FAIL test_int8_double_inversion:", e)

    try:
        test_int16_not_zero()
        print("OK test_int16_not_zero")
    except e:
        print("FAIL test_int16_not_zero:", e)

    try:
        test_int16_not_neg_one()
        print("OK test_int16_not_neg_one")
    except e:
        print("FAIL test_int16_not_neg_one:", e)

    try:
        test_int16_not_max()
        print("OK test_int16_not_max")
    except e:
        print("FAIL test_int16_not_max:", e)

    try:
        test_int16_double_inversion()
        print("OK test_int16_double_inversion")
    except e:
        print("FAIL test_int16_double_inversion:", e)

    print("\n=== Int8/Int16 Bitwise NOT Tests Complete ===")
