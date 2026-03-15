"""Tests for the bitwise NOT (~) operator on signed integer types (part 2).

Covers Int32 and Int64 with signed-specific boundary values using two's complement
semantics: ~x == -x - 1.

Follow-up from #3293 (issue #3896).

Note: Split from part1 due to Mojo 0.26.1 heap corruption bug that occurs after
~15 cumulative tests. See ADR-009 and Issue #2942.
"""


fn test_int32_not_zero() raises:
    """~Int32(0) should equal -1 (two's complement: ~0 == -1)."""
    var result: Int32 = ~Int32(0)
    if result != -1:
        raise Error("~Int32(0) expected -1, got " + String(result))


fn test_int32_not_neg_one() raises:
    """~Int32(-1) should equal 0 (two's complement: ~(-1) == 0)."""
    var result: Int32 = ~Int32(-1)
    if result != 0:
        raise Error("~Int32(-1) expected 0, got " + String(result))


fn test_int32_not_max() raises:
    """~Int32(2147483647) should equal -2147483648 (two's complement: ~MAX == MIN)."""
    var result: Int32 = ~Int32(2147483647)
    if result != -2147483648:
        raise Error(
            "~Int32(2147483647) expected -2147483648, got " + String(result)
        )


fn test_int32_double_inversion() raises:
    """~~Int32(12345) should equal 12345 (double complement identity)."""
    var val: Int32 = 12345
    if ~~val != val:
        raise Error("~~Int32(12345) expected 12345")


fn test_int64_not_zero() raises:
    """~Int64(0) should equal -1 (two's complement: ~0 == -1)."""
    var result: Int64 = ~Int64(0)
    if result != -1:
        raise Error("~Int64(0) expected -1, got " + String(result))


fn test_int64_not_neg_one() raises:
    """~Int64(-1) should equal 0 (two's complement: ~(-1) == 0)."""
    var result: Int64 = ~Int64(-1)
    if result != 0:
        raise Error("~Int64(-1) expected 0, got " + String(result))


fn test_int64_not_max() raises:
    """~Int64(9223372036854775807) should equal -9223372036854775808 (~MAX == MIN)."""
    var result: Int64 = ~Int64(9223372036854775807)
    if result != -9223372036854775808:
        raise Error(
            "~Int64(9223372036854775807) expected -9223372036854775808, got "
            + String(result)
        )


fn test_int64_double_inversion() raises:
    """~~Int64(999999) should equal 999999 (double complement identity)."""
    var val: Int64 = 999999
    if ~~val != val:
        raise Error("~~Int64(999999) expected 999999")


fn main():
    """Main test runner for Int32 and Int64 bitwise NOT operator tests."""
    try:
        test_int32_not_zero()
        print("OK test_int32_not_zero")
    except e:
        print("FAIL test_int32_not_zero:", e)

    try:
        test_int32_not_neg_one()
        print("OK test_int32_not_neg_one")
    except e:
        print("FAIL test_int32_not_neg_one:", e)

    try:
        test_int32_not_max()
        print("OK test_int32_not_max")
    except e:
        print("FAIL test_int32_not_max:", e)

    try:
        test_int32_double_inversion()
        print("OK test_int32_double_inversion")
    except e:
        print("FAIL test_int32_double_inversion:", e)

    try:
        test_int64_not_zero()
        print("OK test_int64_not_zero")
    except e:
        print("FAIL test_int64_not_zero:", e)

    try:
        test_int64_not_neg_one()
        print("OK test_int64_not_neg_one")
    except e:
        print("FAIL test_int64_not_neg_one:", e)

    try:
        test_int64_not_max()
        print("OK test_int64_not_max")
    except e:
        print("FAIL test_int64_not_max:", e)

    try:
        test_int64_double_inversion()
        print("OK test_int64_double_inversion")
    except e:
        print("FAIL test_int64_double_inversion:", e)

    print("\n=== Int32/Int64 Bitwise NOT Tests Complete ===")
