"""Tests for the bitwise NOT (~) operator on signed integer types (part 1).

Covers Int8 and Int16 with signed-specific boundary values using two's complement
semantics: ~x == -x - 1.

Follow-up from #3293 (issue #3896).

Note: Split from part2 due to Mojo 0.26.1 heap corruption bug that occurs after
~15 cumulative tests.
"""


def test_int8_not_zero() raises:
    """~Int8(0) should equal -1 (two's complement: ~0 == -1)."""
    var result: Int8 = ~Int8(0)
    if result != -1:
        raise Error("~Int8(0) expected -1, got " + String(result))


def test_int8_not_neg_one() raises:
    """~Int8(-1) should equal 0 (two's complement: ~(-1) == 0)."""
    var result: Int8 = ~Int8(-1)
    if result != 0:
        raise Error("~Int8(-1) expected 0, got " + String(result))


def test_int8_not_positive() raises:
    """~Int8(1) should equal -2 (two's complement: ~1 == -2)."""
    var result: Int8 = ~Int8(1)
    if result != -2:
        raise Error("~Int8(1) expected -2, got " + String(result))


def test_int8_not_negative() raises:
    """~Int8(-2) should equal 1 (two's complement: ~(-2) == 1)."""
    var result: Int8 = ~Int8(-2)
    if result != 1:
        raise Error("~Int8(-2) expected 1, got " + String(result))


def test_int8_not_max() raises:
    """~Int8(127) should equal -128 (two's complement)."""
    var result: Int8 = ~Int8(127)
    if result != -128:
        raise Error("~Int8(127) expected -128, got " + String(result))


def test_int8_not_min() raises:
    """~Int8(-128) should equal 127 (two's complement)."""
    var result: Int8 = ~Int8(-128)
    if result != 127:
        raise Error("~Int8(-128) expected 127, got " + String(result))


def test_int8_double_inversion() raises:
    """~~Int8(42) should equal 42 (double complement identity)."""
    var val: Int8 = 42
    if ~~val != val:
        raise Error("~~Int8(42) expected 42")


def test_int16_not_zero() raises:
    """~Int16(0) should equal -1 (two's complement)."""
    var result: Int16 = ~Int16(0)
    if result != -1:
        raise Error("~Int16(0) expected -1, got " + String(result))


def test_int16_not_neg_one() raises:
    """~Int16(-1) should equal 0 (two's complement)."""
    var result: Int16 = ~Int16(-1)
    if result != 0:
        raise Error("~Int16(-1) expected 0, got " + String(result))


def test_int16_not_positive() raises:
    """~Int16(1000) should equal -1001 (two's complement)."""
    var result: Int16 = ~Int16(1000)
    if result != -1001:
        raise Error("~Int16(1000) expected -1001, got " + String(result))


def test_int16_not_max() raises:
    """~Int16(32767) should equal -32768 (two's complement)."""
    var result: Int16 = ~Int16(32767)
    if result != -32768:
        raise Error("~Int16(32767) expected -32768, got " + String(result))


def test_int16_not_min() raises:
    """~Int16(-32768) should equal 32767 (two's complement)."""
    var result: Int16 = ~Int16(-32768)
    if result != 32767:
        raise Error("~Int16(-32768) expected 32767, got " + String(result))


def test_int16_double_inversion() raises:
    """~~Int16(5000) should equal 5000 (double complement identity)."""
    var val: Int16 = 5000
    if ~~val != val:
        raise Error("~~Int16(5000) expected 5000")


def test_int32_not_zero() raises:
    """~Int32(0) should equal -1 (two's complement)."""
    var result: Int32 = ~Int32(0)
    if result != -1:
        raise Error("~Int32(0) expected -1, got " + String(result))


def test_int32_not_neg_one() raises:
    """~Int32(-1) should equal 0 (two's complement)."""
    var result: Int32 = ~Int32(-1)
    if result != 0:
        raise Error("~Int32(-1) expected 0, got " + String(result))


def test_int32_not_positive() raises:
    """~Int32(12345) should equal -12346 (two's complement)."""
    var result: Int32 = ~Int32(12345)
    if result != -12346:
        raise Error("~Int32(12345) expected -12346, got " + String(result))


def test_int32_not_max() raises:
    """~Int32(2147483647) should equal -2147483648 (two's complement)."""
    var result: Int32 = ~Int32(2147483647)
    if result != -2147483648:
        raise Error(
            "~Int32(2147483647) expected -2147483648, got " + String(result)
        )


def test_int32_not_min() raises:
    """~Int32(-2147483648) should equal 2147483647 (two's complement)."""
    var result: Int32 = ~Int32(-2147483648)
    if result != 2147483647:
        raise Error(
            "~Int32(-2147483648) expected 2147483647, got " + String(result)
        )


def test_int32_double_inversion() raises:
    """~~Int32(555555) should equal 555555 (double complement identity)."""
    var val: Int32 = 555555
    if ~~val != val:
        raise Error("~~Int32(555555) expected 555555")


def test_int64_not_zero() raises:
    """~Int64(0) should equal -1 (two's complement)."""
    var result: Int64 = ~Int64(0)
    if result != -1:
        raise Error("~Int64(0) expected -1, got " + String(result))


def test_int64_not_neg_one() raises:
    """~Int64(-1) should equal 0 (two's complement)."""
    var result: Int64 = ~Int64(-1)
    if result != 0:
        raise Error("~Int64(-1) expected 0, got " + String(result))


def test_int64_not_positive() raises:
    """~Int64(999999) should equal -1000000 (two's complement)."""
    var result: Int64 = ~Int64(999999)
    if result != -1000000:
        raise Error("~Int64(999999) expected -1000000, got " + String(result))


def test_int64_not_max() raises:
    """~Int64(9223372036854775807) should equal -9223372036854775808 (two's complement)."""
    var result: Int64 = ~Int64(9223372036854775807)
    if result != -9223372036854775808:
        raise Error(
            "~Int64(9223372036854775807) expected -9223372036854775808, got "
            + String(result)
        )


def test_int64_not_min() raises:
    """~Int64(-9223372036854775808) should equal 9223372036854775807 (two's complement)."""
    var result: Int64 = ~Int64(-9223372036854775808)
    if result != 9223372036854775807:
        raise Error(
            "~Int64(-9223372036854775808) expected 9223372036854775807, got "
            + String(result)
        )


def test_int64_double_inversion() raises:
    """~~Int64(123456789) should equal 123456789 (double complement identity)."""
    var val: Int64 = 123456789
    if ~~val != val:
        raise Error("~~Int64(123456789) expected 123456789")


def main() raises:
    """Run all test_int_bitwise_not tests."""
    print("Running test_int_bitwise_not tests...")

    test_int8_not_zero()
    print("✓ test_int8_not_zero")

    test_int8_not_neg_one()
    print("✓ test_int8_not_neg_one")

    test_int8_not_positive()
    print("✓ test_int8_not_positive")

    test_int8_not_negative()
    print("✓ test_int8_not_negative")

    test_int8_not_max()
    print("✓ test_int8_not_max")

    test_int8_not_min()
    print("✓ test_int8_not_min")

    test_int8_double_inversion()
    print("✓ test_int8_double_inversion")

    test_int16_not_zero()
    print("✓ test_int16_not_zero")

    test_int16_not_neg_one()
    print("✓ test_int16_not_neg_one")

    test_int16_not_positive()
    print("✓ test_int16_not_positive")

    test_int16_not_max()
    print("✓ test_int16_not_max")

    test_int16_not_min()
    print("✓ test_int16_not_min")

    test_int16_double_inversion()
    print("✓ test_int16_double_inversion")

    test_int32_not_zero()
    print("✓ test_int32_not_zero")

    test_int32_not_neg_one()
    print("✓ test_int32_not_neg_one")

    test_int32_not_positive()
    print("✓ test_int32_not_positive")

    test_int32_not_max()
    print("✓ test_int32_not_max")

    test_int32_not_min()
    print("✓ test_int32_not_min")

    test_int32_double_inversion()
    print("✓ test_int32_double_inversion")

    test_int64_not_zero()
    print("✓ test_int64_not_zero")

    test_int64_not_neg_one()
    print("✓ test_int64_not_neg_one")

    test_int64_not_positive()
    print("✓ test_int64_not_positive")

    test_int64_not_max()
    print("✓ test_int64_not_max")

    test_int64_not_min()
    print("✓ test_int64_not_min")

    test_int64_double_inversion()
    print("✓ test_int64_double_inversion")

    print("\nAll test_int_bitwise_not tests passed!")
