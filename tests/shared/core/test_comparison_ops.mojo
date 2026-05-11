"""Tests for AnyTensor comparison operations.

# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under

Tests comparison operations following the Array API Standard:
equal, not_equal.
All operations return boolean tensors (DType.bool).
"""


from shared.tensor.any_tensor import AnyTensor, full, ones, zeros
from shared.core.comparison import (
    equal,
    greater,
    greater_equal,
    less,
    less_equal,
    not_equal,
)
from tests.shared.conftest import (
    assert_dtype,
    assert_numel,
    assert_value_at,
    assert_all_values,
)


def test_equal_same_values() raises:
    """Test equal with identical values."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 2.0, DType.float32)
    var b = full(shape, 2.0, DType.float32)
    var c = equal(a, b)

    assert_dtype(c, DType.bool, "Result should be bool dtype")
    assert_numel(c, 5, "Result should have 5 elements")
    # All values should be True (1)
    for i in range(5):
        assert_value_at(c, i, 1.0, 1e-6, "Equal values should return True")


def test_equal_different_values() raises:
    """Test equal with different values."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 2.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)
    var c = equal(a, b)

    assert_dtype(c, DType.bool, "Result should be bool dtype")
    # All values should be False (0)
    for i in range(5):
        assert_value_at(c, i, 0.0, 1e-6, "Different values should return False")


def test_equal_with_dunder() raises:
    """Test equal using == operator."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 2.0, DType.float32)
    var b = full(shape, 2.0, DType.float32)
    var c = a == b

    assert_dtype(c, DType.bool, "Result should be bool dtype")
    for i in range(5):
        assert_value_at(c, i, 1.0, 1e-6, "a == b should work via __eq__")


def test_not_equal_same_values() raises:
    """Test not_equal with identical values."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 2.0, DType.float32)
    var b = full(shape, 2.0, DType.float32)
    var c = not_equal(a, b)

    assert_dtype(c, DType.bool, "Result should be bool dtype")
    # All values should be False (0)
    for i in range(5):
        assert_value_at(
            c, i, 0.0, 1e-6, "Equal values should return False for !="
        )


def test_not_equal_different_values() raises:
    """Test not_equal with different values."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 2.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)
    var c = not_equal(a, b)

    assert_dtype(c, DType.bool, "Result should be bool dtype")
    # All values should be True (1)
    for i in range(5):
        assert_value_at(
            c, i, 1.0, 1e-6, "Different values should return True for !="
        )


def test_not_equal_with_dunder() raises:
    """Test not_equal using != operator."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 2.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)
    var c = a != b

    assert_dtype(c, DType.bool, "Result should be bool dtype")
    for i in range(5):
        assert_value_at(c, i, 1.0, 1e-6, "a != b should work via __ne__")


def test_less_true() raises:
    """Test less when first tensor has smaller values."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 2.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)
    var c = less(a, b)

    assert_dtype(c, DType.bool, "Result should be bool dtype")
    # All values should be True (1)
    for i in range(5):
        assert_value_at(c, i, 1.0, 1e-6, "2.0 < 3.0 should be True")


def test_less_false() raises:
    """Test less when first tensor has larger values."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 5.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)
    var c = less(a, b)

    assert_dtype(c, DType.bool, "Result should be bool dtype")
    # All values should be False (0)
    for i in range(5):
        assert_value_at(c, i, 0.0, 1e-6, "5.0 < 3.0 should be False")


def test_less_with_dunder() raises:
    """Test less using < operator."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 2.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)
    var c = a < b

    assert_dtype(c, DType.bool, "Result should be bool dtype")
    for i in range(5):
        assert_value_at(c, i, 1.0, 1e-6, "a < b should work via __lt__")


def test_less_equal_true_less() raises:
    """Test less_equal when values are less."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 2.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)
    var c = less_equal(a, b)

    assert_dtype(c, DType.bool, "Result should be bool dtype")
    for i in range(5):
        assert_value_at(c, i, 1.0, 1e-6, "2.0 <= 3.0 should be True")


def test_less_equal_true_equal() raises:
    """Test less_equal when values are equal."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 3.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)
    var c = less_equal(a, b)

    assert_dtype(c, DType.bool, "Result should be bool dtype")
    for i in range(5):
        assert_value_at(c, i, 1.0, 1e-6, "3.0 <= 3.0 should be True")


def test_less_equal_with_dunder() raises:
    """Test less_equal using <= operator."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 2.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)
    var c = a <= b

    assert_dtype(c, DType.bool, "Result should be bool dtype")
    for i in range(5):
        assert_value_at(c, i, 1.0, 1e-6, "a <= b should work via __le__")


def test_greater_true() raises:
    """Test greater when first tensor has larger values."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 5.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)
    var c = greater(a, b)

    assert_dtype(c, DType.bool, "Result should be bool dtype")
    for i in range(5):
        assert_value_at(c, i, 1.0, 1e-6, "5.0 > 3.0 should be True")


def test_greater_false() raises:
    """Test greater when first tensor has smaller values."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 2.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)
    var c = greater(a, b)

    assert_dtype(c, DType.bool, "Result should be bool dtype")
    for i in range(5):
        assert_value_at(c, i, 0.0, 1e-6, "2.0 > 3.0 should be False")


def test_greater_with_dunder() raises:
    """Test greater using > operator."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 5.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)
    var c = a > b

    assert_dtype(c, DType.bool, "Result should be bool dtype")
    for i in range(5):
        assert_value_at(c, i, 1.0, 1e-6, "a > b should work via __gt__")


def test_greater_equal_true_greater() raises:
    """Test greater_equal when values are greater."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 5.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)
    var c = greater_equal(a, b)

    assert_dtype(c, DType.bool, "Result should be bool dtype")
    for i in range(5):
        assert_value_at(c, i, 1.0, 1e-6, "5.0 >= 3.0 should be True")


def test_greater_equal_true_equal() raises:
    """Test greater_equal when values are equal."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 3.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)
    var c = greater_equal(a, b)

    assert_dtype(c, DType.bool, "Result should be bool dtype")
    for i in range(5):
        assert_value_at(c, i, 1.0, 1e-6, "3.0 >= 3.0 should be True")


def test_greater_equal_with_dunder() raises:
    """Test greater_equal using >= operator."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 5.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)
    var c = a >= b

    assert_dtype(c, DType.bool, "Result should be bool dtype")
    for i in range(5):
        assert_value_at(c, i, 1.0, 1e-6, "a >= b should work via __ge__")


def test_comparison_with_negatives() raises:
    """Test comparisons with negative values."""
    var shape = List[Int]()
    shape.append(5)

    var a = full(shape, -2.0, DType.float32)
    var b = full(shape, -5.0, DType.float32)

    # -2.0 > -5.0 should be True
    var c_greater = greater(a, b)
    for i in range(5):
        assert_value_at(c_greater, i, 1.0, 1e-6, "-2.0 > -5.0 should be True")

    # -2.0 < -5.0 should be False
    var c_less = less(a, b)
    for i in range(5):
        assert_value_at(c_less, i, 0.0, 1e-6, "-2.0 < -5.0 should be False")


def main() raises:
    """Run all test_comparison_ops tests."""
    print("Running test_comparison_ops tests...")

    test_equal_same_values()
    print("✓ test_equal_same_values")

    test_equal_different_values()
    print("✓ test_equal_different_values")

    test_equal_with_dunder()
    print("✓ test_equal_with_dunder")

    test_not_equal_same_values()
    print("✓ test_not_equal_same_values")

    test_not_equal_different_values()
    print("✓ test_not_equal_different_values")

    test_not_equal_with_dunder()
    print("✓ test_not_equal_with_dunder")

    test_less_true()
    print("✓ test_less_true")

    test_less_false()
    print("✓ test_less_false")

    test_less_with_dunder()
    print("✓ test_less_with_dunder")

    test_less_equal_true_less()
    print("✓ test_less_equal_true_less")

    test_less_equal_true_equal()
    print("✓ test_less_equal_true_equal")

    test_less_equal_with_dunder()
    print("✓ test_less_equal_with_dunder")

    test_greater_true()
    print("✓ test_greater_true")

    test_greater_false()
    print("✓ test_greater_false")

    test_greater_with_dunder()
    print("✓ test_greater_with_dunder")

    test_greater_equal_true_greater()
    print("✓ test_greater_equal_true_greater")

    test_greater_equal_true_equal()
    print("✓ test_greater_equal_true_equal")

    test_greater_equal_with_dunder()
    print("✓ test_greater_equal_with_dunder")

    test_comparison_with_negatives()
    print("✓ test_comparison_with_negatives")

    print("\nAll test_comparison_ops tests passed!")
