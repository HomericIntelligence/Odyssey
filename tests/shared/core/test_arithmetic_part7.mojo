"""Tests for operator overloading (dunders) and dtype preservation.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_arithmetic.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests cover:
- Operator overloading: __add__, __sub__, __mul__
- Chained and complex expressions
- DType preservation: float32, float64, int32
"""

from tests.shared.conftest import (
    assert_all_values,
    assert_dtype,
)
from shared.core.extensor import ExTensor, zeros, ones, full
from shared.core.arithmetic import (
    add,
    multiply,
)


fn test_dunder_add() raises:
    """Test __add__ operator overloading (a + b)."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 3.0, DType.float32)
    var b = full(shape, 2.0, DType.float32)
    var c = a + b

    assert_all_values(c, 5.0, 1e-6, "a + b should work via __add__")


fn test_dunder_sub() raises:
    """Test __sub__ operator overloading (a - b)."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 7.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)
    var c = a - b

    assert_all_values(c, 4.0, 1e-6, "a - b should work via __sub__")


fn test_dunder_mul() raises:
    """Test __mul__ operator overloading (a * b)."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 4.0, DType.float32)
    var b = full(shape, 2.0, DType.float32)
    var c = a * b

    assert_all_values(c, 8.0, 1e-6, "a * b should work via __mul__")


fn test_chained_operations() raises:
    """Test chained operations with multiple operators."""
    var shape = List[Int]()
    shape.append(5)
    var a = full(shape, 2.0, DType.float32)
    var b = full(shape, 3.0, DType.float32)
    var c = full(shape, 1.0, DType.float32)

    # (a + b) * c = (2 + 3) * 1 = 5
    var result = (a + b) * c
    assert_all_values(result, 5.0, 1e-6, "(2 + 3) * 1 should be 5")


fn test_complex_expression() raises:
    """Test complex expression with multiple operations."""
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var a = ones(shape, DType.float32)
    var b = full(shape, 2.0, DType.float32)
    var c = full(shape, 3.0, DType.float32)

    # a + b * c = 1 + 2 * 3 = 1 + 6 = 7
    var result = a + b * c
    assert_all_values(result, 7.0, 1e-6, "1 + 2 * 3 should be 7")


fn test_add_preserves_dtype_float32() raises:
    """Test that add preserves float32 dtype."""
    var shape = List[Int]()
    shape.append(5)
    var a = ones(shape, DType.float32)
    var b = ones(shape, DType.float32)
    var c = add(a, b)

    assert_dtype(c, DType.float32, "Result should preserve float32 dtype")


fn test_add_preserves_dtype_float64() raises:
    """Test that add preserves float64 dtype."""
    var shape = List[Int]()
    shape.append(5)
    var a = ones(shape, DType.float64)
    var b = ones(shape, DType.float64)
    var c = add(a, b)

    assert_dtype(c, DType.float64, "Result should preserve float64 dtype")


fn test_multiply_preserves_dtype_int32() raises:
    """Test that multiply preserves int32 dtype."""
    var shape = List[Int]()
    shape.append(5)
    var a = ones(shape, DType.int32)
    var b = full(shape, 2.0, DType.int32)
    var c = multiply(a, b)

    assert_dtype(c, DType.int32, "Result should preserve int32 dtype")


fn main() raises:
    """Run operator overloading and dtype preservation tests."""
    print("Running operator overloading and dtype tests (part 7)...")

    test_dunder_add()
    print("    ✓ test_dunder_add")
    test_dunder_sub()
    print("    ✓ test_dunder_sub")
    test_dunder_mul()
    print("    ✓ test_dunder_mul")
    test_chained_operations()
    print("    ✓ test_chained_operations")
    test_complex_expression()
    print("    ✓ test_complex_expression")
    test_add_preserves_dtype_float32()
    print("    ✓ test_add_preserves_dtype_float32")
    test_add_preserves_dtype_float64()
    print("    ✓ test_add_preserves_dtype_float64")
    test_multiply_preserves_dtype_int32()
    print("    ✓ test_multiply_preserves_dtype_int32")

    print("\nAll arithmetic part 7 tests passed! (8 tests)")
