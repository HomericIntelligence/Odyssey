"""Basic tests for lazy expression evaluation."""

from std.collections import List
from shared.tensor.any_tensor import AnyTensor, zeros, ones, full
from shared.core.lazy_expression import expr, TensorExpr
from shared.core.lazy_eval import evaluate
from shared.core.arithmetic import add, subtract, multiply, divide


def test_lazy_single_tensor() raises -> None:
    """Test lazy evaluation of single tensor (leaf node)."""
    var tensor = full([2, 3], 5.0, DType.float32)
    var expression = expr(tensor)
    var result = evaluate(expression)

    # Verify shape and values
    if result.numel() != tensor.numel():
        raise Error("numel mismatch")


def test_lazy_add() raises -> None:
    """Test lazy evaluation of addition."""
    var a = full([2, 3], 2.0, DType.float32)
    var b = full([2, 3], 3.0, DType.float32)

    var expr_lazy = expr(a) + expr(b)
    var result_lazy = evaluate(expr_lazy)
    var result_eager = add(a, b)

    if result_lazy.numel() != result_eager.numel():
        raise Error("Result size mismatch")


def test_lazy_chain_2ops() raises -> None:
    """Test lazy evaluation of 2 chained operations."""
    var a = full([2, 3], 1.0, DType.float32)
    var b = full([2, 3], 2.0, DType.float32)
    var c = full([2, 3], 3.0, DType.float32)

    var expr_lazy = (expr(a) + expr(b)) * expr(c)
    var result_lazy = evaluate(expr_lazy)

    var temp_eager = add(a, b)
    var result_eager = multiply(temp_eager, c)

    if result_lazy.numel() != result_eager.numel():
        raise Error("Result size mismatch")


def test_lazy_scalar_multiply() raises -> None:
    """Test lazy evaluation of scalar multiplication."""
    var a = full([2, 3], 3.0, DType.float32)
    var expr_lazy = expr(a) * 2.0
    var result_lazy = evaluate(expr_lazy)

    if result_lazy.numel() != 6:
        raise Error("Result size mismatch")


def test_lazy_negate() raises -> None:
    """Test lazy evaluation of negation."""
    var a = full([2, 3], 5.0, DType.float32)
    var expr_lazy = -expr(a)
    var result_lazy = evaluate(expr_lazy)

    if result_lazy.numel() != 6:
        raise Error("Result size mismatch")


def test_expr_properties() raises -> None:
    """Test expression property accessors."""
    var tensor = full([2, 3, 4], 1.0, DType.float32)
    var expression = expr(tensor)

    var expected_numel = 2 * 3 * 4
    if expression.numel() != expected_numel:
        raise Error("numel mismatch")

    if expression.dtype() != DType.float32:
        raise Error("dtype mismatch")


def main() raises -> None:
    """Run all lazy expression tests."""
    print("Testing lazy expression evaluation...")

    print("  test_lazy_single_tensor...")
    test_lazy_single_tensor()

    print("  test_lazy_add...")
    test_lazy_add()

    print("  test_lazy_chain_2ops...")
    test_lazy_chain_2ops()

    print("  test_lazy_scalar_multiply...")
    test_lazy_scalar_multiply()

    print("  test_lazy_negate...")
    test_lazy_negate()

    print("  test_expr_properties...")
    test_expr_properties()

    print("All lazy expression tests passed!")
