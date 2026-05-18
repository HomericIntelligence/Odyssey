"""Meta-tests for gradient checker validation.

These tests verify that the gradient checker itself works correctly by using
a simple function with known analytical gradients.

Theory:
    For f(x) = x², the analytical gradient is df/dx = 2x
    This allows us to test:
    - Correct gradient implementation should PASS gradient check
    - Wrong gradient implementation should FAIL gradient check

This meta-testing approach ensures the gradient checker is working properly
before using it to validate layer implementations.

Test Cases:
    1. Correct gradient (2x): Must pass check_gradients
    2. Correct gradient for multiple values
    3. Correct gradient for multidimensional inputs
    4. Wrong gradient (x): Must fail check_gradients
    5. Wrong gradient (3x): Must fail check_gradients
    6. Wrong gradient fails multiple values
    7. Numerical gradient matches analytical
    8. Relative error sensitivity

References:
    - Gradient Checker Design: tests/projectodyssey/testing/test_gradient_checker_meta.mojo
    - Gradient Checker Implementation: src/projectodyssey/testing/gradient_checker.mojo
"""


from std.testing import assert_true, assert_equal
from projectodyssey.testing import (
    NumericalForward,
    NumericalBackward,
    check_gradients,
    compute_numerical_gradient,
    relative_error,
    check_gradients_verbose,
)
from projectodyssey.tensor.any_tensor import (
    AnyTensor,
    zeros,
    ones,
    full,
    zeros_like,
)


def square_forward(x: AnyTensor) raises -> AnyTensor:
    """Forward pass: f(x) = x^2.

    Args:
        x: Input tensor.

    Returns:
        Squared result (x^2): Element-wise squaring.
    """
    var result = zeros_like(x)
    for i in range(x.numel()):
        var val = x._get_float64(i)
        result._set_float64(i, val * val)
    return result^


def square_backward_correct(
    grad_out: AnyTensor, x: AnyTensor
) raises -> AnyTensor:
    """Correct backward pass for f(x) = x^2: df/dx = 2x."""
    var grad_in = zeros_like(x)
    for i in range(x.numel()):
        var x_val = x._get_float64(i)
        var grad_out_val = grad_out._get_float64(i)
        # Correct: df/dx = 2x
        grad_in._set_float64(i, grad_out_val * 2.0 * x_val)
    return grad_in^


def square_backward_wrong_linear(
    grad_out: AnyTensor, x: AnyTensor
) raises -> AnyTensor:
    """Wrong backward pass for f(x) = x^2: Using df/dx = x (incorrect!)."""
    var grad_in = zeros_like(x)
    for i in range(x.numel()):
        var x_val = x._get_float64(i)
        var grad_out_val = grad_out._get_float64(i)
        # WRONG: missing factor of 2
        grad_in._set_float64(i, grad_out_val * x_val)
    return grad_in^


def square_backward_wrong_triple(
    grad_out: AnyTensor, x: AnyTensor
) raises -> AnyTensor:
    """Wrong backward pass for f(x) = x^2: Using df/dx = 3x (incorrect!)."""
    var grad_in = zeros_like(x)
    for i in range(x.numel()):
        var x_val = x._get_float64(i)
        var grad_out_val = grad_out._get_float64(i)
        # WRONG: coefficient of 3 instead of 2
        grad_in._set_float64(i, grad_out_val * 3.0 * x_val)
    return grad_in^


# ============================================================================
# NumericalForward/NumericalBackward struct wrappers
#
# Mojo 0.26.3 makes all inner `def` closures nonescaping, so they cannot be
# passed to functions expecting bare function pointer types. These structs
# wrap the top-level square_forward/backward_* functions via @fieldwise_init
# to satisfy the NumericalForward and NumericalBackward traits.
# ============================================================================


@fieldwise_init
struct _SquareFwd(NumericalForward):
    """Forward pass wrapper: f(x) = x^2."""

    def __call__(self, x: AnyTensor) raises -> AnyTensor:
        return square_forward(x)


@fieldwise_init
struct _SquareBwdCorrect(NumericalBackward):
    """Correct backward wrapper: df/dx = 2x."""

    def __call__(self, grad_out: AnyTensor, x: AnyTensor) raises -> AnyTensor:
        return square_backward_correct(grad_out, x)


@fieldwise_init
struct _SquareBwdWrongLinear(NumericalBackward):
    """Wrong backward wrapper: df/dx = x (incorrect)."""

    def __call__(self, grad_out: AnyTensor, x: AnyTensor) raises -> AnyTensor:
        return square_backward_wrong_linear(grad_out, x)


@fieldwise_init
struct _SquareBwdWrongTriple(NumericalBackward):
    """Wrong backward wrapper: df/dx = 3x (incorrect)."""

    def __call__(self, grad_out: AnyTensor, x: AnyTensor) raises -> AnyTensor:
        return square_backward_wrong_triple(grad_out, x)


def test_gradient_checker_accepts_correct_gradient() raises:
    """Meta-test: Gradient checker should PASS for correct gradient.

    For f(x) = x^2, the gradient df/dx = 2x is correct.
    The gradient checker should verify this successfully.

    Tests:
    - Single positive value (x = 1.0).
    - Gradient check passes without error.
    """
    print("Meta-test: Gradient checker accepts correct gradient...")

    var x = full([1], 1.0, DType.float32)

    var passed = check_gradients(
        _SquareFwd(),
        _SquareBwdCorrect(),
        x,
        epsilon=1e-5,
        tolerance=1e-2,
    )

    assert_true(passed, "Gradient checker should accept correct gradient")
    print("  OK: Gradient checker correctly accepts gradient df/dx = 2x")


def test_gradient_checker_correct_gradient_multiple_values() raises:
    """Meta-test: Correct gradient passes for multiple input values.

    For f(x) = x^2, the gradient df/dx = 2x should work for all values.
    Tests: negative, zero, positive, fractional.
    """
    print("Meta-test: Correct gradient passes multiple input values...")

    var test_values = List[Float64]()
    test_values.append(-1.0)
    test_values.append(0.0)
    test_values.append(1.0)
    test_values.append(2.5)

    for i in range(test_values.__len__()):
        var test_val = test_values[i]
        var x = full([1], test_val, DType.float32)

        var passed = check_gradients(
            _SquareFwd(),
            _SquareBwdCorrect(),
            x,
            epsilon=1e-5,
            tolerance=1e-2,
        )

        assert_true(passed, "Gradient checker should pass for value")

    print("  OK: Gradient checker passes for all input values")


def test_gradient_checker_correct_gradient_multidimensional() raises:
    """Meta-test: Correct gradient passes for multidimensional inputs.

    The gradient checker should work with tensors of any shape.
    Tests 2x3 tensor.
    """
    print("Meta-test: Correct gradient passes for multidimensional...")

    var x = full([2, 3], 1.5, DType.float32)

    var passed = check_gradients(
        _SquareFwd(),
        _SquareBwdCorrect(),
        x,
        epsilon=1e-5,
        tolerance=1e-2,
    )

    assert_true(
        passed,
        "Gradient checker should pass for multidimensional input",
    )
    print("  OK: Gradient checker correctly accepts 2x3 tensor")


def test_gradient_checker_rejects_wrong_gradient_linear() raises:
    """Meta-test: Gradient checker should FAIL for incorrect gradient.

    For f(x) = x^2, using df/dx = x (instead of correct df/dx = 2x)
    is wrong and should be caught by the gradient checker.
    """
    print("Meta-test: Gradient checker rejects wrong gradient (x)...")

    var x = full([1], 1.0, DType.float32)

    var passed = check_gradients(
        _SquareFwd(),
        _SquareBwdWrongLinear(),
        x,
        epsilon=1e-5,
        tolerance=1e-2,
    )

    assert_true(not passed, "Gradient checker should reject wrong gradient")
    print("  OK: Gradient checker correctly rejects df/dx = x")


def test_gradient_checker_rejects_wrong_gradient_triple() raises:
    """Meta-test: Gradient checker should FAIL for coefficient error.

    For f(x) = x^2, using df/dx = 3x (instead of correct df/dx = 2x)
    is wrong and should be caught.
    """
    print("Meta-test: Gradient checker rejects wrong gradient (3x)...")

    var x = full([1], 1.0, DType.float32)

    var passed = check_gradients(
        _SquareFwd(),
        _SquareBwdWrongTriple(),
        x,
        epsilon=1e-5,
        tolerance=1e-2,
    )

    assert_true(not passed, "Gradient checker should reject wrong gradient")
    print("  OK: Gradient checker correctly rejects df/dx = 3x")


def test_gradient_checker_wrong_gradient_multiple_values() raises:
    """Meta-test: Wrong gradient fails for all input values.

    Even with different input values, wrong gradient should fail.
    Tests that error detection is consistent.
    """
    print("Meta-test: Wrong gradient fails for multiple values...")

    var test_values = List[Float64]()
    test_values.append(0.5)
    test_values.append(1.0)
    test_values.append(2.0)

    for i in range(test_values.__len__()):
        var test_val = test_values[i]
        var x = full([1], test_val, DType.float32)

        var passed = check_gradients(
            _SquareFwd(),
            _SquareBwdWrongLinear(),
            x,
            epsilon=1e-5,
            tolerance=1e-2,
        )

        assert_true(
            not passed,
            "Wrong gradient should fail for all input values",
        )

    print("  OK: Wrong gradient consistently fails")


def test_compute_numerical_gradient_matches_analytical() raises:
    """Meta-test: Numerical and analytical gradients should match.

    Using compute_numerical_gradient directly, the numerical gradient
    should closely match the analytical gradient (2x) for the squaring function.
    """
    print("Meta-test: Numerical gradient computation...")

    var x = full([1], 2.0, DType.float32)

    var numerical_grad = compute_numerical_gradient(
        _SquareFwd(), x, epsilon=1e-5
    )

    # Expected: df/dx = 2x = 2*2.0 = 4.0
    var expected = 4.0
    var actual = numerical_grad._get_float64(0)

    var diff = abs(actual - expected)
    assert_true(
        diff < 0.01,
        "Numerical gradient should match analytical",
    )
    print("  OK: Numerical gradient computation correct")


def test_relative_error_sensitivity() raises:
    """Meta-test: Relative error should distinguish correct vs wrong gradients.

    The relative error between correct and wrong gradients should be
    larger than numerical error, making detection reliable.
    """
    print("Meta-test: Relative error sensitivity...")

    var x_val = 1.5
    var correct_grad = 2.0 * x_val
    var wrong_grad = 1.0 * x_val

    var err = relative_error(correct_grad, wrong_grad)

    assert_true(err > 0.4, "Relative error should detect gradient mismatch")
    print("  OK: Relative error correctly identifies mismatch")


def test_gradient_checker_zero_input() raises:
    """Meta-test: Gradient checker should handle zero input correctly.

    For x = 0:
    - f(0) = 0
    - df/dx = 0

    This is an important edge case since the gradient is zero.
    """
    print("Meta-test: Gradient checker handles zero input...")

    var x = full([1], 0.0, DType.float32)

    var passed = check_gradients(
        _SquareFwd(),
        _SquareBwdCorrect(),
        x,
        epsilon=1e-5,
        tolerance=1e-2,
    )

    assert_true(passed, "Gradient checker should handle x=0 correctly")
    print("  OK: Gradient checker handles zero input")


def test_gradient_checker_negative_input() raises:
    """Meta-test: Gradient checker handles negative inputs.

    For x = -2.0:
    - f(-2) = 4
    - df/dx = -4

    Negative gradients should be handled correctly.
    """
    print("Meta-test: Gradient checker handles negative input...")

    var x = full([1], -2.0, DType.float32)

    var passed = check_gradients(
        _SquareFwd(),
        _SquareBwdCorrect(),
        x,
        epsilon=1e-5,
        tolerance=1e-2,
    )

    assert_true(passed, "Gradient checker should handle negative inputs")
    print("  OK: Gradient checker handles negative input")


def test_gradient_checker_large_input() raises:
    """Meta-test: Gradient checker works for moderate input values.

    For x = 5.0:
    - f(5) = 25
    - df/dx = 10

    Should work with moderate absolute values.
    Note: Uses larger tolerance due to accumulated floating-point error.
    """
    print("Meta-test: Gradient checker handles moderate input...")

    var x = full([1], 5.0, DType.float32)

    # Moderate inputs still need larger tolerance for float32 precision
    var passed = check_gradients(
        _SquareFwd(),
        _SquareBwdCorrect(),
        x,
        epsilon=1e-5,
        tolerance=0.05,
    )

    assert_true(passed, "Gradient checker should handle moderate inputs")
    print("  OK: Gradient checker handles moderate input values")


def test_gradient_checker_small_epsilon() raises:
    """Meta-test: Gradient checker works with reasonable small epsilon.

    Smaller epsilon (1e-4) provides better numerical accuracy.
    This tests gradient checker stability.
    """
    print("Meta-test: Gradient checker with smaller epsilon...")

    var x = full([1], 1.0, DType.float32)

    var passed = check_gradients(
        _SquareFwd(),
        _SquareBwdCorrect(),
        x,
        epsilon=1e-4,
        tolerance=1e-2,
    )

    assert_true(passed, "Gradient checker should work with smaller epsilon")
    print("  OK: Gradient checker stable with smaller epsilon")


def test_gradient_checker_large_epsilon() raises:
    """Meta-test: Gradient checker works with larger epsilon.

    Larger epsilon (1e-3) should still pass with correct gradient.
    This tests that checker is not overly sensitive to epsilon choice.
    """
    print("Meta-test: Gradient checker with large epsilon...")

    var x = full([1], 1.0, DType.float32)

    var passed = check_gradients(
        _SquareFwd(),
        _SquareBwdCorrect(),
        x,
        epsilon=1e-3,
        tolerance=1e-2,
    )

    assert_true(passed, "Gradient checker should work with large epsilon")
    print("  OK: Gradient checker stable with large epsilon")


def test_check_gradients_does_not_mutate_input() raises:
    """Regression test: check_gradients must not mutate the original input tensor.

    Verifies that the deep-copy fix for the shallow-copy memory hazard is in
    place. Before the fix, `input.copy()` (a `__copyinit__` shallow copy) shared
    the `_data` buffer with the original tensor, so `_set_float64` calls inside
    check_gradients corrupted the caller's tensor. After the fix, `clone()`
    allocates an independent buffer and the original is never modified.
    """
    print("Memory safety: check_gradients does not mutate input...")

    var x = full([2, 2], 1.0, DType.float32)

    # Snapshot all values before the call
    var before = List[Float64]()
    for i in range(x.numel()):
        before.append(x._get_float64(i))

    _ = check_gradients(
        _SquareFwd(), _SquareBwdCorrect(), x, epsilon=1e-5, tolerance=1e-2
    )

    # Assert every element is unchanged
    for i in range(x.numel()):
        var after_val = x._get_float64(i)
        assert_equal(
            after_val,
            before[i],
            "check_gradients mutated input at index " + String(i),
        )

    print("  OK: check_gradients does not mutate the original input tensor")


def test_check_gradients_verbose_does_not_mutate_input() raises:
    """Regression test: check_gradients_verbose must not mutate the original
    input tensor.

    Parallel test to `test_check_gradients_does_not_mutate_input` but for the
    verbose variant. Verifies that the deep-copy fix is applied to both the
    standard and verbose variants of the gradient checker.

    The verbose variant had the same shallow-copy bug and was also fixed.
    This test ensures the fix is in place for both code paths.
    """
    print("Memory safety: check_gradients_verbose does not mutate input...")

    var x = full([2, 2], 1.0, DType.float32)

    # Snapshot all values before the call
    var before = List[Float64]()
    for i in range(x.numel()):
        before.append(x._get_float64(i))

    _ = check_gradients_verbose(
        _SquareFwd(), _SquareBwdCorrect(), x, epsilon=1e-5, tolerance=1e-2
    )

    # Assert every element is unchanged
    for i in range(x.numel()):
        var after_val = x._get_float64(i)
        assert_equal(
            after_val,
            before[i],
            "check_gradients_verbose mutated input at index " + String(i),
        )

    print(
        "  OK: check_gradients_verbose does not mutate the original input"
        " tensor"
    )


def main() raises:
    """Run all test_gradient_checker_meta tests."""
    print("Running test_gradient_checker_meta tests...")

    test_gradient_checker_accepts_correct_gradient()
    print("✓ test_gradient_checker_accepts_correct_gradient")

    test_gradient_checker_correct_gradient_multiple_values()
    print("✓ test_gradient_checker_correct_gradient_multiple_values")

    test_gradient_checker_correct_gradient_multidimensional()
    print("✓ test_gradient_checker_correct_gradient_multidimensional")

    test_gradient_checker_rejects_wrong_gradient_linear()
    print("✓ test_gradient_checker_rejects_wrong_gradient_linear")

    test_gradient_checker_rejects_wrong_gradient_triple()
    print("✓ test_gradient_checker_rejects_wrong_gradient_triple")

    test_gradient_checker_wrong_gradient_multiple_values()
    print("✓ test_gradient_checker_wrong_gradient_multiple_values")

    test_compute_numerical_gradient_matches_analytical()
    print("✓ test_compute_numerical_gradient_matches_analytical")

    test_relative_error_sensitivity()
    print("✓ test_relative_error_sensitivity")

    test_gradient_checker_zero_input()
    print("✓ test_gradient_checker_zero_input")

    test_gradient_checker_negative_input()
    print("✓ test_gradient_checker_negative_input")

    test_gradient_checker_large_input()
    print("✓ test_gradient_checker_large_input")

    test_gradient_checker_small_epsilon()
    print("✓ test_gradient_checker_small_epsilon")

    test_gradient_checker_large_epsilon()
    print("✓ test_gradient_checker_large_epsilon")

    test_check_gradients_does_not_mutate_input()
    print("✓ test_check_gradients_does_not_mutate_input")

    test_check_gradients_verbose_does_not_mutate_input()
    print("✓ test_check_gradients_verbose_does_not_mutate_input")

    print("\nAll test_gradient_checker_meta tests passed!")
