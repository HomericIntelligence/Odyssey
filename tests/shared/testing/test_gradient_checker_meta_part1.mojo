# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_gradient_checker_meta.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Meta-tests for gradient checker validation (Part 1 of 2).

These tests verify that the gradient checker itself works correctly by using
a simple function with known analytical gradients.

Theory:
    For f(x) = x², the analytical gradient is df/dx = 2x
    This allows us to test:
    - Correct gradient implementation should PASS gradient check
    - Wrong gradient implementation should FAIL gradient check

This meta-testing approach ensures the gradient checker is working properly
before using it to validate layer implementations.

Test Cases (Part 1):
    1. Correct gradient (2x): Must pass check_gradients
    2. Correct gradient for multiple values
    3. Correct gradient for multidimensional inputs
    4. Wrong gradient (x): Must fail check_gradients
    5. Wrong gradient (3x): Must fail check_gradients
    6. Wrong gradient fails multiple values
    7. Numerical gradient matches analytical
    8. Relative error sensitivity

References:
    - Gradient Checker Design: tests/shared/testing/test_gradient_checker_meta.mojo
    - Gradient Checker Implementation: shared/testing/gradient_checker.mojo
"""

from testing import assert_true, assert_equal
from shared.testing import (
    check_gradients,
    compute_numerical_gradient,
    relative_error,
)
from shared.core import ExTensor, zeros, ones, full, zeros_like


# ============================================================================
# Test Helper Functions (Simple f(x) = x^2)
# ============================================================================


fn square_forward(input: ExTensor) raises escaping -> ExTensor:
    """Forward pass: f(x) = x^2

    Args:
        input: Input tensor.

    Returns:
        input^2: Element-wise squaring.
    """
    var result = zeros_like(input)
    for i in range(input.numel()):
        var val = input._get_float64(i)
        result._set_float64(i, val * val)
    return result^


fn square_backward_correct(
    grad_out: ExTensor, input: ExTensor
) raises escaping -> ExTensor:
    """Correct backward pass for f(x) = x^2: df/dx = 2x.

    This is the mathematically correct gradient.

    Args:
        grad_out: Gradient from upstream (typically ones_like(output)).
        input: Input tensor (needed to compute gradient).

    Returns:
        Gradient w.r.t input: grad_out * 2x.
    """
    var grad_in = zeros_like(input)
    for i in range(input.numel()):
        var x_val = input._get_float64(i)
        var grad_out_val = grad_out._get_float64(i)
        # Correct: df/dx = 2x
        grad_in._set_float64(i, grad_out_val * 2.0 * x_val)
    return grad_in^


fn square_backward_wrong_linear(
    grad_out: ExTensor, input: ExTensor
) raises escaping -> ExTensor:
    """Wrong backward pass for f(x) = x^2: Using df/dx = x (incorrect!).

    This is intentionally wrong to test that gradient checker catches errors.
    The correct derivative is 2x, not x.

    Args:
        grad_out: Gradient from upstream.
        input: Input tensor.

    Returns:
        Wrong gradient: grad_out * x (missing factor of 2).
    """
    var grad_in = zeros_like(input)
    for i in range(input.numel()):
        var x_val = input._get_float64(i)
        var grad_out_val = grad_out._get_float64(i)
        # WRONG: missing factor of 2
        grad_in._set_float64(i, grad_out_val * x_val)
    return grad_in^


fn square_backward_wrong_triple(
    grad_out: ExTensor, input: ExTensor
) raises escaping -> ExTensor:
    """Wrong backward pass for f(x) = x^2: Using df/dx = 3x (incorrect!).

    This is intentionally wrong to test gradient checker sensitivity.

    Args:
        grad_out: Gradient from upstream.
        input: Input tensor.

    Returns:
        Wrong gradient: grad_out * 3x (wrong coefficient).
    """
    var grad_in = zeros_like(input)
    for i in range(input.numel()):
        var x_val = input._get_float64(i)
        var grad_out_val = grad_out._get_float64(i)
        # WRONG: coefficient of 3 instead of 2
        grad_in._set_float64(i, grad_out_val * 3.0 * x_val)
    return grad_in^


# ============================================================================
# Meta-Tests: Correct Gradient Should Pass
# ============================================================================


fn test_gradient_checker_accepts_correct_gradient() raises:
    """Meta-test: Gradient checker should PASS for correct gradient.

    For f(x) = x^2, the gradient df/dx = 2x is correct.
    The gradient checker should verify this successfully.

    Tests:
    - Single positive value (x = 1.0).
    - Gradient check passes without error.
    """
    print("Meta-test: Gradient checker accepts correct gradient...")

    var x = full([1], 1.0, DType.float32)

    fn forward(t: ExTensor) raises escaping -> ExTensor:
        return square_forward(t)^

    fn backward(grad: ExTensor, inp: ExTensor) raises escaping -> ExTensor:
        return square_backward_correct(grad, inp)^

    var passed = check_gradients(
        forward,
        backward,
        x,
        epsilon=1e-5,
        tolerance=1e-2,
    )

    assert_true(passed, "Gradient checker should accept correct gradient")
    print("  OK: Gradient checker correctly accepts gradient df/dx = 2x")


fn test_gradient_checker_correct_gradient_multiple_values() raises:
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

        fn forward(t: ExTensor) raises escaping -> ExTensor:
            return square_forward(t)^

        fn backward(grad: ExTensor, inp: ExTensor) raises escaping -> ExTensor:
            return square_backward_correct(grad, inp)^

        var passed = check_gradients(
            forward,
            backward,
            x,
            epsilon=1e-5,
            tolerance=1e-2,
        )

        assert_true(passed, "Gradient checker should pass for value")

    print("  OK: Gradient checker passes for all input values")


fn test_gradient_checker_correct_gradient_multidimensional() raises:
    """Meta-test: Correct gradient passes for multidimensional inputs.

    The gradient checker should work with tensors of any shape.
    Tests 2x3 tensor.
    """
    print("Meta-test: Correct gradient passes for multidimensional...")

    var x = full([2, 3], 1.5, DType.float32)

    fn forward(t: ExTensor) raises escaping -> ExTensor:
        return square_forward(t)^

    fn backward(grad: ExTensor, inp: ExTensor) raises escaping -> ExTensor:
        return square_backward_correct(grad, inp)^

    var passed = check_gradients(
        forward,
        backward,
        x,
        epsilon=1e-5,
        tolerance=1e-2,
    )

    assert_true(
        passed,
        "Gradient checker should pass for multidimensional input",
    )
    print("  OK: Gradient checker correctly accepts 2x3 tensor")


# ============================================================================
# Meta-Tests: Wrong Gradients Should Fail
# ============================================================================


fn test_gradient_checker_rejects_wrong_gradient_linear() raises:
    """Meta-test: Gradient checker should FAIL for incorrect gradient.

    For f(x) = x^2, using df/dx = x (instead of correct df/dx = 2x)
    is wrong and should be caught by the gradient checker.
    """
    print("Meta-test: Gradient checker rejects wrong gradient (x)...")

    var x = full([1], 1.0, DType.float32)

    fn forward(t: ExTensor) raises escaping -> ExTensor:
        return square_forward(t)^

    fn backward(grad: ExTensor, inp: ExTensor) raises escaping -> ExTensor:
        return square_backward_wrong_linear(grad, inp)^

    var passed = check_gradients(
        forward,
        backward,
        x,
        epsilon=1e-5,
        tolerance=1e-2,
    )

    assert_true(not passed, "Gradient checker should reject wrong gradient")
    print("  OK: Gradient checker correctly rejects df/dx = x")


fn test_gradient_checker_rejects_wrong_gradient_triple() raises:
    """Meta-test: Gradient checker should FAIL for coefficient error.

    For f(x) = x^2, using df/dx = 3x (instead of correct df/dx = 2x)
    is wrong and should be caught.
    """
    print("Meta-test: Gradient checker rejects wrong gradient (3x)...")

    var x = full([1], 1.0, DType.float32)

    fn forward(t: ExTensor) raises escaping -> ExTensor:
        return square_forward(t)^

    fn backward(grad: ExTensor, inp: ExTensor) raises escaping -> ExTensor:
        return square_backward_wrong_triple(grad, inp)^

    var passed = check_gradients(
        forward,
        backward,
        x,
        epsilon=1e-5,
        tolerance=1e-2,
    )

    assert_true(not passed, "Gradient checker should reject wrong gradient")
    print("  OK: Gradient checker correctly rejects df/dx = 3x")


fn test_gradient_checker_wrong_gradient_multiple_values() raises:
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

        fn forward(t: ExTensor) raises escaping -> ExTensor:
            return square_forward(t)^

        fn backward(grad: ExTensor, inp: ExTensor) raises escaping -> ExTensor:
            return square_backward_wrong_linear(grad, inp)^

        var passed = check_gradients(
            forward,
            backward,
            x,
            epsilon=1e-5,
            tolerance=1e-2,
        )

        assert_true(
            not passed,
            "Wrong gradient should fail for all input values",
        )

    print("  OK: Wrong gradient consistently fails")


# ============================================================================
# Meta-Tests: Numerical Gradient Computation
# ============================================================================


fn test_compute_numerical_gradient_matches_analytical() raises:
    """Meta-test: Numerical and analytical gradients should match.

    Using compute_numerical_gradient directly, the numerical gradient
    should closely match the analytical gradient (2x) for the squaring function.
    """
    print("Meta-test: Numerical gradient computation...")

    var x = full([1], 2.0, DType.float32)

    fn forward(t: ExTensor) raises escaping -> ExTensor:
        return square_forward(t)^

    var numerical_grad = compute_numerical_gradient(forward, x, epsilon=1e-5)

    # Expected: df/dx = 2x = 2*2.0 = 4.0
    var expected = 4.0
    var actual = numerical_grad._get_float64(0)

    var diff = abs(actual - expected)
    assert_true(
        diff < 0.01,
        "Numerical gradient should match analytical",
    )
    print("  OK: Numerical gradient computation correct")


fn test_relative_error_sensitivity() raises:
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


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run gradient checker meta-tests (Part 1 of 2).

    Tests 1-8: Correct/wrong gradient tests and numerical gradient computation.
    """
    print("=" * 70)
    print("GRADIENT CHECKER META-TESTS PART 1 (Validation Tests)")
    print("=" * 70)
    print("Testing gradient checker with f(x) = x^2, df/dx = 2x")
    print("=" * 70)

    print("\n[1] Correct Gradient Tests (Should PASS)")
    print("-" * 70)
    test_gradient_checker_accepts_correct_gradient()
    test_gradient_checker_correct_gradient_multiple_values()
    test_gradient_checker_correct_gradient_multidimensional()

    print("\n[2] Wrong Gradient Tests (Should FAIL and be caught)")
    print("-" * 70)
    test_gradient_checker_rejects_wrong_gradient_linear()
    test_gradient_checker_rejects_wrong_gradient_triple()
    test_gradient_checker_wrong_gradient_multiple_values()

    print("\n[3] Numerical Gradient Tests")
    print("-" * 70)
    test_compute_numerical_gradient_matches_analytical()
    test_relative_error_sensitivity()

    print("\n" + "=" * 70)
    print("ALL GRADIENT CHECKER META-TESTS PART 1 PASSED!")
    print("=" * 70)
    print("Gradient checker is working correctly for layer testing.")
    print("=" * 70)
