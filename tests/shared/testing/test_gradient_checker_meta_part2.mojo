# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_gradient_checker_meta.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md
"""Meta-tests for gradient checker validation (Part 2 of 2).

These tests verify that the gradient checker itself works correctly by using
a simple function with known analytical gradients.

Theory:
    For f(x) = x², the analytical gradient is df/dx = 2x
    This allows us to test:
    - Edge cases (zero, negative, moderate inputs)
    - Epsilon parameter sensitivity
    - Memory safety (no mutation of input tensor)

Test Cases (Part 2):
    1. Gradient checker with zero input
    2. Gradient checker with negative input
    3. Gradient checker with moderate input values
    4. Gradient checker with smaller epsilon
    5. Gradient checker with larger epsilon
    6. Memory safety: check_gradients does not mutate input

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


# ============================================================================
# Meta-Tests: Edge Cases
# ============================================================================


fn test_gradient_checker_zero_input() raises:
    """Meta-test: Gradient checker should handle zero input correctly.

    For x = 0:
    - f(0) = 0
    - df/dx = 0

    This is an important edge case since the gradient is zero.
    """
    print("Meta-test: Gradient checker handles zero input...")

    var x = full([1], 0.0, DType.float32)

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

    assert_true(passed, "Gradient checker should handle x=0 correctly")
    print("  OK: Gradient checker handles zero input")


fn test_gradient_checker_negative_input() raises:
    """Meta-test: Gradient checker handles negative inputs.

    For x = -2.0:
    - f(-2) = 4
    - df/dx = -4

    Negative gradients should be handled correctly.
    """
    print("Meta-test: Gradient checker handles negative input...")

    var x = full([1], -2.0, DType.float32)

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

    assert_true(passed, "Gradient checker should handle negative inputs")
    print("  OK: Gradient checker handles negative input")


fn test_gradient_checker_large_input() raises:
    """Meta-test: Gradient checker works for moderate input values.

    For x = 5.0:
    - f(5) = 25
    - df/dx = 10

    Should work with moderate absolute values.
    Note: Uses larger tolerance due to accumulated floating-point error.
    """
    print("Meta-test: Gradient checker handles moderate input...")

    var x = full([1], 5.0, DType.float32)

    fn forward(t: ExTensor) raises escaping -> ExTensor:
        return square_forward(t)^

    fn backward(grad: ExTensor, inp: ExTensor) raises escaping -> ExTensor:
        return square_backward_correct(grad, inp)^

    # Moderate inputs still need larger tolerance for float32 precision
    var passed = check_gradients(
        forward,
        backward,
        x,
        epsilon=1e-5,
        tolerance=0.05,
    )

    assert_true(passed, "Gradient checker should handle moderate inputs")
    print("  OK: Gradient checker handles moderate input values")


# ============================================================================
# Meta-Tests: Gradient Checker Epsilon Parameter
# ============================================================================


fn test_gradient_checker_small_epsilon() raises:
    """Meta-test: Gradient checker works with reasonable small epsilon.

    Smaller epsilon (1e-4) provides better numerical accuracy.
    This tests gradient checker stability.
    """
    print("Meta-test: Gradient checker with smaller epsilon...")

    var x = full([1], 1.0, DType.float32)

    fn forward(t: ExTensor) raises escaping -> ExTensor:
        return square_forward(t)^

    fn backward(grad: ExTensor, inp: ExTensor) raises escaping -> ExTensor:
        return square_backward_correct(grad, inp)^

    var passed = check_gradients(
        forward,
        backward,
        x,
        epsilon=1e-4,
        tolerance=1e-2,
    )

    assert_true(passed, "Gradient checker should work with smaller epsilon")
    print("  OK: Gradient checker stable with smaller epsilon")


fn test_gradient_checker_large_epsilon() raises:
    """Meta-test: Gradient checker works with larger epsilon.

    Larger epsilon (1e-3) should still pass with correct gradient.
    This tests that checker is not overly sensitive to epsilon choice.
    """
    print("Meta-test: Gradient checker with large epsilon...")

    var x = full([1], 1.0, DType.float32)

    fn forward(t: ExTensor) raises escaping -> ExTensor:
        return square_forward(t)^

    fn backward(grad: ExTensor, inp: ExTensor) raises escaping -> ExTensor:
        return square_backward_correct(grad, inp)^

    var passed = check_gradients(
        forward,
        backward,
        x,
        epsilon=1e-3,
        tolerance=1e-2,
    )

    assert_true(passed, "Gradient checker should work with large epsilon")
    print("  OK: Gradient checker stable with large epsilon")


# ============================================================================
# Memory Safety Tests
# ============================================================================


fn test_check_gradients_does_not_mutate_input() raises:
    """Regression test: check_gradients must not mutate the original input tensor.

    Verifies that the deep-copy fix for the shallow-copy memory hazard is in
    place. Before the fix, `input.copy()` (a `__copyinit__` shallow copy) shared
    the `_data` buffer with the original tensor, so `_set_float64` calls inside
    check_gradients corrupted the caller's tensor. After the fix, `_deep_copy`
    allocates an independent buffer and the original is never modified.
    """
    print("Memory safety: check_gradients does not mutate input...")

    var x = full([2, 2], 1.0, DType.float32)

    # Snapshot all values before the call
    var before = List[Float64]()
    for i in range(x.numel()):
        before.append(x._get_float64(i))

    fn forward(t: ExTensor) raises escaping -> ExTensor:
        return square_forward(t)^

    fn backward(grad: ExTensor, inp: ExTensor) raises escaping -> ExTensor:
        return square_backward_correct(grad, inp)^

    _ = check_gradients(forward, backward, x, epsilon=1e-5, tolerance=1e-2)

    # Assert every element is unchanged
    for i in range(x.numel()):
        var after_val = x._get_float64(i)
        assert_equal(
            after_val,
            before[i],
            "check_gradients mutated input at index " + String(i),
        )

    print("  OK: check_gradients does not mutate the original input tensor")


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run gradient checker meta-tests (Part 2 of 2).

    Tests 9-14: Edge cases, epsilon sensitivity, and memory safety.
    """
    print("=" * 70)
    print("GRADIENT CHECKER META-TESTS PART 2 (Validation Tests)")
    print("=" * 70)
    print("Testing gradient checker with f(x) = x^2, df/dx = 2x")
    print("=" * 70)

    print("\n[4] Edge Case Tests")
    print("-" * 70)
    test_gradient_checker_zero_input()
    test_gradient_checker_negative_input()
    test_gradient_checker_large_input()

    print("\n[5] Epsilon Parameter Tests")
    print("-" * 70)
    test_gradient_checker_small_epsilon()
    test_gradient_checker_large_epsilon()

    print("\n[6] Memory Safety Tests")
    print("-" * 70)
    test_check_gradients_does_not_mutate_input()

    print("\n" + "=" * 70)
    print("ALL GRADIENT CHECKER META-TESTS PART 2 PASSED!")
    print("=" * 70)
    print("Gradient checker is working correctly for layer testing.")
    print("=" * 70)
