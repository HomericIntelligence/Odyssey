"""Gradient checking utilities for validating backward passes.

Implements numerical gradient computation using finite differences to
verify analytical gradients computed by backward passes.

Theory:
    Numerical gradient: f'(x) ≈ [f(x + ε) - f(x - ε)] / (2ε)
    Analytical gradient: Computed by backward pass
    If correct: |numerical - analytical| < tolerance

Benefits:
- Catches gradient bugs early in development
- Validates complex backward pass implementations
- Essential for debugging custom layers
- Provides confidence in optimization

Usage:
    from shared.testing.gradient_checker import check_gradients

    fn forward(x: ExTensor) -> ExTensor:
        return relu(x)

    fn backward(grad_out: ExTensor, x: ExTensor) -> ExTensor:
        return relu_backward(grad_out, x)

    var input = randn([3, 4], DType.float32)
    var passed = check_gradients(forward, backward, input)
    assert_true(passed, "ReLU gradient check failed")

Performance:
    - O(n) function evaluations (n = input size)
    - Use sparingly (expensive for large tensors)
    - Typically run only in test suite, not production

References:
    - CS231n: http://cs231n.github.io/neural-networks-3/#gradcheck
    - Goodfellow et al., Deep Learning, Chapter 4.3
"""

from shared.core import ExTensor, zeros_like


# ============================================================================
# Gradient Checking Constants
# ============================================================================

# Epsilon for float32 gradient checking in matmul-heavy layers (conv2d, linear).
# Using 1e-5 causes ~56% precision loss; 1e-4 gives 3.3% error (above tolerance).
# 3e-4 gives 1.2% error, within the 1.5% tolerance threshold.
# See issue #2704 (Floating-point precision loss in matmul) for full analysis.
alias GRADIENT_CHECK_EPSILON_FLOAT32: Float64 = 3e-4

# Epsilon for non-float32 dtypes (BF16, FP16) in gradient checking.
alias GRADIENT_CHECK_EPSILON_OTHER: Float64 = 1e-3


@fieldwise_init
struct IndexGradientPair(Copyable, Movable):
    """Simple wrapper for (index, gradient) pair returned by sampled gradient checking.
    """

    var index: Int
    var gradient: Float64


fn check_gradients(
    forward_fn: fn (ExTensor) raises escaping -> ExTensor,
    backward_fn: fn (ExTensor, ExTensor) raises escaping -> ExTensor,
    input: ExTensor,
    epsilon: Float64 = 3e-4,  # Changed from 1e-5 - see #2704
    tolerance: Float64 = 1e-2,
) raises -> Bool:
    """Verify gradients using finite differences.

        Compares analytical gradients from backward_fn against numerical
        gradients computed using finite differences. Returns True if all
        gradients match within tolerance.

    Args:
            forward_fn: Forward pass function: input -> output.
            backward_fn: Backward pass function: (grad_output, input) -> grad_input.
            input: Input tensor for testing.
            epsilon: Step size for finite differences (default: 1e-5).
            tolerance: Maximum allowed difference (default: 1e-2).

    Returns:
            True if gradients are correct, False otherwise.

    Raises:
            Error: If forward/backward functions fail.

        Algorithm:
            1. Run forward pass to get output.
            2. Compute analytical gradient using backward pass.
            3. For each input element:
                a. Perturb by +ε, compute f(x+ε).
                b. Perturb by -ε, compute f(x-ε).
                c. Numerical gradient = [f(x+ε) - f(x-ε)] / (2ε).
                d. Compare with analytical gradient.
            4. Return True if max difference < tolerance.

        Example:
            ```mojo
            fn my_forward(x: ExTensor) -> ExTensor:
                return x * x  # f(x) = x²

            fn my_backward(grad_out: ExTensor, x: ExTensor) -> ExTensor:
                return multiply(grad_out, multiply(x, full_like(x, 2.0)))  # f'(x) = 2x

            var x = full([3, 4], 2.0, DType.float32)
            var passed = check_gradients(my_forward, my_backward, x)
            # Should return True (2x is correct derivative of x²)
            ```

        Notes:
            - Expensive: O(n) forward passes where n = input.numel().
            - Use small tensors for testing (e.g., 3x4 instead of 1024x1024).
            - Typical tolerance: 1e-2 for float32 (accounts for accumulated numerical error).
            - Lower epsilon = more accurate but numerically unstable.
            - Higher epsilon = more stable but less accurate.
    """
    # Step 1: Compute analytical gradient
    var output = forward_fn(input)
    var grad_output = zeros_like(output)

    # Set grad_output to ones (∂L/∂output = 1 for all elements)
    for i in range(output.numel()):
        grad_output._set_float64(i, 1.0)

    var analytical_grad = backward_fn(grad_output, input)

    # Step 2: Compute numerical gradient using finite differences
    var numerical_grad = zeros_like(input)
    var input_copy_plus = input.copy()
    var input_copy_minus = input.copy()

    for i in range(input.numel()):
        # Save original value
        var original_val = input._get_float64(i)

        # f(x + ε)
        input_copy_plus._set_float64(i, original_val + epsilon)
        var output_plus = forward_fn(input_copy_plus)

        # f(x - ε)
        input_copy_minus._set_float64(i, original_val - epsilon)
        var output_minus = forward_fn(input_copy_minus)

        # Compute per-element numerical gradient: sum([f(x+ε) - f(x-ε)] / (2ε))
        # This matches the per-element analytical gradient from ones_like(output)
        var numerical_sum = 0.0
        for j in range(output_plus.numel()):
            var diff = output_plus._get_float64(j) - output_minus._get_float64(
                j
            )
            numerical_sum += diff / (2.0 * epsilon)
        numerical_grad._set_float64(i, numerical_sum)

        # Restore original value for next iteration
        input_copy_plus._set_float64(i, original_val)
        input_copy_minus._set_float64(i, original_val)

    # Step 3: Compare analytical vs numerical gradients
    var max_diff = 0.0
    var max_diff_idx = 0

    for i in range(input.numel()):
        var analytical = analytical_grad._get_float64(i)
        var numerical = numerical_grad._get_float64(i)
        var diff = abs(analytical - numerical)

        if diff > max_diff:
            max_diff = diff
            max_diff_idx = i

    # Print diagnostics if gradients don't match
    if max_diff >= tolerance:
        print("Gradient check FAILED:")
        print("  Max difference:", max_diff)
        print("  At index:", max_diff_idx)
        print("  Analytical:", analytical_grad._get_float64(max_diff_idx))
        print("  Numerical:", numerical_grad._get_float64(max_diff_idx))
        print("  Tolerance:", tolerance)
        return False

    return True


fn check_gradients_verbose(
    forward_fn: fn (ExTensor) raises escaping -> ExTensor,
    backward_fn: fn (ExTensor, ExTensor) raises escaping -> ExTensor,
    input: ExTensor,
    epsilon: Float64 = 3e-4,  # Changed from 1e-5 - see #2704
    tolerance: Float64 = 1e-2,
    print_all: Bool = False,
) raises -> Bool:
    """Gradient checking with detailed output.

        Same as check_gradients but prints all differences, not just maximum.
        Useful for debugging specific gradient issues.

    Args:
            forward_fn: Forward pass function.
            backward_fn: Backward pass function.
            input: Input tensor.
            epsilon: Finite difference step size.
            tolerance: Maximum allowed difference.
            print_all: If True, print all gradients (even passing ones).

    Returns:
            True if gradients correct, False otherwise.

        Example:
            ```mojo
            var passed = check_gradients_verbose(
                forward, backward, input,
                print_all=True  # Print all gradient comparisons
            )
            ```

    Raises:
            Error: If operation fails.
    """
    # Run standard gradient check
    var passed = check_gradients(
        forward_fn, backward_fn, input, epsilon, tolerance
    )

    if print_all or not passed:
        print("\n=== Gradient Check Details ===")
        print("Input shape:", String(input.numel()), "elements")
        print("Epsilon:", epsilon)
        print("Tolerance:", tolerance)

        # Recompute for printing
        var output = forward_fn(input)
        var grad_output = zeros_like(output)
        for i in range(output.numel()):
            grad_output._set_float64(i, 1.0)
        var analytical_grad = backward_fn(grad_output, input)

        var numerical_grad = zeros_like(input)
        var input_copy_plus = input.copy()
        var input_copy_minus = input.copy()

        for i in range(input.numel()):
            var original_val = input._get_float64(i)

            input_copy_plus._set_float64(i, original_val + epsilon)
            var output_plus = forward_fn(input_copy_plus)

            input_copy_minus._set_float64(i, original_val - epsilon)
            var output_minus = forward_fn(input_copy_minus)

            var numerical_sum = 0.0
            for j in range(output_plus.numel()):
                var diff = output_plus._get_float64(
                    j
                ) - output_minus._get_float64(j)
                numerical_sum += diff / (2.0 * epsilon)
            numerical_grad._set_float64(i, numerical_sum)

            input_copy_plus._set_float64(i, original_val)
            input_copy_minus._set_float64(i, original_val)

        print("\nGradient Comparisons:")
        print("Index | Analytical | Numerical | Diff | Status")
        print("-" * 60)

        for i in range(min(input.numel(), 20)):  # Print first 20
            var analytical = analytical_grad._get_float64(i)
            var numerical = numerical_grad._get_float64(i)
            var diff = abs(analytical - numerical)
            var status = "PASS" if diff < tolerance else "FAIL"

            if print_all or diff >= tolerance:
                print(
                    i,
                    " | ",
                    analytical,
                    " | ",
                    numerical,
                    " | ",
                    diff,
                    " | ",
                    status,
                )

        if input.numel() > 20:
            print("... (" + String(input.numel() - 20) + " more elements)")

        print("=" * 60)

    return passed


fn relative_error(analytical: Float64, numerical: Float64) -> Float64:
    """Compute relative error between analytical and numerical gradients.

        Uses formula: |a - n| / max(|a|, |n|, 1e-8).
        Handles edge cases where gradients are near zero.

    Args:
            analytical: Analytical gradient value.
            numerical: Numerical gradient value.

    Returns:
            Relative error (typically 0-1, < 0.01 is good).

        Example:
            ```mojo
            var err = relative_error(0.5, 0.501)  # Returns ~0.002 (0.2%)
            ```
    """
    var numerator = abs(analytical - numerical)
    var denominator = max(abs(analytical), max(abs(numerical), 1e-8))
    return numerator / denominator


fn compute_numerical_gradient(
    forward_fn: fn (ExTensor) raises escaping -> ExTensor,
    x: ExTensor,
    epsilon: Float64 = 3e-4,  # Changed from 1e-5 - see #2704
) raises -> ExTensor:
    """Compute numerical gradient using finite differences.

        Uses central difference formula: ∇f(x) ≈ (f(x + ε) - f(x - ε)) / 2ε.

        This is the gold standard for validating analytical gradients. The central
        difference method has O(ε²) error compared to O(ε) for forward/backward
        differences, making it much more accurate.

    Args:
            forward_fn: Function that computes forward pass (takes ExTensor, returns ExTensor).
            x: Input tensor at which to compute gradient.
            epsilon: Small perturbation for finite differences (default: 1e-5).

    Returns:
            ExTensor containing numerical gradient, same shape as x.

    Raises:
            Error: If forward function fails or dtypes are incompatible.

        Notes:
            - For scalar outputs, gradient shape matches input shape.
            - For vector outputs, this computes Jacobian row-by-row (expensive!).
            - Epsilon of 1e-5 is a good compromise between roundoff and truncation error.
            - Use smaller epsilon (1e-7) for Float64, larger (1e-4) for Float16.

        Mathematical basis:
            Taylor expansion: f(x+ε) = f(x) + ε·f'(x) + O(ε²).
            Taylor expansion: f(x-ε) = f(x) - ε·f'(x) + O(ε²).
            Subtracting: f(x+ε) - f(x-ε) = 2ε·f'(x) + O(ε³).
            Therefore: f'(x) ≈ (f(x+ε) - f(x-ε)) / 2ε  [O(ε²) error].

        Example:
            ```mojo
             Validate ReLU gradient
            fn relu_forward(x: ExTensor) raises -> ExTensor:
                return relu(x)

            var x = ExTensor(List[Int](), DType.float32)
            var numerical_grad = compute_numerical_gradient(relu_forward, x)
            var analytical_grad = relu_backward(ones_like(x), x)
            assert_gradients_close(analytical_grad, numerical_grad, rtol=1e-4)
            ```
    """
    # Create gradient tensor (same shape as input)
    var grad = zeros_like(x)

    # Compute gradient for each element using finite differences
    for i in range(x.numel()):
        # Save original value
        var original_val = x._get_float64(i)

        # Compute f(x + ε)
        x._set_float64(i, original_val + epsilon)
        var f_plus = forward_fn(x)

        # Compute f(x - ε)
        x._set_float64(i, original_val - epsilon)
        var f_minus = forward_fn(x)

        # Restore original value
        x._set_float64(i, original_val)

        # Central difference: (f(x+ε) - f(x-ε)) / 2ε
        # Handle both scalar and vector outputs
        var grad_val: Float64
        if f_plus.numel() == 1:
            # Scalar output: gradient is simply the finite difference
            grad_val = (f_plus._get_float64(0) - f_minus._get_float64(0)) / (
                2.0 * epsilon
            )
        else:
            # Vector output: sum of gradients (for loss functions)
            # This assumes we're computing gradient of sum(output) w.r.t input
            grad_val = 0.0
            for j in range(f_plus.numel()):
                grad_val += (
                    f_plus._get_float64(j) - f_minus._get_float64(j)
                ) / (2.0 * epsilon)

        grad._set_float64(i, grad_val)

    return grad^


fn compute_sampled_numerical_gradient(
    forward_fn: fn (ExTensor) raises escaping -> ExTensor,
    x: ExTensor,
    num_samples: Int = 100,
    epsilon: Float64 = 3e-4,  # Changed from 1e-5 - see #2704
    seed: Int = 42,
) raises -> List[IndexGradientPair]:
    """Compute numerical gradient for random sample of input elements.

    Uses simple linear congruential generator (LCG) for reproducible sampling.
    Samples input elements and computes finite difference gradients only for
    those elements, reducing computation by factor of (numel / num_samples).

    This is 99% faster than exhaustive gradient checking while maintaining
    statistical confidence with 100+ samples from large tensors.

    Args:
        forward_fn: Function that computes forward pass (x -> output).
        x: Input tensor.
        num_samples: Number of elements to sample (default: 100).
        epsilon: Perturbation for finite differences (default: 1e-5).
        seed: Random seed for reproducibility (default: 42).

    Returns:
        List of (index, gradient_value) tuples for sampled elements.

    Raises:
        Error: If forward function fails.

    Notes:
        - Always includes first (index 0) and last (index numel-1) elements
        - Remaining samples generated via LCG: x_{n+1} = (a*x_n + c) mod m
        - LCG parameters: a=1103515245, c=12345, m=numel
        - For 4096-element tensors with 100 samples: ~40x speedup vs exhaustive

    Example:
        ```mojo
        fn forward(x: ExTensor) raises escaping -> ExTensor:
            return relu(x)

        var x = ExTensor([100, 100], DType.float32)
        var sampled = compute_sampled_numerical_gradient(
            forward, x, num_samples=100, epsilon=3e-4, seed=42
        )
        # sampled contains ~100 (index, gradient) tuples
        ```
    """
    var numel = x.numel()
    var actual_samples = min(num_samples, numel)

    # Generate sample indices using simple LCG for reproducibility
    var indices = List[Int]()

    # Always include boundary indices
    indices.append(0)
    indices.append(numel - 1)

    # Generate additional random indices
    var rng_state = seed
    var samples_needed = actual_samples - 2
    var count = 0

    while count < samples_needed:
        # LCG: x_{n+1} = (a * x_n + c) mod m
        rng_state = ((rng_state * 1103515245 + 12345) % 2147483648) % numel
        indices.append(rng_state)
        count += 1

    # Compute gradients for sampled indices
    var gradients = List[IndexGradientPair]()

    for idx in indices:
        var original_val = x._get_float64(idx)

        # f(x + ε)
        x._set_float64(idx, original_val + epsilon)
        var f_plus = forward_fn(x)
        var f_plus_sum: Float64 = 0.0
        for j in range(f_plus.numel()):
            f_plus_sum += f_plus._get_float64(j)

        # f(x - ε)
        x._set_float64(idx, original_val - epsilon)
        var f_minus = forward_fn(x)
        var f_minus_sum: Float64 = 0.0
        for j in range(f_minus.numel()):
            f_minus_sum += f_minus._get_float64(j)

        # Restore original
        x._set_float64(idx, original_val)

        # Compute gradient: (f(x + ε) - f(x - ε)) / (2ε)
        var grad = (f_plus_sum - f_minus_sum) / (2.0 * epsilon)
        gradients.append(IndexGradientPair(idx, grad))

    return gradients^


fn assert_sampled_gradients_close(
    analytical_grad: ExTensor,
    sampled_numerical: List[IndexGradientPair],
    rtol: Float64 = 1e-2,
    atol: Float64 = 1e-2,  # 1% absolute tolerance for small gradients
    message: String = "Sampled gradients mismatch",
) raises:
    """Compare analytical gradient with sampled numerical gradients.

    Validates that analytical gradients match numerical gradients at randomly
    sampled locations. If any sampled gradient exceeds relative tolerance,
    raises error with details of worst mismatch.

    This complements compute_sampled_numerical_gradient() to provide hybrid
    validation: fast analytical gradients verified by statistical sampling.

    Args:
        analytical_grad: Full analytical gradient tensor.
        sampled_numerical: List of IndexGradientPair samples from sampling.
        rtol: Relative tolerance (default: 1e-2 for float32).
        message: Error message prefix.

    Raises:
        Error: If any sampled gradient exceeds tolerance, includes worst case details.

    Notes:
        - Relative error: |analytical - numerical| / max(|analytical|, |numerical|, 1e-8)
        - Handles gradients near zero with epsilon=1e-8
        - Reports worst mismatch (highest relative error) for debugging
        - Typical tolerance: 1e-2 for float32, 1e-1 for float16

    Example:
        ```mojo
        var analytical = relu_backward(grad_output, x)
        var sampled = compute_sampled_numerical_gradient(
            relu_forward, x, num_samples=100
        )
        assert_sampled_gradients_close(analytical, sampled, rtol=1e-2)
        ```
    """
    var max_error: Float64 = 0.0
    var worst_idx: Int = -1
    var any_failed = False

    for sample in sampled_numerical:
        var idx = sample.index
        var numerical = sample.gradient
        var analytical = analytical_grad._get_float64(idx)

        var abs_diff = analytical - numerical
        if abs_diff < 0.0:
            abs_diff = -abs_diff

        # Combined absolute + relative tolerance check
        # Passes if: |analytical - numerical| <= atol + rtol * max(|analytical|, |numerical|)
        var abs_analytical = analytical if analytical >= 0.0 else -analytical
        var abs_numerical = numerical if numerical >= 0.0 else -numerical
        var max_magnitude = (
            abs_analytical if abs_analytical > abs_numerical else abs_numerical
        )
        var tolerance = atol + rtol * max_magnitude
        var rel_error = abs_diff / (max_magnitude + 1e-8)  # For reporting only

        # Check if this sample fails the combined tolerance
        if abs_diff > tolerance:
            any_failed = True
            if rel_error > max_error:
                max_error = rel_error
                worst_idx = idx

    # Fail if any sample exceeded the combined tolerance
    if any_failed:
        var analytical_val = analytical_grad._get_float64(worst_idx)
        var msg = (
            message
            + ": max relative error "
            + String(max_error)
            + " at index "
            + String(worst_idx)
            + " exceeds tolerance "
            + String(rtol)
        )
        msg += " (analytical=" + String(analytical_val) + ")"
        raise Error(msg)


fn assert_gradients_close(
    analytical: ExTensor,
    numerical: ExTensor,
    rtol: Float64 = 1e-3,
    atol: Float64 = 1e-6,
    message: String = "Gradients do not match",
) raises:
    """Assert analytical and numerical gradients are close.

        Uses relative and absolute tolerance to handle both small and large gradients:
            |analytical - numerical| <= atol + rtol * |numerical|.

    Args:
            analytical: Gradient computed by backward pass.
            numerical: Gradient computed by finite differences.
            rtol: Relative tolerance (default: 1e-4, suitable for float32).
            atol: Absolute tolerance (default: 1e-7).
            message: Error message prefix.

    Raises:
            Error: If gradients differ beyond tolerance.

        Tolerance Guidelines:
            Float16: rtol=1e-2, atol=1e-4.
            Float32: rtol=1e-4, atol=1e-7.
            Float64: rtol=1e-7, atol=1e-10.

        Example:
            ```mojo
            var analytical = relu_backward(grad_output, x)
            var numerical = compute_numerical_gradient(relu, x)
            assert_gradients_close(analytical, numerical)  # Uses default tolerances
            ```
    """
    # Check shapes match
    if analytical.numel() != numerical.numel():
        raise Error(message + ": shape mismatch")

    # Check dtypes match
    if analytical._dtype != numerical._dtype:
        raise Error(message + ": dtype mismatch")

    # Compare element-wise
    var max_diff: Float64 = 0.0
    var max_rel_diff: Float64 = 0.0
    var worst_idx: Int = -1
    var worst_tolerance: Float64 = 0.0
    var tolerance_exceeded: Bool = False

    for i in range(analytical.numel()):
        var a = analytical._get_float64(i)
        var n = numerical._get_float64(i)
        var abs_diff: Float64
        if a - n < 0:
            abs_diff = -(a - n)
        else:
            abs_diff = a - n

        # Use max(|a|, |n|) for relative tolerance to handle near-zero gradients
        var abs_a = a if a >= 0.0 else -a
        var abs_n = n if n >= 0.0 else -n
        var max_abs = abs_a if abs_a > abs_n else abs_n
        var tolerance = atol + rtol * max_abs

        if abs_diff > max_diff:
            max_diff = abs_diff
            worst_idx = i
            worst_tolerance = tolerance

        # Compute relative difference for reporting (using max_abs from above)
        if max_abs > 1e-10:
            var rel_diff = abs_diff / max_abs
            if rel_diff > max_rel_diff:
                max_rel_diff = rel_diff

        if abs_diff > tolerance:
            tolerance_exceeded = True

    # Report error after finding worst element
    if tolerance_exceeded:
        var a = analytical._get_float64(worst_idx)
        var n = numerical._get_float64(worst_idx)
        var msg = (
            message + ": worst gradient mismatch at index " + String(worst_idx)
        )
        msg += "\n  Analytical: " + String(a)
        msg += "\n  Numerical:  " + String(n)
        msg += "\n  Difference: " + String(max_diff)
        msg += "\n  Tolerance:  " + String(worst_tolerance)
        msg += "\n  Total elements: " + String(analytical.numel())
        raise Error(msg)


fn _deep_copy(tensor: ExTensor) raises -> ExTensor:
    """Create a deep copy of a tensor with independent data buffer.

        ExTensor's __copyinit__ creates shallow copies (shared data via reference counting).
        This function creates a true deep copy with separate memory allocation.

    Args:
            tensor: Tensor to deep copy.

    Returns:
            New tensor with copied data (independent memory allocation).
    """
    # Create new tensor with same shape and dtype
    var result = ExTensor(tensor.shape(), tensor._dtype)

    # Copy all data elements
    for i in range(tensor.numel()):
        result._set_float64(i, tensor._get_float64(i))

    return result^


fn check_gradient(
    forward_fn: fn (ExTensor) raises escaping -> ExTensor,
    backward_fn: fn (ExTensor, ExTensor) raises escaping -> ExTensor,
    x: ExTensor,
    grad_output: ExTensor,
    epsilon: Float64 = 0.0,  # Auto-select based on dtype if 0.0
    rtol: Float64 = 1e-3,
    atol: Float64 = 1e-6,
) raises:
    """Comprehensive gradient check helper.

        Combines numerical gradient computation and comparison in one call.
        This is the recommended way to validate backward passes in tests.

    Args:
            forward_fn: Forward pass function.
            backward_fn: Backward pass function (takes grad_output and x).
            x: Input tensor.
            grad_output: Gradient from upstream (typically ones_like(output)).
            epsilon: Perturbation size for finite differences (0.0 = auto-select).
            rtol: Relative tolerance.
            atol: Absolute tolerance.

    Raises:
            Error: If gradients don't match within tolerance.

        Example:
            ```mojo
            fn test_relu_gradient() raises:
                var x = ExTensor(List[Int](), DType.float32)
                # ... initialize x with test values ...

                fn forward(inp: ExTensor) raises escaping -> ExTensor:
                    return relu(inp)

                fn backward_wrapper(grad: ExTensor, x: ExTensor) raises escaping -> ExTensor:
                    return relu_backward(grad, x)

                var grad_out = ones_like(relu(x))
                check_gradient(forward, backward_wrapper, x, grad_out)
            ```
    """
    # Auto-select epsilon and atol based on dtype if not specified
    var eps = epsilon
    var auto_atol = atol
    if eps == 0.0:
        # Use sqrt(machine_epsilon) for numerical stability
        # Float32: machine eps ~1.2e-7, sqrt ~3.5e-4, use 1e-4
        # Float64: machine eps ~2.2e-16, sqrt ~1.5e-8, use 1e-7
        if x._dtype == DType.float32:
            eps = 1e-4
            # For near-zero gradients, numerical error is O(eps), so set atol >= eps
            if atol < 1e-4:  # If atol is too small for eps=1e-4
                auto_atol = 1e-4
        elif x._dtype == DType.float64:
            eps = 1e-7
            if atol < 1e-7:  # If atol is too small for eps=1e-7
                auto_atol = 1e-7
        else:
            eps = 1e-5  # Default for other types

    # Compute analytical gradient
    var analytical = backward_fn(grad_output, x)

    # Compute numerical gradient for scalar loss
    # For non-scalar outputs, we compute gradient of: loss = sum(forward(x) * grad_output)
    # We'll approximate this by perturbing x and seeing how the scalar loss changes
    var grad = zeros_like(x)

    for i in range(x.numel()):
        # Create deep copies to avoid corrupting original x
        # (ExTensor.__copyinit__ creates shallow copies with shared data)
        var x_plus = _deep_copy(x)
        var old_val = x._get_float64(i)
        x_plus._set_float64(i, old_val + eps)
        var out_plus = forward_fn(x_plus)
        var loss_plus: Float64 = 0.0
        for j in range(out_plus.numel()):
            loss_plus += out_plus._get_float64(j) * grad_output._get_float64(j)

        # Backward perturbation
        var x_minus = _deep_copy(x)
        x_minus._set_float64(i, old_val - eps)
        var out_minus = forward_fn(x_minus)
        var loss_minus: Float64 = 0.0
        for j in range(out_minus.numel()):
            loss_minus += out_minus._get_float64(j) * grad_output._get_float64(
                j
            )

        # Central difference
        var numerical_grad = (loss_plus - loss_minus) / (2.0 * eps)
        grad._set_float64(i, numerical_grad)

    # Compare
    assert_gradients_close(
        analytical,
        grad,
        rtol,
        auto_atol,
        "Gradient check failed for " + String(x._dtype),
    )
