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

    def forward(x: AnyTensor) -> AnyTensor:
        return relu(x)

    def backward(grad_out: AnyTensor, x: AnyTensor) -> AnyTensor:
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

Fix for intermittent JIT crashes (issue #5104):
    All tight loops now use data_ptr[dtype]() to obtain typed pointers once
    per loop scope, replacing per-element _get_float64/_set_float64 calls.
    The per-element bitcast in _get/_set_float64 could trigger ASAP destruction
    of the tensor while bitcast-derived pointers were still live
    (modular/modular#6187). The data_ptr approach keeps the tensor alive for
    the entire pointer scope.
"""

from shared.tensor.any_tensor import AnyTensor, zeros_like


# ============================================================================
# Forward Function Trait
# ============================================================================

trait NumericalForward(Copyable, Movable):
    """Trait for forward functions used in numerical gradient checking.

    Implement this trait on a struct to pass capturing closures to
    compute_numerical_gradient and compute_sampled_numerical_gradient.
    This is required because Mojo 0.26.3 does not allow non-escaping
    capturing closures to be passed as runtime function arguments.
    """

    def __call__(self, x: AnyTensor) raises -> AnyTensor: ...


trait NumericalBackward(Copyable, Movable):
    """Trait for backward functions used in gradient checking.

    Implement this trait on a struct to pass capturing closures to
    check_gradient and check_gradients. Required because Mojo 0.26.3
    does not allow non-escaping capturing closures to be passed as
    runtime function arguments.

    The __call__ signature matches the backward_fn parameter of
    check_gradient and check_gradients:
        backward_fn(grad_out: AnyTensor, x: AnyTensor) -> AnyTensor
    """

    def __call__(self, grad_out: AnyTensor, x: AnyTensor) raises -> AnyTensor: ...


# ============================================================================
# Gradient Checking Constants
# ============================================================================

# Epsilon for float32 gradient checking in matmul-heavy layers (conv2d, linear).
# Using 1e-5 causes ~56% precision loss; 1e-4 gives 3.3% error (above tolerance).
# 3e-4 gives 1.2% error, within the 1.5% tolerance threshold.
# See issue #2704 (Floating-point precision loss in matmul) for full analysis.
comptime GRADIENT_CHECK_EPSILON_FLOAT32: Float64 = 3e-4

# Epsilon for non-float32 dtypes (BF16, FP16) in gradient checking.
comptime GRADIENT_CHECK_EPSILON_OTHER: Float64 = 1e-3


@fieldwise_init
struct IndexGradientPair(Copyable, Movable):
    """Simple wrapper for (index, gradient) pair returned by sampled gradient checking.
    """

    var index: Int
    var gradient: Float64


# ============================================================================
# Typed pointer helpers — eliminate per-element bitcast (fixes #5104)
#
# These helpers use data_ptr[dtype]() to get a typed pointer ONCE, then
# index that pointer throughout the loop. This keeps the source tensor
# alive for the entire scope, preventing ASAP destruction from freeing
# tensor memory while a bitcast-derived pointer is still in use
# (modular/modular#6187).
# ============================================================================


def _get_val_as_f64[dtype: DType](tensor: AnyTensor, index: Int) -> Float64:
    """Read tensor element at flat index as Float64 using typed pointer."""
    var ptr = tensor.data_ptr[dtype]()
    return Float64(ptr[index])


def _set_val_from_f64[dtype: DType](tensor: AnyTensor, index: Int, value: Float64):
    """Write Float64 value to tensor element at flat index using typed pointer."""
    var ptr = tensor.data_ptr[dtype]()
    ptr[index] = Scalar[dtype](value)


def _fill_ones[dtype: DType](tensor: AnyTensor):
    """Fill tensor with 1.0 using typed pointer."""
    var ptr = tensor.data_ptr[dtype]()
    for i in range(tensor.numel()):
        ptr[i] = Scalar[dtype](1.0)


def _is_uniform_tensor(tensor: AnyTensor) -> Bool:
    """Check if all elements in a tensor have the same value (uniform tensor).

    Args:
        tensor: Tensor to check.

    Returns:
        True if all elements are identical, False otherwise.

    Note:
        Uniform tensors (e.g., ones_like) cause degenerate gradient checking
        for normalization layers (batch norm, layer norm) where variance-based
        computations produce incorrect or undefined gradients. This is a gotcha
        documented in #3282.
    """
    if tensor.numel() == 0:
        return True
    if tensor.numel() == 1:
        return True

    var first_val = tensor._get_float64(0)
    for i in range(1, tensor.numel()):
        var val = tensor._get_float64(i)
        # Use approximate equality for floating-point comparison
        if abs(val - first_val) > 1e-9:
            return False
    return True


# ============================================================================
# Dtype-dispatched perturbation loop for check_gradients
# ============================================================================


def _check_gradients_perturb[
    dtype: DType,
](
    forward_fn: def (AnyTensor) raises -> AnyTensor,
    input: AnyTensor,
    input_copy_plus: AnyTensor,
    input_copy_minus: AnyTensor,
    numerical_grad: AnyTensor,
    epsilon: Float64,
) raises:
    """Run the finite-difference perturbation loop using typed pointers.

    Gets data_ptr[dtype]() once per tensor and indexes throughout the loop,
    keeping all tensors alive and avoiding per-element bitcast (fixes #5104).
    """
    var in_ptr = input.data_ptr[dtype]()
    var plus_ptr = input_copy_plus.data_ptr[dtype]()
    var minus_ptr = input_copy_minus.data_ptr[dtype]()
    var grad_ptr = numerical_grad.data_ptr[dtype]()

    for i in range(input.numel()):
        # Save original value
        var original_val = Float64(in_ptr[i])

        # f(x + ε)
        plus_ptr[i] = Scalar[dtype](original_val + epsilon)
        var output_plus = forward_fn(input_copy_plus)

        # f(x - ε)
        minus_ptr[i] = Scalar[dtype](original_val - epsilon)
        var output_minus = forward_fn(input_copy_minus)

        # Compute per-element numerical gradient: sum([f(x+ε) - f(x-ε)] / (2ε))
        # Use data_ptr[dtype]() once per tensor to keep output_plus/output_minus
        # alive for the entire loop scope. Per-element _get_float64 performs a
        # bitcast on each call and can trigger ASAP destruction of the temporary
        # tensor before all elements are read (modular/modular#6187).
        var out_plus_ptr = output_plus.data_ptr[dtype]()
        var out_minus_ptr = output_minus.data_ptr[dtype]()
        var numerical_sum: Float64 = 0.0
        for j in range(output_plus.numel()):
            var diff = Float64(out_plus_ptr[j]) - Float64(out_minus_ptr[j])
            numerical_sum += diff / (2.0 * epsilon)
        grad_ptr[i] = Scalar[dtype](numerical_sum)

        # Restore original value for next iteration
        plus_ptr[i] = Scalar[dtype](original_val)
        minus_ptr[i] = Scalar[dtype](original_val)


def _check_gradients_perturb[
    dtype: DType,
    F: NumericalForward,
](
    forward_fn: F,
    input: AnyTensor,
    input_copy_plus: AnyTensor,
    input_copy_minus: AnyTensor,
    numerical_grad: AnyTensor,
    epsilon: Float64,
) raises:
    """Run the finite-difference perturbation loop using typed pointers — trait-parameterized overload.

    Gets data_ptr[dtype]() once per tensor and indexes throughout the loop,
    keeping all tensors alive and avoiding per-element bitcast (fixes #5104).

    Use this overload when forward_fn is a struct implementing NumericalForward.
    """
    var in_ptr = input.data_ptr[dtype]()
    var plus_ptr = input_copy_plus.data_ptr[dtype]()
    var minus_ptr = input_copy_minus.data_ptr[dtype]()
    var grad_ptr = numerical_grad.data_ptr[dtype]()

    for i in range(input.numel()):
        # Save original value
        var original_val = Float64(in_ptr[i])

        # f(x + ε)
        plus_ptr[i] = Scalar[dtype](original_val + epsilon)
        # Capture actual quantized value (for FP16/BF16 the step may be asymmetric).
        var actual_plus = Float64(plus_ptr[i])
        var output_plus = forward_fn(input_copy_plus)

        # f(x - ε)
        minus_ptr[i] = Scalar[dtype](original_val - epsilon)
        var actual_minus = Float64(minus_ptr[i])
        var output_minus = forward_fn(input_copy_minus)

        # Compute per-element numerical gradient: sum([f(x+ε) - f(x-ε)] / (2ε))
        # Use data_ptr[dtype]() once per tensor to keep output_plus/output_minus
        # alive for the entire loop scope. Per-element _get_float64 performs a
        # bitcast on each call and can trigger ASAP destruction of the temporary
        # tensor before all elements are read (modular/modular#6187).
        # Use actual quantized step size; for FP32/FP64 this equals epsilon, but
        # for FP16/BF16 the quantized step may differ from the requested epsilon,
        # causing systematic errors in the gradient estimate.
        var actual_eps = (actual_plus - actual_minus) / 2.0
        var denom = actual_eps if actual_eps > 0.0 else epsilon
        var out_plus_ptr = output_plus.data_ptr[dtype]()
        var out_minus_ptr = output_minus.data_ptr[dtype]()
        var numerical_sum: Float64 = 0.0
        for j in range(output_plus.numel()):
            var diff = Float64(out_plus_ptr[j]) - Float64(out_minus_ptr[j])
            numerical_sum += diff / (2.0 * denom)
        grad_ptr[i] = Scalar[dtype](numerical_sum)

        # Restore original value for next iteration
        plus_ptr[i] = Scalar[dtype](original_val)
        minus_ptr[i] = Scalar[dtype](original_val)


def _dispatch_check_gradients_perturb(
    forward_fn: def(AnyTensor) raises -> AnyTensor,
    input: AnyTensor,
    input_copy_plus: AnyTensor,
    input_copy_minus: AnyTensor,
    numerical_grad: AnyTensor,
    epsilon: Float64,
) raises:
    """Dispatch perturbation loop to dtype-specific implementation."""
    if input._dtype == DType.float16:
        _check_gradients_perturb[DType.float16](
            forward_fn, input, input_copy_plus, input_copy_minus,
            numerical_grad, epsilon,
        )
    elif input._dtype == DType.bfloat16:
        _check_gradients_perturb[DType.bfloat16](
            forward_fn, input, input_copy_plus, input_copy_minus,
            numerical_grad, epsilon,
        )
    elif input._dtype == DType.float32:
        _check_gradients_perturb[DType.float32](
            forward_fn, input, input_copy_plus, input_copy_minus,
            numerical_grad, epsilon,
        )
    elif input._dtype == DType.float64:
        _check_gradients_perturb[DType.float64](
            forward_fn, input, input_copy_plus, input_copy_minus,
            numerical_grad, epsilon,
        )
    else:
        raise Error(
            "Unsupported dtype for gradient checking: " + String(input._dtype)
        )


def _dispatch_check_gradients_perturb_trait[F: NumericalForward](
    forward_fn: F,
    input: AnyTensor,
    input_copy_plus: AnyTensor,
    input_copy_minus: AnyTensor,
    numerical_grad: AnyTensor,
    epsilon: Float64,
) raises:
    """Dispatch perturbation loop to dtype-specific implementation for trait-parameterized forward."""
    if input._dtype == DType.float16:
        _check_gradients_perturb[DType.float16](
            forward_fn, input, input_copy_plus, input_copy_minus,
            numerical_grad, epsilon,
        )
    elif input._dtype == DType.bfloat16:
        _check_gradients_perturb[DType.bfloat16](
            forward_fn, input, input_copy_plus, input_copy_minus,
            numerical_grad, epsilon,
        )
    elif input._dtype == DType.float32:
        _check_gradients_perturb[DType.float32](
            forward_fn, input, input_copy_plus, input_copy_minus,
            numerical_grad, epsilon,
        )
    elif input._dtype == DType.float64:
        _check_gradients_perturb[DType.float64](
            forward_fn, input, input_copy_plus, input_copy_minus,
            numerical_grad, epsilon,
        )
    else:
        raise Error(
            "Unsupported dtype for gradient checking: " + String(input._dtype)
        )


def check_gradients(
    forward_fn: def(AnyTensor) raises -> AnyTensor,
    backward_fn: def(AnyTensor, AnyTensor) raises -> AnyTensor,
    input: AnyTensor,
    epsilon: Float64 = 3e-4,  # Changed from 1e-5 - see #2704
    tolerance: Float64 = 1e-2,
) raises -> Bool:
    """Verify gradients using finite differences.

        Compares analytical gradients from backward_fn against numerical
        gradients computed using finite differences. Returns True if all
        gradients match within tolerance.

    Args:
            forward_fn: Forward function to differentiate.
            backward_fn: Backward function computing analytical gradients.
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
            def my_forward(x: AnyTensor) -> AnyTensor:
                return x * x  # f(x) = x²

            def my_backward(grad_out: AnyTensor, x: AnyTensor) -> AnyTensor:
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
    # Use typed pointer fill to avoid per-element bitcast
    if output._dtype == DType.float16:
        _fill_ones[DType.float16](grad_output)
    elif output._dtype == DType.bfloat16:
        _fill_ones[DType.bfloat16](grad_output)
    elif output._dtype == DType.float32:
        _fill_ones[DType.float32](grad_output)
    elif output._dtype == DType.float64:
        _fill_ones[DType.float64](grad_output)
    else:
        for i in range(output.numel()):
            grad_output._set_float64(i, 1.0)

    # GOTCHA: Warn if grad_output is uniform (all ones)
    # For normalization layers (batch norm, layer norm), uniform grad_output creates
    # a degenerate case where variance becomes zero and gradients are undefined/clamped.
    # See #3282 for discussion. This warning makes the gotcha self-enforcing rather than
    # relying solely on documentation.
    if _is_uniform_tensor(grad_output):
        print("WARNING: check_gradients() got uniform grad_output (likely ones_like)")
        print("This is pathological for normalization layers where variance=0 causes undefined gradients.")
        print("For batch norm, layer norm, etc., use non-uniform grad_output (e.g., randn).")
        print("See issue #3282 for more details.")

    var analytical_grad = backward_fn(grad_output, input)

    # Step 2: Compute numerical gradient using finite differences
    # Uses dtype-dispatched perturbation loop with typed pointers (fixes #5104)
    var numerical_grad = zeros_like(input)
    var input_copy_plus = input.clone()
    var input_copy_minus = input.clone()

    _dispatch_check_gradients_perturb(
        forward_fn, input, input_copy_plus, input_copy_minus,
        numerical_grad, epsilon,
    )

    # Step 3: Compare analytical vs numerical gradients
    var max_diff: Float64 = 0.0
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


def check_gradients[F: NumericalForward, B: NumericalBackward](
    forward_fn: F,
    backward_fn: B,
    input: AnyTensor,
    epsilon: Float64 = 3e-4,
    tolerance: Float64 = 1e-2,
) raises -> Bool:
    """Check gradients using finite differences — trait-parameterized overload.

    Use this overload when forward_fn and backward_fn are structs implementing
    NumericalForward and NumericalBackward traits. This allows wrapping
    capturing closures which cannot be passed directly in Mojo 0.26.3+.

    Args:
        forward_fn: A struct implementing NumericalForward.
        backward_fn: A struct implementing NumericalBackward.
        input: Input tensor to differentiate at.
        epsilon: Finite difference step size (default: 3e-4).
        tolerance: Maximum allowed difference (default: 1e-2).

    Returns:
        True if gradients are correct, False otherwise.

    Raises:
        Error: If forward/backward functions fail.

    Example:
        ```mojo
        @fieldwise_init
        struct MyForward(NumericalForward):
            var param: AnyTensor

            def __call__(self, x: AnyTensor) raises -> AnyTensor:
                return x * self.param

        @fieldwise_init
        struct MyBackward(NumericalBackward):
            var param: AnyTensor

            def __call__(self, grad_out: AnyTensor, x: AnyTensor) raises -> AnyTensor:
                return grad_out * self.param

        var param = ones([3, 4], DType.float32)
        var forward = MyForward(param^)
        var backward = MyBackward(param^)
        var x = full([3, 4], 2.0, DType.float32)
        var passed = check_gradients(forward, backward, x)
        ```
    """
    # Step 1: Compute analytical gradient
    var output = forward_fn(input)
    var grad_output = zeros_like(output)

    # Set grad_output to ones (∂L/∂output = 1 for all elements)
    # Use typed pointer fill to avoid per-element bitcast
    if output._dtype == DType.float16:
        _fill_ones[DType.float16](grad_output)
    elif output._dtype == DType.bfloat16:
        _fill_ones[DType.bfloat16](grad_output)
    elif output._dtype == DType.float32:
        _fill_ones[DType.float32](grad_output)
    elif output._dtype == DType.float64:
        _fill_ones[DType.float64](grad_output)
    else:
        for i in range(output.numel()):
            grad_output._set_float64(i, 1.0)

    # GOTCHA: Warn if grad_output is uniform (all ones)
    # For normalization layers (batch norm, layer norm), uniform grad_output creates
    # a degenerate case where variance becomes zero and gradients are undefined/clamped.
    # See #3282 for discussion. This warning makes the gotcha self-enforcing rather than
    # relying solely on documentation.
    if _is_uniform_tensor(grad_output):
        print("WARNING: check_gradients() got uniform grad_output (likely ones_like)")
        print("This is pathological for normalization layers where variance=0 causes undefined gradients.")
        print("For batch norm, layer norm, etc., use non-uniform grad_output (e.g., randn).")
        print("See issue #3282 for more details.")

    var analytical_grad = backward_fn(grad_output, input)

    # Step 2: Compute numerical gradient using finite differences
    # Uses dtype-dispatched perturbation loop with typed pointers (fixes #5104)
    var numerical_grad = zeros_like(input)
    var input_copy_plus = input.clone()
    var input_copy_minus = input.clone()

    _dispatch_check_gradients_perturb_trait(
        forward_fn, input, input_copy_plus, input_copy_minus,
        numerical_grad, epsilon,
    )

    # Step 3: Compare analytical vs numerical gradients
    var max_diff: Float64 = 0.0
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


def check_gradients_verbose(
    forward_fn: def(AnyTensor) raises -> AnyTensor,
    backward_fn: def(AnyTensor, AnyTensor) raises -> AnyTensor,
    input: AnyTensor,
    epsilon: Float64 = 3e-4,  # Changed from 1e-5 - see #2704
    tolerance: Float64 = 1e-2,
    print_all: Bool = False,
) raises -> Bool:
    """Gradient checking with detailed output.

        Same as check_gradients but prints all differences, not just maximum.
        Useful for debugging specific gradient issues.

    Args:
            forward_fn: Forward function to differentiate.
            backward_fn: Backward function computing analytical gradients.
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
        if output._dtype == DType.float16:
            _fill_ones[DType.float16](grad_output)
        elif output._dtype == DType.bfloat16:
            _fill_ones[DType.bfloat16](grad_output)
        elif output._dtype == DType.float32:
            _fill_ones[DType.float32](grad_output)
        elif output._dtype == DType.float64:
            _fill_ones[DType.float64](grad_output)
        else:
            for i in range(output.numel()):
                grad_output._set_float64(i, 1.0)
        var analytical_grad = backward_fn(grad_output, input)

        var numerical_grad = zeros_like(input)
        var input_copy_plus = input.clone()
        var input_copy_minus = input.clone()

        _dispatch_check_gradients_perturb(
            forward_fn, input, input_copy_plus, input_copy_minus,
            numerical_grad, epsilon,
        )

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


def check_gradients_verbose[F: NumericalForward, B: NumericalBackward](
    forward_fn: F,
    backward_fn: B,
    input: AnyTensor,
    epsilon: Float64 = 3e-4,
    tolerance: Float64 = 1e-2,
    print_all: Bool = False,
) raises -> Bool:
    """Gradient checking with detailed output — trait-parameterized overload.

    Same as check_gradients but prints all differences, not just maximum.
    Useful for debugging specific gradient issues. Use this overload when
    forward_fn and backward_fn are structs implementing NumericalForward
    and NumericalBackward traits.

    Args:
        forward_fn: A struct implementing NumericalForward.
        backward_fn: A struct implementing NumericalBackward.
        input: Input tensor.
        epsilon: Finite difference step size.
        tolerance: Maximum allowed difference.
        print_all: If True, print all gradients (even passing ones).

    Returns:
        True if gradients correct, False otherwise.

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
        if output._dtype == DType.float16:
            _fill_ones[DType.float16](grad_output)
        elif output._dtype == DType.bfloat16:
            _fill_ones[DType.bfloat16](grad_output)
        elif output._dtype == DType.float32:
            _fill_ones[DType.float32](grad_output)
        elif output._dtype == DType.float64:
            _fill_ones[DType.float64](grad_output)
        else:
            for i in range(output.numel()):
                grad_output._set_float64(i, 1.0)
        var analytical_grad = backward_fn(grad_output, input)

        var numerical_grad = zeros_like(input)
        var input_copy_plus = input.clone()
        var input_copy_minus = input.clone()

        _dispatch_check_gradients_perturb_trait(
            forward_fn, input, input_copy_plus, input_copy_minus,
            numerical_grad, epsilon,
        )

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


def relative_error(analytical: Float64, numerical: Float64) -> Float64:
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


# ============================================================================
# Dtype-dispatched perturbation loop for compute_numerical_gradient
# ============================================================================


def _compute_numerical_grad_perturb[
    dtype: DType,
](
    forward_fn: def(AnyTensor) raises -> AnyTensor,
    x: AnyTensor,
    grad: AnyTensor,
    epsilon: Float64,
) raises:
    """Perturbation loop for compute_numerical_gradient using typed pointers."""
    var x_ptr = x.data_ptr[dtype]()
    var grad_ptr = grad.data_ptr[dtype]()

    for i in range(x.numel()):
        var original_val = Float64(x_ptr[i])

        # Compute f(x + ε)
        x_ptr[i] = Scalar[dtype](original_val + epsilon)
        var f_plus = forward_fn(x)

        # Compute f(x - ε)
        x_ptr[i] = Scalar[dtype](original_val - epsilon)
        var f_minus = forward_fn(x)

        # Restore original value
        x_ptr[i] = Scalar[dtype](original_val)

        # Central difference: (f(x+ε) - f(x-ε)) / 2ε
        # Use data_ptr[dtype]() to keep f_plus/f_minus alive for the loop
        # scope. Per-element _get_float64 bitcasts can trigger ASAP destruction
        # of temporary tensors returned by forward_fn (modular/modular#6187).
        var f_plus_ptr = f_plus.data_ptr[dtype]()
        var f_minus_ptr = f_minus.data_ptr[dtype]()
        var grad_val: Float64
        if f_plus.numel() == 1:
            grad_val = (Float64(f_plus_ptr[0]) - Float64(f_minus_ptr[0])) / (
                2.0 * epsilon
            )
        else:
            grad_val = 0.0
            for j in range(f_plus.numel()):
                grad_val += (
                    Float64(f_plus_ptr[j]) - Float64(f_minus_ptr[j])
                ) / (2.0 * epsilon)

        grad_ptr[i] = Scalar[dtype](grad_val)


def _compute_numerical_grad_perturb_trait[
    dtype: DType,
    F: NumericalForward,
](
    forward_fn: F,
    x: AnyTensor,
    grad: AnyTensor,
    epsilon: Float64,
) raises:
    """Trait-based perturbation loop for NumericalForward implementors."""
    var x_ptr = x.data_ptr[dtype]()
    var grad_ptr = grad.data_ptr[dtype]()

    for i in range(x.numel()):
        var original_val = Float64(x_ptr[i])

        x_ptr[i] = Scalar[dtype](original_val + epsilon)
        var f_plus = forward_fn(x)

        x_ptr[i] = Scalar[dtype](original_val - epsilon)
        var f_minus = forward_fn(x)

        x_ptr[i] = Scalar[dtype](original_val)

        var f_plus_ptr = f_plus.data_ptr[dtype]()
        var f_minus_ptr = f_minus.data_ptr[dtype]()
        var grad_val: Float64
        if f_plus.numel() == 1:
            grad_val = (Float64(f_plus_ptr[0]) - Float64(f_minus_ptr[0])) / (
                2.0 * epsilon
            )
        else:
            grad_val = 0.0
            for j in range(f_plus.numel()):
                grad_val += (
                    Float64(f_plus_ptr[j]) - Float64(f_minus_ptr[j])
                ) / (2.0 * epsilon)

        grad_ptr[i] = Scalar[dtype](grad_val)


def compute_numerical_gradient(
    forward_fn: def(AnyTensor) raises -> AnyTensor,
    x: AnyTensor,
    epsilon: Float64 = 3e-4,  # Changed from 1e-5 - see #2704
) raises -> AnyTensor:
    """Compute numerical gradient using finite differences.

        Uses central difference formula: ∇f(x) ≈ (f(x + ε) - f(x - ε)) / 2ε.

        This is the gold standard for validating analytical gradients. The central
        difference method has O(ε²) error compared to O(ε) for forward/backward
        differences, making it much more accurate.

    Args:
            forward_fn: Forward function to differentiate.
            x: Input tensor at which to compute gradient.
            epsilon: Small perturbation for finite differences (default: 1e-5).

    Returns:
            AnyTensor containing numerical gradient, same shape as x.

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
            def relu_forward(x: AnyTensor) raises -> AnyTensor:
                return relu(x)

            var x = AnyTensor(List[Int](), DType.float32)
            var numerical_grad = compute_numerical_gradient(relu_forward, x)
            var analytical_grad = relu_backward(ones_like(x), x)
            assert_gradients_close(analytical_grad, numerical_grad, rtol=1e-4)
            ```
    """
    # Create gradient tensor (same shape as input)
    var grad = zeros_like(x)

    # Dispatch to dtype-specific perturbation loop (fixes #5104)
    if x._dtype == DType.float16:
        _compute_numerical_grad_perturb[DType.float16](
            forward_fn, x, grad, epsilon,
        )
    elif x._dtype == DType.bfloat16:
        _compute_numerical_grad_perturb[DType.bfloat16](
            forward_fn, x, grad, epsilon,
        )
    elif x._dtype == DType.float32:
        _compute_numerical_grad_perturb[DType.float32](
            forward_fn, x, grad, epsilon,
        )
    elif x._dtype == DType.float64:
        _compute_numerical_grad_perturb[DType.float64](
            forward_fn, x, grad, epsilon,
        )
    else:
        raise Error(
            "Unsupported dtype for gradient checking: " + String(x._dtype)
        )

    return grad^


def compute_numerical_gradient[F: NumericalForward](
    forward_fn: F,
    x: AnyTensor,
    epsilon: Float64 = 3e-4,
) raises -> AnyTensor:
    """Compute numerical gradient for a NumericalForward trait implementor.

    Overload for use with capturing closures wrapped in a struct implementing
    NumericalForward. Required in Mojo 0.26.3 because non-escaping closures
    cannot be passed as runtime function arguments.
    """
    var grad = zeros_like(x)
    if x._dtype == DType.float16:
        _compute_numerical_grad_perturb_trait[DType.float16, F](
            forward_fn, x, grad, epsilon,
        )
    elif x._dtype == DType.bfloat16:
        _compute_numerical_grad_perturb_trait[DType.bfloat16, F](
            forward_fn, x, grad, epsilon,
        )
    elif x._dtype == DType.float32:
        _compute_numerical_grad_perturb_trait[DType.float32, F](
            forward_fn, x, grad, epsilon,
        )
    elif x._dtype == DType.float64:
        _compute_numerical_grad_perturb_trait[DType.float64, F](
            forward_fn, x, grad, epsilon,
        )
    else:
        raise Error(
            "Unsupported dtype for gradient checking: " + String(x._dtype)
        )
    return grad^


# ============================================================================
# Dtype-dispatched perturbation loop for compute_sampled_numerical_gradient
# ============================================================================


def _compute_sampled_grad_perturb[
    dtype: DType,
](
    forward_fn: def(AnyTensor) raises -> AnyTensor,
    x: AnyTensor,
    indices: List[Int],
    mut gradients: List[IndexGradientPair],
    epsilon: Float64,
) raises:
    """Perturbation loop for sampled gradient computation using typed pointers."""
    var x_ptr = x.data_ptr[dtype]()

    for idx in indices:
        var original_val = Float64(x_ptr[idx])

        # f(x + ε)
        x_ptr[idx] = Scalar[dtype](original_val + epsilon)
        var f_plus = forward_fn(x)
        # Use data_ptr[dtype]() to keep f_plus alive across the loop (modular/modular#6187)
        var f_plus_ptr = f_plus.data_ptr[dtype]()
        var f_plus_sum: Float64 = 0.0
        for j in range(f_plus.numel()):
            f_plus_sum += Float64(f_plus_ptr[j])

        # f(x - ε)
        x_ptr[idx] = Scalar[dtype](original_val - epsilon)
        var f_minus = forward_fn(x)
        # Use data_ptr[dtype]() to keep f_minus alive across the loop (modular/modular#6187)
        var f_minus_ptr = f_minus.data_ptr[dtype]()
        var f_minus_sum: Float64 = 0.0
        for j in range(f_minus.numel()):
            f_minus_sum += Float64(f_minus_ptr[j])

        # Restore original
        x_ptr[idx] = Scalar[dtype](original_val)

        # Compute gradient: (f(x + ε) - f(x - ε)) / (2ε)
        var grad = (f_plus_sum - f_minus_sum) / (2.0 * epsilon)
        gradients.append(IndexGradientPair(idx, grad))


def _compute_sampled_grad_perturb_trait[
    dtype: DType,
    F: NumericalForward,
](
    forward_fn: F,
    x: AnyTensor,
    indices: List[Int],
    mut gradients: List[IndexGradientPair],
    epsilon: Float64,
) raises:
    """Trait-based perturbation loop for sampled gradient computation."""
    var x_ptr = x.data_ptr[dtype]()

    for idx in indices:
        var original_val = Float64(x_ptr[idx])

        x_ptr[idx] = Scalar[dtype](original_val + epsilon)
        var f_plus = forward_fn(x)
        var f_plus_ptr = f_plus.data_ptr[dtype]()
        var f_plus_sum: Float64 = 0.0
        for j in range(f_plus.numel()):
            f_plus_sum += Float64(f_plus_ptr[j])

        x_ptr[idx] = Scalar[dtype](original_val - epsilon)
        var f_minus = forward_fn(x)
        var f_minus_ptr = f_minus.data_ptr[dtype]()
        var f_minus_sum: Float64 = 0.0
        for j in range(f_minus.numel()):
            f_minus_sum += Float64(f_minus_ptr[j])

        x_ptr[idx] = Scalar[dtype](original_val)

        var grad = (f_plus_sum - f_minus_sum) / (2.0 * epsilon)
        gradients.append(IndexGradientPair(idx, grad))


def compute_sampled_numerical_gradient(
    forward_fn: def(AnyTensor) raises -> AnyTensor,
    x: AnyTensor,
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
        forward_fn: Forward function to differentiate.
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
        def forward(x: AnyTensor) raises -> AnyTensor:
            return relu(x)

        var x = AnyTensor([100, 100], DType.float32)
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

    # Compute gradients for sampled indices using typed pointers (fixes #5104)
    var gradients = List[IndexGradientPair]()

    if x._dtype == DType.float16:
        _compute_sampled_grad_perturb[DType.float16](
            forward_fn, x, indices, gradients, epsilon,
        )
    elif x._dtype == DType.bfloat16:
        _compute_sampled_grad_perturb[DType.bfloat16](
            forward_fn, x, indices, gradients, epsilon,
        )
    elif x._dtype == DType.float32:
        _compute_sampled_grad_perturb[DType.float32](
            forward_fn, x, indices, gradients, epsilon,
        )
    elif x._dtype == DType.float64:
        _compute_sampled_grad_perturb[DType.float64](
            forward_fn, x, indices, gradients, epsilon,
        )
    else:
        raise Error(
            "Unsupported dtype for gradient checking: " + String(x._dtype)
        )

    return gradients^


def compute_sampled_numerical_gradient[F: NumericalForward](
    forward_fn: F,
    x: AnyTensor,
    num_samples: Int = 100,
    epsilon: Float64 = 3e-4,
    seed: Int = 42,
) raises -> List[IndexGradientPair]:
    """NumericalForward trait overload of compute_sampled_numerical_gradient.

    For use with capturing closures wrapped in a struct implementing NumericalForward.
    """
    var numel = x.numel()
    var actual_samples = min(num_samples, numel)
    var indices = List[Int]()
    indices.append(0)
    indices.append(numel - 1)
    var rng_state = seed
    var samples_needed = actual_samples - 2
    var count = 0
    while count < samples_needed:
        rng_state = ((rng_state * 1103515245 + 12345) % 2147483648) % numel
        indices.append(rng_state)
        count += 1
    var gradients = List[IndexGradientPair]()
    if x._dtype == DType.float16:
        _compute_sampled_grad_perturb_trait[DType.float16, F](
            forward_fn, x, indices, gradients, epsilon,
        )
    elif x._dtype == DType.bfloat16:
        _compute_sampled_grad_perturb_trait[DType.bfloat16, F](
            forward_fn, x, indices, gradients, epsilon,
        )
    elif x._dtype == DType.float32:
        _compute_sampled_grad_perturb_trait[DType.float32, F](
            forward_fn, x, indices, gradients, epsilon,
        )
    elif x._dtype == DType.float64:
        _compute_sampled_grad_perturb_trait[DType.float64, F](
            forward_fn, x, indices, gradients, epsilon,
        )
    else:
        raise Error(
            "Unsupported dtype for gradient checking: " + String(x._dtype)
        )
    return gradients^


def assert_sampled_gradients_close(
    analytical_grad: AnyTensor,
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
        atol: Absolute tolerance threshold.
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


def assert_gradients_close(
    analytical: AnyTensor,
    numerical: AnyTensor,
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


# ============================================================================
# Dtype-dispatched perturbation loop for check_gradient
# ============================================================================


def _check_gradient_perturb[
    dtype: DType,
](
    forward_fn: def(AnyTensor) raises -> AnyTensor,
    x: AnyTensor,
    grad_output: AnyTensor,
    grad: AnyTensor,
    eps: Float64,
) raises:
    """Perturbation loop for check_gradient using typed pointers.

    Uses data_ptr[dtype]() for the input tensor to avoid per-element bitcast.
    Each iteration creates fresh clones (x_plus, x_minus), so we get a typed
    pointer to x once for reading old_val, and to grad for writing results.
    """
    var x_ptr = x.data_ptr[dtype]()
    var grad_ptr = grad.data_ptr[dtype]()

    for i in range(x.numel()):
        # Create deep copies to avoid corrupting original x
        var x_plus = x.clone()
        var old_val = Float64(x_ptr[i])
        # Use typed pointer for setting the perturbed value in the clone
        var plus_ptr = x_plus.data_ptr[dtype]()
        plus_ptr[i] = Scalar[dtype](old_val + eps)
        # Capture the ACTUAL quantized perturbation (important for FP16/BF16 where
        # old_val ± eps may round to different bucket sizes, making the effective
        # perturbation asymmetric; using the nominal eps in the denominator then
        # produces a wrong gradient — use the real Δx instead).
        var actual_x_plus = Float64(plus_ptr[i])
        var out_plus = forward_fn(x_plus)
        # Use data_ptr[dtype]() to keep out_plus alive across the loop (modular/modular#6187)
        var out_plus_ptr = out_plus.data_ptr[dtype]()
        var grad_out_ptr = grad_output.data_ptr[dtype]()
        var loss_plus: Float64 = 0.0
        for j in range(out_plus.numel()):
            loss_plus += Float64(out_plus_ptr[j]) * Float64(grad_out_ptr[j])

        # Backward perturbation
        var x_minus = x.clone()
        var minus_ptr = x_minus.data_ptr[dtype]()
        minus_ptr[i] = Scalar[dtype](old_val - eps)
        var actual_x_minus = Float64(minus_ptr[i])
        var out_minus = forward_fn(x_minus)
        # Use data_ptr[dtype]() to keep out_minus alive across the loop (modular/modular#6187)
        var out_minus_ptr = out_minus.data_ptr[dtype]()
        var loss_minus: Float64 = 0.0
        for j in range(out_minus.numel()):
            loss_minus += Float64(out_minus_ptr[j]) * Float64(grad_out_ptr[j])

        # Central difference: use actual quantized step size, not the requested eps.
        # For FP32/FP64 actual_eps ≈ eps; for FP16/BF16 they can differ due to rounding.
        var actual_eps = (actual_x_plus - actual_x_minus) / 2.0
        # Fall back to requested eps if perturbations collapsed to the same bucket.
        var denom = actual_eps if actual_eps > 0.0 else eps
        var numerical_grad = (loss_plus - loss_minus) / (2.0 * denom)
        grad_ptr[i] = Scalar[dtype](numerical_grad)


def _check_gradient_perturb[
    dtype: DType,
    F: NumericalForward,
](
    forward_fn: F,
    x: AnyTensor,
    grad_output: AnyTensor,
    grad: AnyTensor,
    eps: Float64,
) raises:
    """Perturbation loop for check_gradient using typed pointers — trait-parameterized overload.

    Uses data_ptr[dtype]() for the input tensor to avoid per-element bitcast.
    Each iteration creates fresh clones (x_plus, x_minus), so we get a typed
    pointer to x once for reading old_val, and to grad for writing results.

    Use this overload when forward_fn is a struct implementing NumericalForward.
    """
    var x_ptr = x.data_ptr[dtype]()
    var grad_ptr = grad.data_ptr[dtype]()

    for i in range(x.numel()):
        # Create deep copies to avoid corrupting original x
        var x_plus = x.clone()
        var old_val = Float64(x_ptr[i])
        # Use typed pointer for setting the perturbed value in the clone
        var plus_ptr = x_plus.data_ptr[dtype]()
        plus_ptr[i] = Scalar[dtype](old_val + eps)
        # Capture the ACTUAL quantized perturbation (important for FP16/BF16 where
        # old_val ± eps may round to different bucket sizes, making the effective
        # perturbation asymmetric; using the nominal eps in the denominator then
        # produces a wrong gradient — use the real Δx instead).
        var actual_x_plus = Float64(plus_ptr[i])
        var out_plus = forward_fn(x_plus)
        # Use data_ptr[dtype]() to keep out_plus alive across the loop (modular/modular#6187)
        var out_plus_ptr = out_plus.data_ptr[dtype]()
        var grad_out_ptr = grad_output.data_ptr[dtype]()
        var loss_plus: Float64 = 0.0
        for j in range(out_plus.numel()):
            loss_plus += Float64(out_plus_ptr[j]) * Float64(grad_out_ptr[j])

        # Backward perturbation
        var x_minus = x.clone()
        var minus_ptr = x_minus.data_ptr[dtype]()
        minus_ptr[i] = Scalar[dtype](old_val - eps)
        var actual_x_minus = Float64(minus_ptr[i])
        var out_minus = forward_fn(x_minus)
        # Use data_ptr[dtype]() to keep out_minus alive across the loop (modular/modular#6187)
        var out_minus_ptr = out_minus.data_ptr[dtype]()
        var loss_minus: Float64 = 0.0
        for j in range(out_minus.numel()):
            loss_minus += Float64(out_minus_ptr[j]) * Float64(grad_out_ptr[j])

        # Central difference: use actual quantized step size, not the requested eps.
        # For FP32/FP64 actual_eps ≈ eps; for FP16/BF16 they can differ due to rounding.
        var actual_eps = (actual_x_plus - actual_x_minus) / 2.0
        # Fall back to requested eps if perturbations collapsed to the same bucket.
        var denom = actual_eps if actual_eps > 0.0 else eps
        var numerical_grad = (loss_plus - loss_minus) / (2.0 * denom)
        grad_ptr[i] = Scalar[dtype](numerical_grad)


# ============================================================================
# Epsilon and tolerance auto-selection helper
# ============================================================================


struct _EpsAtol(Copyable, Movable):
    """Helper struct returned by _select_epsilon."""
    var eps: Float64
    var atol: Float64

    def __init__(out self, eps: Float64, atol: Float64):
        self.eps = eps
        self.atol = atol


def _select_epsilon(dtype: DType, requested_eps: Float64, atol: Float64) -> _EpsAtol:
    """Auto-select finite-difference epsilon and minimum atol for a given dtype.

    For reduced-precision dtypes (FP16/BF16) the standard epsilon (1e-5) rounds
    to zero, producing degenerate numerical gradients. This function selects a
    dtype-appropriate epsilon and ensures atol is large enough to accommodate the
    increased numerical error.

    Args:
        dtype: Element dtype of the tensor being checked.
        requested_eps: Caller-supplied epsilon; 0.0 means "auto-select".
        atol: Caller-supplied absolute tolerance.

    Returns:
        _EpsAtol with chosen eps and (possibly adjusted) atol.
    """
    if requested_eps != 0.0:
        return _EpsAtol(requested_eps, atol)
    if dtype == DType.float16 or dtype == DType.bfloat16:
        # FP16/BF16 machine epsilon ~9.77e-4; 1e-5 rounds to zero
        var new_atol = atol if atol >= 1e-2 else 1e-2
        return _EpsAtol(1e-3, new_atol)
    elif dtype == DType.float32:
        var new_atol = atol if atol >= 1e-4 else 1e-4
        return _EpsAtol(1e-4, new_atol)
    elif dtype == DType.float64:
        var new_atol = atol if atol >= 1e-7 else 1e-7
        return _EpsAtol(1e-7, new_atol)
    else:
        return _EpsAtol(1e-5, atol)


def check_gradient(
    forward_fn: def(AnyTensor) raises -> AnyTensor,
    backward_fn: def(AnyTensor, AnyTensor) raises -> AnyTensor,
    x: AnyTensor,
    grad_output: AnyTensor,
    epsilon: Float64 = 0.0,  # Auto-select based on dtype if 0.0
    rtol: Float64 = 1e-3,
    atol: Float64 = 1e-6,
) raises:
    """Comprehensive gradient check helper.

        Combines numerical gradient computation and comparison in one call.
        This is the recommended way to validate backward passes in tests.

    Args:
            forward_fn: Forward function to differentiate.
            backward_fn: Backward function computing analytical gradients.
            x: Input tensor.
            grad_output: Gradient from upstream (typically ones_like(output)).
            epsilon: Perturbation size for finite differences (0.0 = auto-select).
            rtol: Relative tolerance.
            atol: Absolute tolerance.

    Raises:
            Error: If gradients don't match within tolerance.

        Example:
            ```mojo
            def test_relu_gradient() raises:
                var x = AnyTensor(List[Int](), DType.float32)
                # ... initialize x with test values ...

                def forward(inp: AnyTensor) raises -> AnyTensor:
                    return relu(inp)

                def backward_wrapper(grad: AnyTensor, x: AnyTensor) raises -> AnyTensor:
                    return relu_backward(grad, x)

                var grad_out = ones_like(relu(x))
                check_gradient(forward, backward_wrapper, x, grad_out)
            ```
    """
    # Auto-select epsilon and atol using the shared helper (avoids code duplication
    # and keeps each overload small enough to avoid ASAN JIT compilation crashes).
    var sel = _select_epsilon(x._dtype, epsilon, atol)
    var eps = sel.eps
    var auto_atol = sel.atol

    # Compute analytical gradient
    var analytical = backward_fn(grad_output, x)

    # Compute numerical gradient using typed pointer perturbation (fixes #5104)
    var grad = zeros_like(x)

    if x._dtype == DType.float16:
        _check_gradient_perturb[DType.float16](
            forward_fn, x, grad_output, grad, eps,
        )
    elif x._dtype == DType.bfloat16:
        _check_gradient_perturb[DType.bfloat16](
            forward_fn, x, grad_output, grad, eps,
        )
    elif x._dtype == DType.float32:
        _check_gradient_perturb[DType.float32](
            forward_fn, x, grad_output, grad, eps,
        )
    elif x._dtype == DType.float64:
        _check_gradient_perturb[DType.float64](
            forward_fn, x, grad_output, grad, eps,
        )
    else:
        raise Error(
            "Unsupported dtype for gradient checking: " + String(x._dtype)
        )

    # Compare
    assert_gradients_close(
        analytical,
        grad,
        rtol,
        auto_atol,
        "Gradient check failed for " + String(x._dtype),
    )


def check_gradient[F: NumericalForward, B: NumericalBackward](
    forward_fn: F,
    backward_fn: B,
    x: AnyTensor,
    grad_output: AnyTensor,
    epsilon: Float64 = 0.0,  # Auto-select based on dtype if 0.0
    rtol: Float64 = 1e-3,
    atol: Float64 = 1e-6,
) raises:
    """Comprehensive gradient check helper — trait-parameterized overload.

    Combines numerical gradient computation and comparison in one call.
    This is the recommended way to validate backward passes in tests.
    Use this overload when forward_fn and backward_fn are structs
    implementing NumericalForward and NumericalBackward traits.

    Args:
        forward_fn: A struct implementing NumericalForward.
        backward_fn: A struct implementing NumericalBackward.
        x: Input tensor.
        grad_output: Gradient from upstream (typically ones_like(output)).
        epsilon: Perturbation size for finite differences (0.0 = auto-select).
        rtol: Relative tolerance.
        atol: Absolute tolerance.

    Raises:
        Error: If gradients don't match within tolerance.
    """
    # Auto-select epsilon and atol based on dtype if not specified
    var sel = _select_epsilon(x._dtype, epsilon, atol)
    var eps = sel.eps
    var auto_atol = sel.atol

    # Compute analytical gradient
    var analytical = backward_fn(grad_output, x)

    # Compute numerical gradient using typed pointer perturbation (fixes #5104)
    var grad = zeros_like(x)

    if x._dtype == DType.float16:
        _check_gradient_perturb[DType.float16](
            forward_fn, x, grad_output, grad, eps,
        )
    elif x._dtype == DType.bfloat16:
        _check_gradient_perturb[DType.bfloat16](
            forward_fn, x, grad_output, grad, eps,
        )
    elif x._dtype == DType.float32:
        _check_gradient_perturb[DType.float32](
            forward_fn, x, grad_output, grad, eps,
        )
    elif x._dtype == DType.float64:
        _check_gradient_perturb[DType.float64](
            forward_fn, x, grad_output, grad, eps,
        )
    else:
        raise Error(
            "Unsupported dtype for gradient checking: " + String(x._dtype)
        )

    # Compare
    assert_gradients_close(
        analytical,
        grad,
        rtol,
        auto_atol,
        "Gradient check failed for " + String(x._dtype),
    )
