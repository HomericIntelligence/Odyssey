"""Gradient container types for backward pass functions.

Provides type-safe containers for multiple gradient returns, replacing tuple
return types which are not fully supported in the current Mojo version.

This module defines:

- GradientPair: For binary operations returning 2 gradients.
- GradientTriple: For ternary operations returning 3 gradients.
- GradientQuad: For quaternary operations returning 4 gradients.
- Conv2dNoBiasGradient: For conv2d_no_bias_backward with semantic field names.
- DepthwiseConv2dNoBiasGradient: For depthwise_conv2d_no_bias_backward.
- DepthwiseSeparableConv2dGradient: For depthwise_separable_conv2d_backward with semantic field names.

Type Selection Guide:
    Choose the container that matches the number of inputs your backward function
    differentiates with respect to. This table maps operation categories to types:

    ┌─────────────────────────┬──────────────────┬──────────────────────────┐
    │ Operation Category      │ Container Type   │ Field Names              │
    ├─────────────────────────┼──────────────────┼──────────────────────────┤
    │ Binary Operations       │ GradientPair     │ grad_a, grad_b           │
    │ (add, subtract, multiply│                  │                          │
    │  divide, matmul, etc.)  │                  │                          │
    ├─────────────────────────┼──────────────────┼──────────────────────────┤
    │ Linear Layer            │ GradientTriple   │ grad_input, grad_weights,│
    │ (linear_backward)       │                  │ grad_bias                │
    ├─────────────────────────┼──────────────────┼──────────────────────────┤
    │ Conv2D with Bias        │ GradientTriple   │ grad_input, grad_weights,│
    │ (conv2d_backward)       │                  │ grad_bias                │
    ├─────────────────────────┼──────────────────┼──────────────────────────┤
    │ Conv2D without Bias     │ Conv2dNoBias     │ grad_input, grad_weights │
    │ (conv2d_no_bias_bwd)    │ Gradient         │                          │
    ├─────────────────────────┼──────────────────┼──────────────────────────┤
    │ Depthwise Conv2D        │ Depthwise        │ grad_input, grad_weights │
    │ no Bias                 │ Conv2dNoBias     │                          │
    │ (depthwise_conv2d_      │ Gradient         │                          │
    │  no_bias_backward)      │                  │                          │
    ├─────────────────────────┼──────────────────┼──────────────────────────┤
    │ Batch Normalization     │ GradientQuad     │ grad_input, grad_weights,│
    │ and other 4-input ops   │                  │ grad_bias, grad_extra    │
    ├─────────────────────────┼──────────────────┼──────────────────────────┤
    │ >4 gradients            │ Custom struct    │ Meaningful semantic names│
    │ (new operations)        │ (define new)     │                          │
    └─────────────────────────┴──────────────────┴──────────────────────────┘

When to Create a New Type:
    If your operation returns more than 4 gradients or requires custom field
    semantics, define a dedicated named struct (don't extend GradientQuad) so
    that field names meaningfully map to operation inputs.

Guidelines for Implementation:
    1. Count the number of inputs your operation differentiates with respect to.
    2. Use the table above to select the appropriate container type.
    3. For specialized cases (bias-free convolutions), use the dedicated type.
    4. Implement __init__ following the pattern in this module.
    5. Add docstrings with concrete examples showing usage.
"""

from .extensor import AnyTensor


struct GradientPair(Copyable, Movable):
    """Container for gradients from binary operations.

    Used for backward functions that compute gradients with respect to
    two inputs (e.g., add_backward, multiply_backward).

    Attributes:
        grad_a: Gradient with respect to first input.
        grad_b: Gradient with respect to second input.

    Examples:
        ```mojo
        var grads = add_backward(grad_output, a_shape, b_shape)
        var grad_a = grads.grad_a
        var grad_b = grads.grad_b
        ```
    """

    var grad_a: AnyTensor
    """Gradient with respect to first input."""
    var grad_b: AnyTensor
    """Gradient with respect to second input."""

    fn __init__(out self, var grad_a: AnyTensor, var grad_b: AnyTensor):
        """Initialize gradient pair.

        Args:
            grad_a: Gradient tensor for first input.
            grad_b: Gradient tensor for second input.
        """
        self.grad_a = grad_a^
        self.grad_b = grad_b^


struct GradientTriple(Copyable, Movable):
    """Container for gradients from ternary operations.

    Used for backward functions that compute gradients with respect to
    three inputs (e.g., linear_backward, conv2d_backward).

    Attributes:
        grad_input: Gradient with respect to input activation.
        grad_weights: Gradient with respect to weights.
        grad_bias: Gradient with respect to bias.

    Examples:
        ```mojo
        var grads = linear_backward(grad_output, x, weights)
        var grad_input = grads.grad_input
        var grad_weights = grads.grad_weights
        var grad_bias = grads.grad_bias
        ```
    """

    var grad_input: AnyTensor
    """Gradient with respect to input activation."""
    var grad_weights: AnyTensor
    """Gradient with respect to weights."""
    var grad_bias: AnyTensor
    """Gradient with respect to bias."""

    fn __init__(
        out self,
        var grad_input: AnyTensor,
        var grad_weights: AnyTensor,
        var grad_bias: AnyTensor,
    ):
        """Initialize gradient triple.

        Args:
            grad_input: Gradient tensor for input.
            grad_weights: Gradient tensor for weights.
            grad_bias: Gradient tensor for bias.
        """
        self.grad_input = grad_input^
        self.grad_weights = grad_weights^
        self.grad_bias = grad_bias^


struct GradientQuad(Copyable, Movable):
    """Container for gradients from quaternary operations.

    Used for backward functions that compute gradients with respect to
    four inputs (reserved for future use in complex backward passes).

    Attributes:
        grad_a: Gradient with respect to first input.
        grad_b: Gradient with respect to second input.
        grad_c: Gradient with respect to third input.
        grad_d: Gradient with respect to fourth input.

    Examples:
        ```mojo
        var grads = complex_backward(grad_output, a, b, c, d)
        var grad_a = grads.grad_a
        var grad_b = grads.grad_b
        var grad_c = grads.grad_c
        var grad_d = grads.grad_d
        ```
    """

    var grad_a: AnyTensor
    """Gradient with respect to first input."""
    var grad_b: AnyTensor
    """Gradient with respect to second input."""
    var grad_c: AnyTensor
    """Gradient with respect to third input."""
    var grad_d: AnyTensor
    """Gradient with respect to fourth input."""

    fn __init__(
        out self,
        var grad_a: AnyTensor,
        var grad_b: AnyTensor,
        var grad_c: AnyTensor,
        var grad_d: AnyTensor,
    ):
        """Initialize gradient quad.

        Args:
            grad_a: Gradient tensor for first input.
            grad_b: Gradient tensor for second input.
            grad_c: Gradient tensor for third input.
            grad_d: Gradient tensor for fourth input.
        """
        self.grad_a = grad_a^
        self.grad_b = grad_b^
        self.grad_c = grad_c^
        self.grad_d = grad_d^


struct Conv2dNoBiasGradient(Copyable, Movable):
    """Container for gradients from conv2d_no_bias_backward.

    Uses the same field naming convention as GradientTriple to align with
    conv2d_backward, making no-bias and biased conv backward results consistent.

    Attributes:
        grad_input: Gradient with respect to input activation.
        grad_weights: Gradient with respect to convolution kernel.

    Examples:
        ```mojo
        var grads = conv2d_no_bias_backward(grad_output, x, kernel)
        var grad_input = grads.grad_input
        var grad_weights = grads.grad_weights
        ```
    """

    var grad_input: AnyTensor
    """Gradient with respect to input activation."""
    var grad_weights: AnyTensor
    """Gradient with respect to convolution kernel."""

    fn __init__(out self, var grad_input: AnyTensor, var grad_weights: AnyTensor):
        """Initialize conv2d no-bias gradient.

        Args:
            grad_input: Gradient tensor for input.
            grad_weights: Gradient tensor for kernel weights.
        """
        self.grad_input = grad_input^
        self.grad_weights = grad_weights^


struct DepthwiseConv2dNoBiasGradient(Copyable, Movable):
    """Container for gradients from depthwise_conv2d_no_bias_backward.

    Uses the same field naming convention as GradientTriple to align with
    depthwise_conv2d_backward.

    Attributes:
        grad_input: Gradient with respect to input activation.
        grad_weights: Gradient with respect to depthwise convolution kernel.

    Examples:
        ```mojo
        var grads = depthwise_conv2d_no_bias_backward(grad_output, x, kernel)
        var grad_input = grads.grad_input
        var grad_weights = grads.grad_weights
        ```
    """

    var grad_input: AnyTensor
    """Gradient with respect to input activation."""
    var grad_weights: AnyTensor
    """Gradient with respect to depthwise convolution kernel."""

    fn __init__(out self, var grad_input: AnyTensor, var grad_weights: AnyTensor):
        """Initialize depthwise conv2d no-bias gradient.

        Args:
            grad_input: Gradient tensor for input.
            grad_weights: Gradient tensor for kernel weights.
        """
        self.grad_input = grad_input^
        self.grad_weights = grad_weights^


struct DepthwiseSeparableConv2dGradient(Copyable, Movable):
    """Container for gradients from depthwise_separable_conv2d_backward.

    Represents gradients for depthwise separable convolution which applies
    depthwise convolution followed by pointwise convolution, returning gradients
    with respect to input, depthwise kernel, pointwise kernel, and bias.

    Attributes:
        grad_input: Gradient with respect to input activation.
        grad_depthwise_kernel: Gradient with respect to depthwise convolution kernel.
        grad_pointwise_kernel: Gradient with respect to pointwise convolution kernel.
        grad_bias: Gradient with respect to bias.

    Examples:
        ```mojo
        var grads = depthwise_separable_conv2d_backward(
            grad_output, x, depthwise_kernel, pointwise_kernel
        )
        var grad_input = grads.grad_input
        var grad_dw_kernel = grads.grad_depthwise_kernel
        var grad_pw_kernel = grads.grad_pointwise_kernel
        var grad_bias = grads.grad_bias
        ```
    """

    var grad_input: AnyTensor
    """Gradient with respect to input activation."""
    var grad_depthwise_kernel: AnyTensor
    """Gradient with respect to depthwise convolution kernel."""
    var grad_pointwise_kernel: AnyTensor
    """Gradient with respect to pointwise convolution kernel."""
    var grad_bias: AnyTensor
    """Gradient with respect to bias."""

    fn __init__(
        out self,
        var grad_input: AnyTensor,
        var grad_depthwise_kernel: AnyTensor,
        var grad_pointwise_kernel: AnyTensor,
        var grad_bias: AnyTensor,
    ):
        """Initialize depthwise separable conv2d gradient.

        Args:
            grad_input: Gradient tensor for input.
            grad_depthwise_kernel: Gradient tensor for depthwise kernel.
            grad_pointwise_kernel: Gradient tensor for pointwise kernel.
            grad_bias: Gradient tensor for bias.
        """
        self.grad_input = grad_input^
        self.grad_depthwise_kernel = grad_depthwise_kernel^
        self.grad_pointwise_kernel = grad_pointwise_kernel^
        self.grad_bias = grad_bias^
