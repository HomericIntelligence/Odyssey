"""Example: Custom Layers - Parametric ReLU

This example demonstrates implementing a custom PReLU activation.

PReLU is similar to Leaky ReLU but uses learnable parameters for the
negative slope. The formula is: PReLU(x) = max(alpha*x, x) = max(0, x) + alpha * min(0, x).

Usage:
    pixi run mojo run examples/custom-layers/prelu_activation.mojo

See documentation: docs/advanced/custom-layers.md
"""

from shared.core import AnyTensor, zeros, clip


fn prelu_simple(input: AnyTensor, alpha: Float32) raises -> AnyTensor:
    """Apply PReLU activation element-wise.

    Args:
        input: Input tensor.
        alpha: Negative slope parameter.

    Returns:
        PReLU(input) = max(0, x) + alpha * min(0, x).
    """
    # Use clip to implement max(0, x) and min(0, x)
    var positive = clip(input, 0.0, 1e9)  # max(0, x)

    # For negative part: min(0, x) = -max(0, -x)
    var result = zeros(input.shape(), input.dtype())
    var input_ptr = input._data.bitcast[Float32]()
    var result_ptr = result._data.bitcast[Float32]()

    for i in range(input.numel()):
        var x = input_ptr[i]
        if x > 0:
            result_ptr[i] = x
        else:
            result_ptr[i] = alpha * x

    return result^


fn main() raises:
    """Demonstrate PReLU activation."""

    print("\n=== PReLU Activation Example ===\n")

    # Create sample input tensor with both positive and negative values
    var input_data = List[Float32]()
    input_data.append(-2.0)
    input_data.append(-1.0)
    input_data.append(0.0)
    input_data.append(1.0)
    input_data.append(2.0)
    var input = AnyTensor(input_data^)

    print("Input values:")
    var input_ptr = input._data.bitcast[Float32]()
    for i in range(input.numel()):
        print("  ", input_ptr[i])

    # Create learnable alpha parameter (slope for negative values)
    # Using alpha = 0.25 for this demonstration
    var alpha: Float32 = 0.25
    print("\nAlpha (negative slope):", alpha)

    # Apply PReLU activation
    var output = prelu_simple(input, alpha)

    print("\nOutput values:")
    var output_ptr = output._data.bitcast[Float32]()
    for i in range(output.numel()):
        print("  ", output_ptr[i])

    print("\nExpected: [-0.5, -0.25, 0.0, 1.0, 2.0]")
    print("  (Positive values unchanged, negative values scaled by alpha)")
    print("\nPReLU activation example complete!")
