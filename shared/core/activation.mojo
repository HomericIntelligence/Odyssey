"""Activation functions for neural networks.

Provides common activation functions including ReLU family, sigmoid, tanh,
softmax, and GELU with numerically stable implementations and gradient support.

Reference implementations follow PyTorch and TensorFlow conventions:
- ReLU family: max(0, x), max(alpha*x, x), parametric variants
- Sigmoid/Tanh: Numerically stable with input clipping
- Softmax: Log-sum-exp trick for numerical stability
- GELU: Both exact (erf) and approximate (tanh) implementations

Type support:
- ReLU family: float16, float32, float64, int8, int16, int32, int64, uint8, uint16, uint32, uint64
- Sigmoid/Tanh/Softmax/GELU: float16, float32, float64

Issues covered:
- #238-242: ReLU Family (ReLU, Leaky ReLU, PReLU)
- #243-247: Sigmoid and Tanh
- #248-252: Softmax and GELU
- #253-257: Activations module integration
"""

from math import exp, erf, sqrt, tanh as math_tanh, log as math_log
from collections import List
from .any_tensor import AnyTensor, full, zeros_like
from shared.tensor.tensor import Tensor
from .arithmetic import add, subtract, multiply
from .reduction import sum as tensor_sum, max as tensor_max
from .elementwise import log
from .dtype_dispatch import (
    dispatch_unary,
    dispatch_binary,
    dispatch_float_unary,
    dispatch_float_binary,
    dispatch_scalar,
    dispatch_softmax,
    dispatch_softmax_backward,
    dispatch_gelu,
    dispatch_gelu_backward,
    dispatch_hard_sigmoid,
    dispatch_hard_sigmoid_backward,
    dispatch_hard_swish,
    dispatch_hard_swish_backward,
    dispatch_hard_tanh,
    dispatch_hard_tanh_backward,
)
from .gradient_types import GradientPair
from .activation_ops import exp_scalar_f32, exp_scalar_f64
from .activation_constants import (
    RELU6_UPPER_BOUND,
    SIGMOID_CLIP_THRESHOLD,
)


# ============================================================================
# ReLU Family (#238-242)
# ============================================================================


@always_inline
fn _relu_op[T: DType](x: Scalar[T]) -> Scalar[T]:
    """ReLU operation: max(0, x).

    Parameters:
        T: Data type of the scalar (float16, float32, float64, int8, etc.).

    Args:
        x: Input scalar value.

    Returns:
        max(0, x).
    """
    return max(Scalar[T](0), x)


fn relu(tensor: AnyTensor) raises -> AnyTensor:
    """Apply ReLU (Rectified Linear Unit) activation: max(0, x).

        ReLU zeros out negative values while preserving positive values unchanged.
        This is the most common activation function in deep learning, promoting
        sparse activation patterns.

        Supported dtypes: float16, float32, float64, int8, int16, int32, int64,
                          uint8, uint16, uint32, uint64.

    Args:
            tensor: Input tensor of any shape.

    Returns:
            New tensor with ReLU applied element-wise.

    Raises:
            Error: If operation fails.

    Examples:
    ```
        var x = AnyTensor(...)  # [-2, -1, 0, 1, 2]
        var y = relu(x)        # [0, 0, 0, 1, 2]
    ```
    """
    return dispatch_unary[_relu_op](tensor)


@always_inline
fn _relu6_op[T: DType](x: Scalar[T]) -> Scalar[T]:
    """ReLU6 operation: min(max(0, x), RELU6_UPPER_BOUND)."""
    return min(max(Scalar[T](0), x), Scalar[T](RELU6_UPPER_BOUND))


fn relu6(tensor: AnyTensor) raises -> AnyTensor:
    """Apply ReLU6 activation: min(max(0, x), 6).

    ReLU6 clamps values to [0, 6], commonly used in MobileNet architectures.
    Supported dtypes: float16, float32, float64, int8, int16, int32, int64.

    Args:
        tensor: Input tensor of any shape.

    Returns:
        New tensor with ReLU6 applied element-wise.

    Examples:
    ```mojo
        var x = AnyTensor(...)  # [-2, 0, 3, 8, 10]
        var y = relu6(x)       # [0, 0, 3, 6, 6]
    ```
    """
    return dispatch_unary[_relu6_op](tensor)


fn leaky_relu(tensor: AnyTensor, alpha: Float64 = 0.01) raises -> AnyTensor:
    """Apply Leaky ReLU activation: max(alpha*x, x).

    Leaky ReLU introduces a small slope for negative values to prevent
    "dying ReLU" problem where neurons can become permanently inactive.

    Supported dtypes: float16, float32, float64, int8, int16, int32, int64.

    Args:
            tensor: Input tensor of any shape.
            alpha: Slope for negative values (default: 0.01).

    Returns:
            New tensor with Leaky ReLU applied element-wise.

    Raises:
            Error: If operation fails.

    Examples:
    ```mojo
            var x = AnyTensor(...)           # [-2, -1, 0, 1, 2]
            var y = leaky_relu(x, 0.01)     # [-0.02, -0.01, 0, 1, 2]
    ```
    """
    var result = AnyTensor(tensor._shape, tensor._dtype)

    if tensor._dtype == DType.float16:
        var alpha16 = Float16(alpha)
        for i in range(tensor._numel):
            var val = tensor._data.bitcast[Float16]()[i]
            result[i] = Float32(max(alpha16 * val, val))
    elif tensor._dtype == DType.float32:
        var alpha32 = Float32(alpha)
        for i in range(tensor._numel):
            var val = tensor._data.bitcast[Float32]()[i]
            result[i] = Float32(max(alpha32 * val, val))
    elif tensor._dtype == DType.float64:
        for i in range(tensor._numel):
            var val = tensor._data.bitcast[Float64]()[i]
            result[i] = Float32(max(alpha * val, val))
    elif tensor._dtype == DType.int8:
        var alpha_scaled = Int8(alpha * 128)  # Scale for fixed-point
        for i in range(tensor._numel):
            var val = tensor._data.bitcast[Int8]()[i]
            var scaled = (alpha_scaled * val) >> 7  # Divide by 128
            result[i] = Float32(scaled)
    elif tensor._dtype == DType.int16:
        var alpha_scaled = Int16(alpha * 32768)
        for i in range(tensor._numel):
            var val = tensor._data.bitcast[Int16]()[i]
            var scaled = (alpha_scaled * val) >> 15
            result[i] = Float32(scaled)
    elif tensor._dtype == DType.int32:
        var alpha32 = Float32(alpha)
        for i in range(tensor._numel):
            var val = tensor._data.bitcast[Int32]()[i]
            var scaled = Int32(alpha32 * Float32(val))
            result[i] = Float32(scaled)
    elif tensor._dtype == DType.int64:
        for i in range(tensor._numel):
            var val = tensor._data.bitcast[Int64]()[i]
            var scaled = Int64(alpha * Float64(val))
            result[i] = Float32(scaled)
    else:
        raise Error(
            "leaky_relu: unsupported dtype (use float16/32/64 or int8/16/32/64)"
        )

    return result


@always_inline
fn _prelu_impl[
    dtype: DType
](result: AnyTensor, tensor: AnyTensor, alpha: AnyTensor, is_scalar: Bool) raises:
    """Dtype-generic implementation of PReLU forward pass.

    Parameters:
            dtype: Compile-time dtype parameter.

    Args:
            result: Output tensor (pre-allocated with same shape as input).
            tensor: Input tensor.
            alpha: Learnable slope parameter (scalar or element-wise).
            is_scalar: Whether alpha is a scalar or element-wise.

    Note:
            This is an internal helper - use prelu() for the public API.
    """
    var data_ptr = tensor._data.bitcast[Scalar[dtype]]()
    var alpha_ptr = alpha._data.bitcast[Scalar[dtype]]()
    var result_ptr = result._data.bitcast[Scalar[dtype]]()

    for i in range(tensor._numel):
        var val = data_ptr[i]
        var a = alpha_ptr[0] if is_scalar else alpha_ptr[i]
        result_ptr[i] = max(a * val, val)


fn prelu(tensor: AnyTensor, alpha: AnyTensor) raises -> AnyTensor:
    """Apply PReLU (Parametric ReLU) activation: max(alpha*x, x).

            PReLU is similar to Leaky ReLU but uses learnable parameters for the
            negative slope. Alpha can be a scalar or have the same shape as tensor
            for per-element or per-channel learned slopes.

            Supported dtypes: float16, float32, float64.

    Args:
            tensor: Input tensor of any shape.
            alpha: Learnable slope parameter (scalar or matching shape).

    Returns:
            New tensor with PReLU applied element-wise.

    Raises:
            Error: If alpha shape is incompatible with tensor shape.

    Examples:
    ```
            var x = AnyTensor(...)      # [-2, -1, 0, 1, 2]
            var a = full(x.shape(), 0.25, DType.float32)
            var y = prelu(x, a)        # [-0.5, -0.25, 0, 1, 2]
    ```
    """
    # Validate alpha is scalar or compatible shape
    if alpha._numel != 1 and alpha._numel != tensor._numel:
        raise Error("prelu: alpha must be scalar or match tensor shape")

    if tensor._dtype != alpha._dtype:
        raise Error("prelu: tensor and alpha must have same dtype")

    var result = AnyTensor(tensor._shape, tensor._dtype)
    var is_scalar = alpha._numel == 1

    if tensor._dtype == DType.float16:
        _prelu_impl[DType.float16](result, tensor, alpha, is_scalar)
    elif tensor._dtype == DType.float32:
        _prelu_impl[DType.float32](result, tensor, alpha, is_scalar)
    elif tensor._dtype == DType.float64:
        _prelu_impl[DType.float64](result, tensor, alpha, is_scalar)
    else:
        raise Error(
            "prelu: only float16, float32, and float64 dtypes supported"
        )

    return result


# ============================================================================
# Sigmoid and Tanh (#243-247)
# ============================================================================


@always_inline
fn _sigmoid_op[T: DType](x: Scalar[T]) -> Scalar[T]:
    """Sigmoid operation with numerical stability: 1 / (1 + exp(-x)).

    Parameters:
        T: Data type of the scalar (float16, float32, float64).

    Args:
        x: Input scalar value.

    Returns:
        Sigmoid(x) = 1 / (1 + exp(-x)), with clipping for numerical stability.
    """
    # Numerically stable sigmoid with clipping
    if x > Scalar[T](SIGMOID_CLIP_THRESHOLD):
        return Scalar[T](1.0)
    elif x < Scalar[T](-SIGMOID_CLIP_THRESHOLD):
        return Scalar[T](0.0)
    else:

        @parameter
        if T == DType.float16:
            # Upcast to Float32 for computation, then cast back
            var x_f32 = Float32(x)
            var result_f32 = Float32(1.0) / (Float32(1.0) + exp(-x_f32))
            return Scalar[T](result_f32)
        else:
            return Scalar[T](1.0) / (Scalar[T](1.0) + exp(-x))


fn sigmoid(tensor: AnyTensor) raises -> AnyTensor:
    """Apply sigmoid activation: 1 / (1 + exp(-x)).

    Sigmoid maps inputs to (0, 1) range. Uses numerically stable implementation
    with input clipping to prevent overflow in exp() computation.

    For large |x| (>20), uses approximations:
    - `x > 20: sigmoid(x) = 1.0`
    - `x < -20: sigmoid(x) = 0.0`

    Supported dtypes: float16, float32, float64.

    This fused implementation handles float16 at tensor level instead of
    per-element conversion, reducing overhead.

    Args:
            tensor: Input tensor of any shape.

    Returns:
            New tensor with sigmoid applied element-wise, values in (0, 1).

    Raises:
            Error: If operation fails.

    Examples:
    ```
            var x = AnyTensor(...)  # [-2, 0, 2]
            var y = sigmoid(x)     # [0.119, 0.5, 0.881]
    ```
    """
    var result = zeros_like(tensor)
    var data_ptr = tensor._data
    var result_ptr = result._data
    var size = tensor.numel()

    if tensor.dtype() == DType.float32:
        for i in range(size):
            var x = data_ptr.bitcast[Float32]()[i]
            if x > Float32(SIGMOID_CLIP_THRESHOLD):
                result[i] = Float32(1.0)
            elif x < Float32(-SIGMOID_CLIP_THRESHOLD):
                result[i] = Float32(0.0)
            else:
                var exp_neg_x = exp_scalar_f32(-x)
                result[i] = Float32(1.0) / (Float32(1.0) + exp_neg_x)
    elif tensor.dtype() == DType.float64:
        for i in range(size):
            var x = data_ptr.bitcast[Float64]()[i]
            if x > Float64(SIGMOID_CLIP_THRESHOLD):
                result[i] = Float32(1.0)
            elif x < Float64(-SIGMOID_CLIP_THRESHOLD):
                result[i] = Float32(0.0)
            else:
                var exp_neg_x = exp_scalar_f64(-x)
                result[i] = Float32(Float32(1.0) / (Float32(1.0) + Float32(exp_neg_x)))
    elif tensor.dtype() == DType.float16:
        # Tensor-level handling: convert to Float32, compute, convert back
        for i in range(size):
            var x = Float32(data_ptr.bitcast[Float16]()[i])
            if x > Float32(SIGMOID_CLIP_THRESHOLD):
                result[i] = Float32(1.0)
            elif x < Float32(-SIGMOID_CLIP_THRESHOLD):
                result[i] = Float32(0.0)
            else:
                var exp_neg_x = exp_scalar_f32(-x)
                result[i] = Float32(1.0) / (Float32(1.0) + exp_neg_x)
    else:
        raise Error(
            "sigmoid only supports float16, float32, float64, got: "
            + String(tensor.dtype())
        )

    return result^


@always_inline
fn _tanh_op[T: DType](x: Scalar[T]) -> Scalar[T]:
    """Tanh operation for float dtypes.

    Parameters:
        T: Data type of the scalar (float16, float32, float64).

    Args:
        x: Input scalar value.

    Returns:
        tanh(x), computed using the math library function.
    """

    @parameter
    if T == DType.float16 or T == DType.float32:
        return Scalar[T](math_tanh(Float32(x)))
    else:  # float64
        return Scalar[T](math_tanh(Float64(x)))


fn tanh(tensor: AnyTensor) raises -> AnyTensor:
    """Apply tanh (hyperbolic tangent) activation.

        Tanh maps inputs to (-1, 1) range. This is a numerically stable
        implementation that leverages the math library tanh function.

        Supported dtypes: float16, float32, float64.

    Args:
            tensor: Input tensor of any shape.

    Returns:
            New tensor with tanh applied element-wise, values in (-1, 1).

    Raises:
            Error: If operation fails.

    Examples:
    ```
            var x = AnyTensor(...)  # [-2, 0, 2]
            var y = tanh(x)        # [-0.964, 0, 0.964]
    ```
    """
    return dispatch_float_unary[_tanh_op](tensor)


# ============================================================================
# Softmax and GELU (#248-252)
# ============================================================================


fn softmax(tensor: AnyTensor, axis: Int = -1) raises -> AnyTensor:
    """Apply softmax activation: exp(x) / sum(exp(x)) along specified axis.

        Softmax converts logits to probability distribution. Uses log-sum-exp trick
        for numerical stability by subtracting max value before exponentiation.

        Outputs sum to 1.0 along the specified axis.

        Supported dtypes: float16, float32, float64.

    Args:
            tensor: Input tensor (logits).
            axis: Axis along which to compute softmax (default: -1, last axis).
                    Supports negative indexing: -1 means last axis.

    Returns:
            New tensor with softmax applied, values sum to 1.0 along axis.

    Raises:
            Error: If axis is out of bounds.

    Examples:
    ```
            var logits = AnyTensor(...)  # [[1, 2, 3], [4, 5, 6]]
            var probs = softmax(logits, axis=-1)
            # [[0.09, 0.24, 0.67], [0.09, 0.24, 0.67]]
    ```
    """
    # Normalize axis
    var ndim = len(tensor._shape)
    var norm_axis = axis if axis >= 0 else ndim + axis

    if norm_axis < 0 or norm_axis >= ndim:
        raise Error("softmax: axis out of bounds")

    # Calculate stride for iterating along the softmax axis
    var axis_stride = 1
    for i in range(norm_axis + 1, ndim):
        axis_stride *= tensor._shape[i]

    # Calculate outer size (product of all dimensions before norm_axis)
    var outer_size = 1
    for i in range(norm_axis):
        outer_size *= tensor._shape[i]

    var axis_size = tensor._shape[norm_axis]

    return dispatch_softmax(tensor, outer_size, axis_size, axis_stride)


fn gelu(tensor: AnyTensor, approximate: Bool = False) raises -> AnyTensor:
    """Apply GELU (Gaussian Error Linear Unit) activation.

        GELU provides smooth, non-linear activation used in transformers (BERT, GPT).

        Exact formula: GELU(x) = x * Phi(x) = x * 0.5 * (1 + erf(x / sqrt(2))).
        Approximate formula: GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).

        The approximate version is faster and was used in the original BERT implementation.

        Supported dtypes: float16, float32, float64.

    Args:
            tensor: Input tensor of any shape.
            approximate: Use tanh approximation (True) or exact erf (False).

    Returns:
            New tensor with GELU applied element-wise.

    Raises:
            Error: If operation fails.

    Examples:
    ```
            var x = AnyTensor(...)     # [-2, 0, 2]
            var y_exact = gelu(x, approximate=False)
            var y_approx = gelu(x, approximate=True)
    ```
    """
    return dispatch_gelu(tensor, approximate)


# ============================================================================
# Backward Pass (Gradient Computation)
# ============================================================================


@always_inline
fn _relu_backward_op[T: DType](grad: Scalar[T], x: Scalar[T]) -> Scalar[T]:
    """ReLU backward: grad * (x > 0).

    Parameters:
        T: Data type of the scalars (float16, float32, float64, int8, etc.).

    Args:
        grad: Gradient of loss w.r.t. output (∂L/∂output).
        x: Input to forward pass (cached from forward).

    Returns:
        Gradient w.r.t. input (∂L/∂input) = grad if x > 0 else 0.
    """
    return grad if x > Scalar[T](0) else Scalar[T](0)


fn relu_backward(
    grad_output: AnyTensor, x: AnyTensor
) raises escaping -> AnyTensor:
    """Compute gradient of ReLU activation.

        ReLU gradient: `dL/dx = dL/dy * (x > 0)`

    Args:
            grad_output: Gradient from upstream (dL/dy).
            x: Input tensor from forward pass.

    Returns:
            Gradient with respect to input (dL/dx).

    Raises:
            Error: If operation fails.

    Examples:
    ```
            var x = AnyTensor(...)  # Input
            var y = relu(x)        # Forward pass
            var grad_y = AnyTensor(...)  # Gradient from loss
            var grad_x = relu_backward(grad_y, x)  # Backward pass
    ```
    """
    if grad_output._dtype != x._dtype:
        raise Error("relu_backward: grad_output and x must have same dtype")
    if grad_output._numel != x._numel:
        raise Error("relu_backward: grad_output and x must have same shape")

    return dispatch_binary[_relu_backward_op](grad_output, x)


@always_inline
fn _leaky_relu_backward_impl[
    dtype: DType
](result: AnyTensor, grad_output: AnyTensor, x: AnyTensor, alpha: Float64) raises:
    """Dtype-generic implementation of leaky ReLU backward pass."""
    var alpha_typed = Scalar[dtype](alpha)
    var result_ptr = result._data.bitcast[Scalar[dtype]]()
    var grad_ptr = grad_output._data.bitcast[Scalar[dtype]]()
    var x_ptr = x._data.bitcast[Scalar[dtype]]()

    for i in range(x._numel):
        var x_val = x_ptr[i]
        var grad = grad_ptr[i]
        result_ptr[i] = grad if x_val > Scalar[dtype](0) else grad * alpha_typed


fn leaky_relu_backward(
    grad_output: AnyTensor, x: AnyTensor, alpha: Float64 = 0.01
) raises escaping -> AnyTensor:
    """Compute gradient of Leaky ReLU activation.

        Leaky ReLU gradient: `dL/dx = dL/dy * (1 if x > 0 else alpha)`.

    Args:
            grad_output: Gradient from upstream (dL/dy).
            x: Input tensor from forward pass.
            alpha: Slope for negative values (default: 0.01).

    Returns:
            Gradient with respect to input (dL/dx).

    Raises:
            Error: If operation fails.
    """
    if grad_output._dtype != x._dtype:
        raise Error(
            "leaky_relu_backward: grad_output and x must have same dtype"
        )
    if grad_output._numel != x._numel:
        raise Error(
            "leaky_relu_backward: grad_output and x must have same shape"
        )

    var result = AnyTensor(x._shape, x._dtype)

    if x._dtype == DType.float16:
        _leaky_relu_backward_impl[DType.float16](result, grad_output, x, alpha)
    elif x._dtype == DType.float32:
        _leaky_relu_backward_impl[DType.float32](result, grad_output, x, alpha)
    elif x._dtype == DType.float64:
        _leaky_relu_backward_impl[DType.float64](result, grad_output, x, alpha)
    else:
        raise Error("leaky_relu_backward: only float16/32/64 dtypes supported")

    return result


@always_inline
fn _prelu_backward_impl[
    dtype: DType
](
    grad_input: AnyTensor,
    grad_alpha: AnyTensor,
    grad_output: AnyTensor,
    x: AnyTensor,
    alpha: AnyTensor,
    is_scalar: Bool,
) raises:
    """Dtype-generic implementation of PReLU backward pass."""
    var grad_input_ptr = grad_input._data.bitcast[Scalar[dtype]]()
    var grad_alpha_ptr = grad_alpha._data.bitcast[Scalar[dtype]]()
    var grad_ptr = grad_output._data.bitcast[Scalar[dtype]]()
    var x_ptr = x._data.bitcast[Scalar[dtype]]()
    var alpha_ptr = alpha._data.bitcast[Scalar[dtype]]()
    var zero = Scalar[dtype](0)

    # Initialize grad_alpha to zero
    for i in range(grad_alpha._numel):
        grad_alpha_ptr[i] = zero

    # Compute gradients
    for i in range(x._numel):
        var x_val = x_ptr[i]
        var grad = grad_ptr[i]
        var a = alpha_ptr[0] if is_scalar else alpha_ptr[i]

        if x_val > zero:
            grad_input_ptr[i] = grad
        else:
            grad_input_ptr[i] = grad * a
            var alpha_idx = 0 if is_scalar else i
            grad_alpha_ptr[alpha_idx] += grad * x_val


fn prelu_backward(
    grad_output: AnyTensor, x: AnyTensor, alpha: AnyTensor
) raises escaping -> GradientPair:
    """Compute gradients of PReLU activation.

        PReLU gradients:
        - `dL/dx = dL/dy * (1 if x > 0 else alpha)`
        - `dL/dalpha = sum(dL/dy * x) where x <= 0`

    Args:
            grad_output: Gradient from upstream (dL/dy).
            x: Input tensor from forward pass.
            alpha: Learnable slope parameter.

    Returns:
            GradientPair containing (grad_input, grad_alpha).

    Raises:
            Error: If operation fails.
    """
    if grad_output._dtype != x._dtype or grad_output._dtype != alpha._dtype:
        raise Error("prelu_backward: all tensors must have same dtype")
    if grad_output._numel != x._numel:
        raise Error("prelu_backward: grad_output and x must have same shape")

    var grad_input = AnyTensor(x._shape, x._dtype)
    var grad_alpha = AnyTensor(alpha._shape, alpha._dtype)
    var is_scalar = alpha._numel == 1

    if x._dtype == DType.float16:
        _prelu_backward_impl[DType.float16](
            grad_input, grad_alpha, grad_output, x, alpha, is_scalar
        )
    elif x._dtype == DType.float32:
        _prelu_backward_impl[DType.float32](
            grad_input, grad_alpha, grad_output, x, alpha, is_scalar
        )
    elif x._dtype == DType.float64:
        _prelu_backward_impl[DType.float64](
            grad_input, grad_alpha, grad_output, x, alpha, is_scalar
        )
    else:
        raise Error("prelu_backward: only float16/32/64 dtypes supported")

    return GradientPair(grad_input, grad_alpha)


@always_inline
fn _sigmoid_backward_op[T: DType](grad: Scalar[T], y: Scalar[T]) -> Scalar[T]:
    """Sigmoid backward: grad * y * (1 - y).

    Parameters:
        T: Data type of the scalars (float16, float32, float64).

    Args:
        grad: Gradient of loss w.r.t. output (∂L/∂output).
        y: Output from forward pass sigmoid(x).

    Returns:
        Gradient w.r.t. input = grad * y * (1 - y).
    """
    return grad * y * (Scalar[T](1.0) - y)


fn sigmoid_backward(
    grad_output: AnyTensor, output: AnyTensor
) raises escaping -> AnyTensor:
    """Compute gradient of sigmoid activation.

        Sigmoid gradient: `dL/dx = dL/dy * y * (1 - y)`
        where `y = sigmoid(x)`

    Args:
            grad_output: Gradient from upstream (dL/dy).
            output: Output from forward pass (sigmoid(x)).

    Returns:
            Gradient with respect to input (dL/dx).

    Raises:
            Error: If operation fails.

    Note:
            Takes output instead of input to avoid recomputing sigmoid.
    """
    if grad_output._dtype != output._dtype:
        raise Error(
            "sigmoid_backward: grad_output and output must have same dtype"
        )
    if grad_output._numel != output._numel:
        raise Error(
            "sigmoid_backward: grad_output and output must have same shape"
        )

    return dispatch_float_binary[_sigmoid_backward_op](grad_output, output)


@always_inline
fn _tanh_backward_op[T: DType](grad: Scalar[T], y: Scalar[T]) -> Scalar[T]:
    """Tanh backward: grad * (1 - y^2).

    Parameters:
        T: Data type of the scalars (float16, float32, float64).

    Args:
        grad: Gradient of loss w.r.t. output (∂L/∂output).
        y: Output from forward pass tanh(x).

    Returns:
        Gradient w.r.t. input = grad * (1 - y^2).
    """
    return grad * (Scalar[T](1.0) - y * y)


fn tanh_backward(
    grad_output: AnyTensor, output: AnyTensor
) raises escaping -> AnyTensor:
    """Compute gradient of tanh activation.

        Tanh gradient: `dL/dx = dL/dy * (1 - y^2)`
        where `y = tanh(x)`

    Args:
            grad_output: Gradient from upstream (dL/dy).
            output: Output from forward pass (tanh(x)).

    Returns:
            Gradient with respect to input (dL/dx).

    Raises:
            Error: If operation fails.

    Note:
            Takes output instead of input to avoid recomputing tanh.
    """
    if grad_output._dtype != output._dtype:
        raise Error(
            "tanh_backward: grad_output and output must have same dtype"
        )
    if grad_output._numel != output._numel:
        raise Error(
            "tanh_backward: grad_output and output must have same shape"
        )

    return dispatch_float_binary[_tanh_backward_op](grad_output, output)


fn gelu_backward(
    grad_output: AnyTensor, x: AnyTensor, approximate: Bool = False
) raises escaping -> AnyTensor:
    """Compute gradient of GELU activation.

        GELU gradient (exact): `dL/dx = dL/dy * [Phi(x) + x*phi(x)]`
        where Phi is CDF and phi is PDF of standard normal.

        GELU gradient (approximate): Uses derivative of tanh approximation.

    Args:
            grad_output: Gradient from upstream (dL/dy).
            x: Input tensor from forward pass.
            approximate: Use tanh approximation (True) or exact erf (False).

    Returns:
            Gradient with respect to input (dL/dx).

    Raises:
            Error: If operation fails.
    """
    if grad_output._dtype != x._dtype:
        raise Error("gelu_backward: grad_output and x must have same dtype")
    if grad_output._numel != x._numel:
        raise Error("gelu_backward: grad_output and x must have same shape")

    return dispatch_gelu_backward(grad_output, x, approximate)


fn softmax_backward(
    grad_output: AnyTensor, output: AnyTensor, axis: Int = -1
) raises escaping -> AnyTensor:
    """Compute gradient of softmax activation.

        Softmax gradient (along axis):
            `dL/dx_i = y_i * (dL/dy_i - sum_j(dL/dy_j * y_j))`

        where y = softmax(x) is the output from the forward pass.

        Supported dtypes: float16, float32, float64.

    Args:
            grad_output: Gradient from upstream (dL/dy).
            output: Softmax output from forward pass (y = softmax(x)).
            axis: Axis along which softmax was computed (default: -1).

    Returns:
            Gradient with respect to input (dL/dx).

    Raises:
        Error: If dtypes don't match or shapes incompatible.
    """
    if grad_output._dtype != output._dtype:
        raise Error(
            "softmax_backward: grad_output and output must have same dtype"
        )
    if grad_output._numel != output._numel:
        raise Error(
            "softmax_backward: grad_output and output must have same shape"
        )

    # Normalize axis to positive index
    var ndim = len(output._shape)
    var normalized_axis = axis if axis >= 0 else ndim + axis

    if normalized_axis < 0 or normalized_axis >= ndim:
        raise Error("softmax_backward: axis out of bounds")

    # Calculate stride for the softmax axis
    var axis_size = output._shape[normalized_axis]
    var axis_stride = 1
    for i in range(normalized_axis + 1, ndim):
        axis_stride *= output._shape[i]

    # Calculate outer size (product of dimensions before axis)
    var outer_size = 1
    for i in range(normalized_axis):
        outer_size *= output._shape[i]

    return dispatch_softmax_backward(
        grad_output, output, outer_size, axis_size, axis_stride
    )


# ============================================================================
# Advanced Activation Functions
# ============================================================================


fn swish(tensor: AnyTensor) raises -> AnyTensor:
    """Swish activation function (also known as SiLU - Sigmoid Linear Unit).

        Swish is a smooth, non-monotonic activation function that performs better
        than ReLU in deep networks.

        Formula:
            swish(x) = x * sigmoid(x).

    Args:
            tensor: Input tensor of any shape.

    Returns:
            Output tensor with swish applied element-wise.

    Raises:
            Error: If operation fails.

        Reference:
            Ramachandran et al., "Searching for Activation Functions" (2017).
    """
    # swish(x) = x * sigmoid(x)
    var sig = sigmoid(tensor)
    return multiply(tensor, sig)


fn softplus(tensor: AnyTensor, beta: Float64 = 1.0) raises -> AnyTensor:
    """Softplus activation function with fused kernel (7x allocation reduction).

    Computes `softplus(x) = (1/beta) * log(1 + exp(beta * x))` element-wise.
    Uses numerically stable formula: `max(0, x) + log(1 + exp(-|x|)) when beta=1`.

    This fused implementation allocates only 1 output tensor instead of 7
    intermediate tensors, providing significant memory and performance benefits.

    Args:
        tensor: Input tensor of any shape.
        beta: Sharpness parameter (default: 1.0). Higher values make it closer to ReLU.

    Returns:
        Output tensor with softplus applied element-wise.

    Performance:
        Before (unfused): 7 tensor allocations per call.
        After (fused): 1 tensor allocation per call (7x reduction).
    """
    var result = zeros_like(tensor)
    var data_ptr = tensor._data
    var result_ptr = result._data
    var size = tensor.numel()

    if tensor.dtype() == DType.float32:
        for i in range(size):
            var x = data_ptr.bitcast[Float32]()[i]
            # Numerically stable: max(0, x) + log(1 + exp(-|x|))
            var x_pos = max(x, Float32(0.0))
            var x_abs = abs(x)
            var exp_neg_abs = exp_scalar_f32(-x_abs)
            var log_term = math_log(Float64(1.0 + exp_neg_abs))
            result[i] = Float32(x_pos + Float32(log_term))
    elif tensor.dtype() == DType.float64:
        for i in range(size):
            var x = data_ptr.bitcast[Float64]()[i]
            var x_pos = max(x, Float64(0.0))
            var x_abs = abs(x)
            var exp_neg_abs = exp_scalar_f64(-x_abs)
            var log_term = math_log(Float64(1.0) + exp_neg_abs)
            result[i] = Float32(x_pos + log_term)
    elif tensor.dtype() == DType.float16:
        for i in range(size):
            var x = Float32(data_ptr.bitcast[Float16]()[i])
            var x_pos = max(x, Float32(0.0))
            var x_abs = abs(x)
            var exp_neg_abs = exp_scalar_f32(-x_abs)
            var log_term = math_log(Float64(1.0 + exp_neg_abs))
            result[i] = Float32(
                x_pos + Float32(log_term)
            )
    else:
        raise Error(
            "softplus only supports float16, float32, float64, got: "
            + String(tensor.dtype())
        )

    return result^


fn mish(tensor: AnyTensor) raises -> AnyTensor:
    """Mish activation function.

    Mish is a smooth, self-regularized non-monotonic activation function
    that has shown improvements over ReLU and Swish in some tasks.

    Formula:
        mish(x) = x * tanh(softplus(x)).
        where softplus(x) = log(1 + exp(x)).

    Args:
        tensor: Input tensor of any shape.

    Returns:
        Output tensor with mish applied element-wise.

    Raises:
            Error: If operation fails.

    Reference:
        Misra, "Mish: A Self Regularized Non-Monotonic Activation Function" (2019).
    """
    # Use fused softplus (1 allocation instead of 7)
    var sp = softplus(tensor)
    var tanh_softplus = tanh(sp)
    return multiply(tensor, tanh_softplus)


fn elu(tensor: AnyTensor, alpha: Float64 = 1.0) raises -> AnyTensor:
    """Exponential Linear Unit (ELU) activation function.

        ELU has negative values which pushes mean unit activations closer to zero,
        reducing bias shift and improving learning.

        Formula:
            `elu(x) = x if x > 0`
            `elu(x) = alpha * (exp(x) - 1) if x <= 0`

    Args:
            tensor: Input tensor of any shape.
            alpha: Scale for negative values (default: 1.0).

    Returns:
            Output tensor with ELU applied element-wise.

    Raises:
            Error: If operation fails.

        Reference:
            Clevert et al., "Fast and Accurate Deep Network Learning by.
            Exponential Linear Units (ELUs)" (2015).
    """
    var result = zeros_like(tensor)
    var data_ptr = tensor._data
    var result_ptr = result._data
    var size = tensor.numel()

    if tensor.dtype() == DType.float32:
        for i in range(size):
            var val = data_ptr.bitcast[Float32]()[i]
            if val > 0:
                result[i] = Float32(val)
            else:
                var val_clipped = max(val, Float32(-20.0))
                var exp_val = exp_scalar_f32(val_clipped)
                result[i] = Float32(alpha) * (exp_val - 1.0)
    elif tensor.dtype() == DType.float64:
        for i in range(size):
            var val = data_ptr.bitcast[Float64]()[i]
            if val > 0:
                result[i] = Float32(val)
            else:
                var val_clipped = max(val, Float64(-20.0))
                var exp_val = exp_scalar_f64(val_clipped)
                result[i] = Float32(alpha * (Float64(exp_val) - 1.0))
    elif tensor.dtype() == DType.float16:
        for i in range(size):
            var val = Float32(data_ptr.bitcast[Float16]()[i])
            if val > 0:
                result[i] = Float32(val)
            else:
                var val_clipped = max(val, Float32(-20.0))
                var exp_val = exp_scalar_f32(val_clipped)
                result[i] = Float32(
                    Float32(alpha) * (exp_val - 1.0)
                )
    else:
        raise Error("elu: only float16/32/64 dtypes supported")

    return result


fn selu(
    tensor: AnyTensor,
    alpha: Float64 = 1.6732632423543772848170429916717,
    lambda_: Float64 = 1.0507009873554804934193349852946,
) raises -> AnyTensor:
    """Scaled Exponential Linear Unit (SELU) activation function.

    SELU enables self-normalizing neural networks where activations
    converge to zero mean and unit variance during training.

    Formula:
        selu(x) = λ * (x if x > 0 else α * (exp(x) - 1))
        where α ≈ 1.6732632423543772848 (optimal scaling for SNNs)
        and λ ≈ 1.0507009873554804934 (variance normalization)

    Args:
        tensor: Input tensor of any shape.
        alpha: Scale for exponential branch (default: optimal value for SNNs).
        lambda_: Overall scale factor (default: optimal value for SNNs).

    Returns:
        Output tensor with SELU applied element-wise.

    Raises:
        Error: If operation fails.

    Reference:
        Klambauer et al., "Self-Normalizing Neural Networks" (2017).
    """
    var result = zeros_like(tensor)
    var data_ptr = tensor._data
    var result_ptr = result._data
    var size = tensor.numel()

    if tensor.dtype() == DType.float32:
        for i in range(size):
            var val = data_ptr.bitcast[Float32]()[i]
            if val > 0:
                result[i] = Float32(lambda_) * val
            else:
                var val_clipped = max(val, Float32(-20.0))
                var exp_val = exp_scalar_f32(val_clipped)
                result[i] = (
                    Float32(lambda_) * Float32(alpha) * (exp_val - 1.0)
                )
    elif tensor.dtype() == DType.float64:
        for i in range(size):
            var val = data_ptr.bitcast[Float64]()[i]
            if val > 0:
                result[i] = Float32(lambda_ * val)
            else:
                var val_clipped = max(val, Float64(-20.0))
                var exp_val = exp_scalar_f64(val_clipped)
                result[i] = Float32(
                    lambda_ * alpha * (exp_val - 1.0)
                )
    elif tensor.dtype() == DType.float16:
        for i in range(size):
            var val = Float32(data_ptr.bitcast[Float16]()[i])
            if val > 0:
                result[i] = Float32(
                    Float32(lambda_) * val
                )
            else:
                var val_clipped = max(val, Float32(-20.0))
                var exp_val = exp_scalar_f32(val_clipped)
                result.set(i, Float16(
                    Float32(lambda_) * Float32(alpha) * (exp_val - 1.0)
                ))
    else:
        raise Error("selu: only float16/32/64 dtypes supported")

    return result


# ============================================================================
# Backward Passes for Advanced Activations
# ============================================================================


fn selu_backward(
    grad_output: AnyTensor,
    x: AnyTensor,
    alpha: Float64 = 1.6732632423543772848170429916717,
    lambda_: Float64 = 1.0507009873554804934193349852946,
) raises escaping -> AnyTensor:
    """Backward pass for SELU activation.

    The derivative is:
        d/dx[selu(x)] = λ if x > 0.
        d/dx[selu(x)] = λ * α * exp(x) if x <= 0.

    Args:
        grad_output: Gradient from upstream.
        x: Input from forward pass.
        alpha: Scale for exponential branch (must match forward pass).
        lambda_: Overall scale factor (must match forward pass).

    Returns:
        Gradient with respect to input.

    Raises:
        Error: If operation fails.
    """
    var result = zeros_like(x)
    var x_ptr = x._data
    var grad_ptr = grad_output._data
    var result_ptr = result._data
    var size = x.numel()

    if x.dtype() == DType.float32:
        for i in range(size):
            var val = x_ptr.bitcast[Float32]()[i]
            var grad_val = grad_ptr.bitcast[Float32]()[i]

            if val > 0:
                result[i] = grad_val * Float32(lambda_)
            else:
                var val_clipped = max(val, Float32(-20.0))
                var exp_val = exp_scalar_f32(val_clipped)
                result[i] = (
                    grad_val * Float32(lambda_) * Float32(alpha) * exp_val
                )
    elif x.dtype() == DType.float64:
        for i in range(size):
            var val = x_ptr.bitcast[Float64]()[i]
            var grad_val = grad_ptr.bitcast[Float64]()[i]

            if val > 0:
                result.set(i, grad_val * lambda_)
            else:
                var val_clipped = max(val, Float64(-20.0))
                var exp_val = exp_scalar_f64(val_clipped)
                result.set(i, (
                    grad_val * lambda_ * alpha * exp_val
                ))
    elif x.dtype() == DType.float16:
        for i in range(size):
            var val = Float32(x_ptr.bitcast[Float16]()[i])
            var grad_val = Float32(grad_ptr.bitcast[Float16]()[i])

            if val > 0:
                result.set(i, Float16(
                    grad_val * Float32(lambda_)
                ))
            else:
                var val_clipped = max(val, Float32(-20.0))
                var exp_val = exp_scalar_f32(val_clipped)
                result.set(i, Float16(
                    grad_val * Float32(lambda_) * Float32(alpha) * exp_val
                ))
    else:
        raise Error("selu_backward: only float16/32/64 dtypes supported")

    return result


fn swish_backward(
    grad_output: AnyTensor, x: AnyTensor
) raises escaping -> AnyTensor:
    """Backward pass for Swish activation.

        The derivative of swish is:
            d/dx[swish(x)] = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)).
                           = sigmoid(x) * (1 + x * (1 - sigmoid(x))).

    Args:
            grad_output: Gradient from upstream.
            x: Input from forward pass.

    Returns:
            Gradient with respect to input.

    Raises:
            Error: If operation fails.
    """
    # Compute sigmoid(x)
    var sig = sigmoid(x)

    # Derivative: sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    var one_minus_sig = subtract(full(sig._shape, 1.0, sig._dtype), sig)
    var x_term = multiply(x, one_minus_sig)
    var one_plus_x_term = add(full(x_term._shape, 1.0, x_term._dtype), x_term)
    var derivative = multiply(sig, one_plus_x_term)

    # Apply chain rule
    return multiply(grad_output, derivative)


fn mish_backward(
    grad_output: AnyTensor, x: AnyTensor
) raises escaping -> AnyTensor:
    """Backward pass for Mish activation.

        The derivative involves the derivative of tanh(softplus(x)).

    Args:
            grad_output: Gradient from upstream.
            x: Input from forward pass.

    Returns:
            Gradient with respect to input.

    Raises:
            Error: If operation fails.
    """
    # Use fused softplus (1 allocation instead of 7)
    var sp = softplus(x)

    # Compute tanh(softplus(x))
    var tanh_sp = tanh(sp)

    # Compute sigmoid(x) for derivative
    var sig = sigmoid(x)

    # Derivative of mish:
    # = tanh(softplus(x)) + x * sech^2(softplus(x)) * sigmoid(x)
    # where sech^2(y) = 1 - tanh^2(y)
    var tanh_sq = multiply(tanh_sp, tanh_sp)
    var sech_sq = subtract(full(tanh_sq._shape, 1.0, tanh_sq._dtype), tanh_sq)
    var x_sech_sig = multiply(multiply(x, sech_sq), sig)
    var derivative = add(tanh_sp, x_sech_sig)

    # Apply chain rule
    return multiply(grad_output, derivative)


fn elu_backward(
    grad_output: AnyTensor, x: AnyTensor, alpha: Float64 = 1.0
) raises escaping -> AnyTensor:
    """Backward pass for ELU activation.

        The derivative is:
            d/dx[elu(x)] = 1 if x > 0.
            d/dx[elu(x)] = alpha * exp(x) if x <= 0.

    Args:
            grad_output: Gradient from upstream.
            x: Input from forward pass.
            alpha: Scale for negative values (must match forward pass).

    Returns:
            Gradient with respect to input.

    Raises:
            Error: If operation fails.
    """
    var result = zeros_like(x)
    var x_ptr = x._data
    var grad_ptr = grad_output._data
    var result_ptr = result._data
    var size = x.numel()

    if x.dtype() == DType.float32:
        for i in range(size):
            var val = x_ptr.bitcast[Float32]()[i]
            var grad_val = grad_ptr.bitcast[Float32]()[i]

            if val > 0:
                result[i] = Float32(grad_val)
            else:
                var val_clipped = max(val, Float32(-20.0))
                var exp_val = exp_scalar_f32(val_clipped)
                result[i] = (
                    grad_val * Float32(alpha) * exp_val
                )
    elif x.dtype() == DType.float64:
        for i in range(size):
            var val = x_ptr.bitcast[Float64]()[i]
            var grad_val = grad_ptr.bitcast[Float64]()[i]

            if val > 0:
                result[i] = Float32(grad_val)
            else:
                var val_clipped = max(val, Float64(-20.0))
                var exp_val = exp_scalar_f64(val_clipped)
                result[i] = Float32(grad_val * alpha * exp_val)
    elif x.dtype() == DType.float16:
        for i in range(size):
            var val = Float32(x_ptr.bitcast[Float16]()[i])
            var grad_val = Float32(grad_ptr.bitcast[Float16]()[i])

            if val > 0:
                result.set(i, Float16(grad_val))
            else:
                var val_clipped = max(val, Float32(-20.0))
                var exp_val = exp_scalar_f32(val_clipped)
                result.set(i, Float16(
                    grad_val * Float32(alpha) * exp_val
                ))
    else:
        raise Error("elu_backward: only float16/32/64 dtypes supported")

    return result


# ============================================================================
# Hard Activation Functions (Piecewise Linear Approximations)
# ============================================================================


fn hard_sigmoid(tensor: AnyTensor) raises -> AnyTensor:
    """Hard Sigmoid activation function.

        A piecewise linear approximation of sigmoid that is faster to compute.
        Commonly used in efficient architectures like MobileNet.

        Formula:
            hard_sigmoid(x) = clip((x + 3) / 6, 0, 1).
            = 0 if x <= -3.
            = 1 if x >= 3.
            = (x + 3) / 6 otherwise.

        Supported dtypes: float16, float32, float64.

    Args:
            tensor: Input tensor of any shape.

    Returns:
            Output tensor with hard_sigmoid applied element-wise, values in [0, 1].

    Raises:
            Error: If operation fails.

        Reference:
            Howard et al., "Searching for MobileNetV3" (2019).
    """
    return dispatch_hard_sigmoid(tensor)


fn hard_swish(tensor: AnyTensor) raises -> AnyTensor:
    """Hard Swish activation function.

        A piecewise linear approximation of Swish using hard_sigmoid.
        Used in MobileNetV3 for efficiency.

        Formula:
            hard_swish(x) = x * hard_sigmoid(x).
            = 0 if x <= -3.
            = x if x >= 3.
            = x * (x + 3) / 6 otherwise.

        Supported dtypes: float16, float32, float64.

    Args:
            tensor: Input tensor of any shape.

    Returns:
            Output tensor with hard_swish applied element-wise.

    Raises:
            Error: If operation fails.

        Reference:
            Howard et al., "Searching for MobileNetV3" (2019).
    """
    return dispatch_hard_swish(tensor)


fn hard_tanh(
    tensor: AnyTensor, min_val: Float64 = -1.0, max_val: Float64 = 1.0
) raises -> AnyTensor:
    """Hard Tanh activation function.

        A piecewise linear approximation of tanh that clips values to a range.

        Formula:
            hard_tanh(x) = clip(x, min_val, max_val).
            = min_val if x < min_val.
            = max_val if x > max_val.
            = x otherwise.

        Supported dtypes: float16, float32, float64.

    Args:
            tensor: Input tensor of any shape.
            min_val: Minimum output value (default: -1.0).
            max_val: Maximum output value (default: 1.0).

    Returns:
            Output tensor with hard_tanh applied element-wise.

    Raises:
            Error: If operation fails.

        Reference:
            Standard activation function used in various architectures.
    """
    return dispatch_hard_tanh(tensor, min_val, max_val)


# ============================================================================
# Hard Activation Backward Passes
# ============================================================================


fn hard_sigmoid_backward(
    grad_output: AnyTensor, x: AnyTensor
) raises escaping -> AnyTensor:
    """Backward pass for Hard Sigmoid activation.

        The derivative is:
            d/dx[hard_sigmoid(x)] = 1/6 if -3 < x < 3.
                                   = 0 otherwise.

    Args:
            grad_output: Gradient from upstream.
            x: Input from forward pass.

    Returns:
            Gradient with respect to input.

    Raises:
            Error: If operation fails.
    """
    if grad_output._dtype != x._dtype:
        raise Error(
            "hard_sigmoid_backward: grad_output and x must have same dtype"
        )
    if grad_output._numel != x._numel:
        raise Error(
            "hard_sigmoid_backward: grad_output and x must have same shape"
        )

    return dispatch_hard_sigmoid_backward(grad_output, x)


fn hard_swish_backward(
    grad_output: AnyTensor, x: AnyTensor
) raises escaping -> AnyTensor:
    """Backward pass for Hard Swish activation.

        The derivative is:
            d/dx[hard_swish(x)] = 0 if x <= -3.
                                = 1 if x >= 3.
                                = (2x + 3) / 6 otherwise.

        This comes from: d/dx[x * (x + 3) / 6] = (2x + 3) / 6.

    Args:
            grad_output: Gradient from upstream.
            x: Input from forward pass.

    Returns:
            Gradient with respect to input.

    Raises:
            Error: If operation fails.
    """
    if grad_output._dtype != x._dtype:
        raise Error(
            "hard_swish_backward: grad_output and x must have same dtype"
        )
    if grad_output._numel != x._numel:
        raise Error(
            "hard_swish_backward: grad_output and x must have same shape"
        )

    return dispatch_hard_swish_backward(grad_output, x)


fn hard_tanh_backward(
    grad_output: AnyTensor,
    x: AnyTensor,
    min_val: Float64 = -1.0,
    max_val: Float64 = 1.0,
) raises escaping -> AnyTensor:
    """Backward pass for Hard Tanh activation.

        The derivative is:
            d/dx[hard_tanh(x)] = 1 if min_val < x < max_val.
                               = 0 otherwise.

    Args:
            grad_output: Gradient from upstream.
            x: Input from forward pass.
            min_val: Minimum value used in forward pass (default: -1.0).
            max_val: Maximum value used in forward pass (default: 1.0).

    Returns:
            Gradient with respect to input.

    Raises:
            Error: If operation fails.
    """
    if grad_output._dtype != x._dtype:
        raise Error(
            "hard_tanh_backward: grad_output and x must have same dtype"
        )
    if grad_output._numel != x._numel:
        raise Error(
            "hard_tanh_backward: grad_output and x must have same shape"
        )

    return dispatch_hard_tanh_backward(grad_output, x, min_val, max_val)


# ============================================================================
# Typed Tensor[dtype] overloads — wrap AnyTensor versions via as_any/as_tensor
# ============================================================================


fn relu_typed[dt: DType](input: Tensor[dt]) raises -> Tensor[dt]:
    """Apply ReLU activation: max(0, x) (typed version).

    Args:
        input: Input typed tensor.

    Returns:
        A new Tensor[dt] with ReLU applied element-wise.
    """
    return relu(input.as_any()).as_tensor[dt]()


fn sigmoid_typed[dt: DType](input: Tensor[dt]) raises -> Tensor[dt]:
    """Apply sigmoid activation: 1 / (1 + exp(-x)) (typed version).

    Args:
        input: Input typed tensor.

    Returns:
        A new Tensor[dt] with sigmoid applied element-wise.
    """
    return sigmoid(input.as_any()).as_tensor[dt]()


fn softmax_typed[dt: DType](
    input: Tensor[dt], axis: Int = -1
) raises -> Tensor[dt]:
    """Apply softmax activation (typed version).

    Args:
        input: Input typed tensor.
        axis: Axis along which to apply softmax (default: -1).

    Returns:
        A new Tensor[dt] with softmax applied along the given axis.
    """
    return softmax(input.as_any(), axis).as_tensor[dt]()
