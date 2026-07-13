"""Typed Tensor[dtype] activation dispatch cores.

Internal module -- not part of the public API.
"""

from std.math import exp, erf, sqrt, tanh as math_tanh, log as math_log
from odyssey.tensor.tensor import Tensor
from odyssey.tensor.any_tensor import AnyTensor
from odyssey.core.activation_constants import (
    RELU6_UPPER_BOUND,
    SIGMOID_CLIP_THRESHOLD,
)
from odyssey.base.dtype_ordinal import (
    dtype_to_ordinal,
    DTYPE_FLOAT16,
    DTYPE_FLOAT32,
    DTYPE_FLOAT64,
    DTYPE_INT8,
    DTYPE_INT16,
    DTYPE_INT32,
    DTYPE_INT64,
    DTYPE_UINT8,
    DTYPE_UINT16,
    DTYPE_UINT32,
    DTYPE_UINT64,
)

# ============================================================================
# Layer 3 (Core): Native Tensor[dtype] Activation Implementations
# ============================================================================


def _relu_typed[dt: DType](input: Tensor[dt]) raises -> Tensor[dt]:
    """Native typed ReLU -- zero dtype branches.

    Tensor[dt]._data is already typed as UnsafePointer[Scalar[dt], MutAnyOrigin].

    Args:
        input: Input tensor (typed).

    Returns:
        Result Tensor[dt] with ReLU applied element-wise.
    """
    var result = Tensor[dt](input.shape())
    var size = input.numel()
    for i in range(size):
        result._data[i] = max(Scalar[dt](0), input._data[i])
    return result^


def _relu6_typed[dt: DType](input: Tensor[dt]) raises -> Tensor[dt]:
    """Native typed ReLU6 -- zero dtype branches.

    Args:
        input: Input tensor (typed).

    Returns:
        Result Tensor[dt] with ReLU6 applied element-wise.
    """
    var result = Tensor[dt](input.shape())
    var size = input.numel()
    for i in range(size):
        result._data[i] = min(
            max(Scalar[dt](0), input._data[i]), Scalar[dt](RELU6_UPPER_BOUND)
        )
    return result^


def _sigmoid_typed[dt: DType](input: Tensor[dt]) raises -> Tensor[dt]:
    """Native typed sigmoid -- zero dtype branches.

    Uses numerically stable implementation with input clipping.

    Args:
        input: Input tensor (typed).

    Returns:
        Result Tensor[dt] with sigmoid applied element-wise.
    """
    var result = Tensor[dt](input.shape())
    var size = input.numel()
    for i in range(size):
        var x = input._data[i]
        if x > Scalar[dt](SIGMOID_CLIP_THRESHOLD):
            result._data[i] = Scalar[dt](1.0)
        elif x < Scalar[dt](-SIGMOID_CLIP_THRESHOLD):
            result._data[i] = Scalar[dt](0.0)
        else:
            result._data[i] = Scalar[dt](1.0) / (
                Scalar[dt](1.0) + Scalar[dt](exp(-Float32(x)))
            )
    return result^


def _leaky_relu_typed[
    dt: DType
](input: Tensor[dt], alpha: Float64) raises -> Tensor[dt]:
    """Native typed leaky ReLU -- zero dtype branches.

    Args:
        input: Input tensor (typed).
        alpha: Slope for negative values.

    Returns:
        Result Tensor[dt] with leaky ReLU applied element-wise.
    """
    var result = Tensor[dt](input.shape())
    var size = input.numel()
    var alpha_typed = Scalar[dt](alpha)
    for i in range(size):
        var val = input._data[i]
        result._data[i] = max(alpha_typed * val, val)
    return result^


def _elu_typed[
    dt: DType
](input: Tensor[dt], alpha: Float64) raises -> Tensor[dt]:
    """Native typed ELU -- zero dtype branches.

    Args:
        input: Input tensor (typed).
        alpha: Scale for negative values.

    Returns:
        Result Tensor[dt] with ELU applied element-wise.
    """
    var result = Tensor[dt](input.shape())
    var size = input.numel()
    var alpha_typed = Scalar[dt](alpha)
    for i in range(size):
        var val = input._data[i]
        if val > Scalar[dt](0):
            result._data[i] = val
        else:
            var val_clipped = max(val, Scalar[dt](-20.0))
            result._data[i] = alpha_typed * (
                Scalar[dt](exp(Float32(val_clipped))) - Scalar[dt](1.0)
            )
    return result^


def _selu_typed[
    dt: DType
](input: Tensor[dt], alpha: Float64, lambda_: Float64) raises -> Tensor[dt]:
    """Native typed SELU -- zero dtype branches.

    Args:
        input: Input tensor (typed).
        alpha: Scale for exponential branch.
        lambda_: Overall scale factor.

    Returns:
        Result Tensor[dt] with SELU applied element-wise.
    """
    var result = Tensor[dt](input.shape())
    var size = input.numel()
    var alpha_typed = Scalar[dt](alpha)
    var lambda_typed = Scalar[dt](lambda_)
    for i in range(size):
        var val = input._data[i]
        if val > Scalar[dt](0):
            result._data[i] = lambda_typed * val
        else:
            var val_clipped = max(val, Scalar[dt](-20.0))
            result._data[i] = (
                lambda_typed
                * alpha_typed
                * (Scalar[dt](exp(Float32(val_clipped))) - Scalar[dt](1.0))
            )
    return result^


# ============================================================================
# Layer 2: AnyTensor Dispatch Helpers (delegates to typed activation cores)
# ============================================================================


def _dispatch_relu(tensor: AnyTensor) raises -> AnyTensor:
    """Runtime dispatch to Tensor[dtype] typed ReLU core (all dtypes)."""
    var ordinal = dtype_to_ordinal(tensor._dtype)
    if ordinal == DTYPE_FLOAT16:
        return _relu_typed[DType.float16](
            tensor.as_tensor[DType.float16]()
        ).as_any()
    elif ordinal == DTYPE_FLOAT32:
        return _relu_typed[DType.float32](
            tensor.as_tensor[DType.float32]()
        ).as_any()
    elif ordinal == DTYPE_FLOAT64:
        return _relu_typed[DType.float64](
            tensor.as_tensor[DType.float64]()
        ).as_any()
    elif ordinal == DTYPE_INT8:
        return _relu_typed[DType.int8](tensor.as_tensor[DType.int8]()).as_any()
    elif ordinal == DTYPE_INT16:
        return _relu_typed[DType.int16](
            tensor.as_tensor[DType.int16]()
        ).as_any()
    elif ordinal == DTYPE_INT32:
        return _relu_typed[DType.int32](
            tensor.as_tensor[DType.int32]()
        ).as_any()
    elif ordinal == DTYPE_INT64:
        return _relu_typed[DType.int64](
            tensor.as_tensor[DType.int64]()
        ).as_any()
    elif ordinal == DTYPE_UINT8:
        return _relu_typed[DType.uint8](
            tensor.as_tensor[DType.uint8]()
        ).as_any()
    elif ordinal == DTYPE_UINT16:
        return _relu_typed[DType.uint16](
            tensor.as_tensor[DType.uint16]()
        ).as_any()
    elif ordinal == DTYPE_UINT32:
        return _relu_typed[DType.uint32](
            tensor.as_tensor[DType.uint32]()
        ).as_any()
    elif ordinal == DTYPE_UINT64:
        return _relu_typed[DType.uint64](
            tensor.as_tensor[DType.uint64]()
        ).as_any()
    else:
        raise Error("relu: unsupported dtype")


def _dispatch_relu6(tensor: AnyTensor) raises -> AnyTensor:
    """Runtime dispatch to Tensor[dtype] typed ReLU6 core."""
    var ordinal = dtype_to_ordinal(tensor._dtype)
    if ordinal == DTYPE_FLOAT16:
        return _relu6_typed[DType.float16](
            tensor.as_tensor[DType.float16]()
        ).as_any()
    elif ordinal == DTYPE_FLOAT32:
        return _relu6_typed[DType.float32](
            tensor.as_tensor[DType.float32]()
        ).as_any()
    elif ordinal == DTYPE_FLOAT64:
        return _relu6_typed[DType.float64](
            tensor.as_tensor[DType.float64]()
        ).as_any()
    elif ordinal == DTYPE_INT8:
        return _relu6_typed[DType.int8](tensor.as_tensor[DType.int8]()).as_any()
    elif ordinal == DTYPE_INT16:
        return _relu6_typed[DType.int16](
            tensor.as_tensor[DType.int16]()
        ).as_any()
    elif ordinal == DTYPE_INT32:
        return _relu6_typed[DType.int32](
            tensor.as_tensor[DType.int32]()
        ).as_any()
    elif ordinal == DTYPE_INT64:
        return _relu6_typed[DType.int64](
            tensor.as_tensor[DType.int64]()
        ).as_any()
    else:
        raise Error("relu6: unsupported dtype")


def _dispatch_sigmoid(tensor: AnyTensor) raises -> AnyTensor:
    """Runtime dispatch to Tensor[dtype] typed sigmoid core (float only)."""
    var ordinal = dtype_to_ordinal(tensor._dtype)
    if ordinal == DTYPE_FLOAT16:
        return _sigmoid_typed[DType.float16](
            tensor.as_tensor[DType.float16]()
        ).as_any()
    elif ordinal == DTYPE_FLOAT32:
        return _sigmoid_typed[DType.float32](
            tensor.as_tensor[DType.float32]()
        ).as_any()
    elif ordinal == DTYPE_FLOAT64:
        return _sigmoid_typed[DType.float64](
            tensor.as_tensor[DType.float64]()
        ).as_any()
    else:
        raise Error(
            "sigmoid only supports float16, float32, float64, got: "
            + String(tensor._dtype)
        )


def _dispatch_leaky_relu(tensor: AnyTensor, alpha: Float64) raises -> AnyTensor:
    """Runtime dispatch to Tensor[dtype] typed leaky ReLU core."""
    var ordinal = dtype_to_ordinal(tensor._dtype)
    if ordinal == DTYPE_FLOAT16:
        return _leaky_relu_typed[DType.float16](
            tensor.as_tensor[DType.float16](), alpha
        ).as_any()
    elif ordinal == DTYPE_FLOAT32:
        return _leaky_relu_typed[DType.float32](
            tensor.as_tensor[DType.float32](), alpha
        ).as_any()
    elif ordinal == DTYPE_FLOAT64:
        return _leaky_relu_typed[DType.float64](
            tensor.as_tensor[DType.float64](), alpha
        ).as_any()
    elif ordinal == DTYPE_INT8:
        return _leaky_relu_typed[DType.int8](
            tensor.as_tensor[DType.int8](), alpha
        ).as_any()
    elif ordinal == DTYPE_INT16:
        return _leaky_relu_typed[DType.int16](
            tensor.as_tensor[DType.int16](), alpha
        ).as_any()
    elif ordinal == DTYPE_INT32:
        return _leaky_relu_typed[DType.int32](
            tensor.as_tensor[DType.int32](), alpha
        ).as_any()
    elif ordinal == DTYPE_INT64:
        return _leaky_relu_typed[DType.int64](
            tensor.as_tensor[DType.int64](), alpha
        ).as_any()
    else:
        raise Error(
            "leaky_relu: unsupported dtype (use float16/32/64 or int8/16/32/64)"
        )


def _dispatch_elu(tensor: AnyTensor, alpha: Float64) raises -> AnyTensor:
    """Runtime dispatch to Tensor[dtype] typed ELU core (float only)."""
    var ordinal = dtype_to_ordinal(tensor._dtype)
    if ordinal == DTYPE_FLOAT16:
        return _elu_typed[DType.float16](
            tensor.as_tensor[DType.float16](), alpha
        ).as_any()
    elif ordinal == DTYPE_FLOAT32:
        return _elu_typed[DType.float32](
            tensor.as_tensor[DType.float32](), alpha
        ).as_any()
    elif ordinal == DTYPE_FLOAT64:
        return _elu_typed[DType.float64](
            tensor.as_tensor[DType.float64](), alpha
        ).as_any()
    else:
        raise Error("elu: only float16/32/64 dtypes supported")


def _dispatch_selu(
    tensor: AnyTensor, alpha: Float64, lambda_: Float64
) raises -> AnyTensor:
    """Runtime dispatch to Tensor[dtype] typed SELU core (float only)."""
    var ordinal = dtype_to_ordinal(tensor._dtype)
    if ordinal == DTYPE_FLOAT16:
        return _selu_typed[DType.float16](
            tensor.as_tensor[DType.float16](), alpha, lambda_
        ).as_any()
    elif ordinal == DTYPE_FLOAT32:
        return _selu_typed[DType.float32](
            tensor.as_tensor[DType.float32](), alpha, lambda_
        ).as_any()
    elif ordinal == DTYPE_FLOAT64:
        return _selu_typed[DType.float64](
            tensor.as_tensor[DType.float64](), alpha, lambda_
        ).as_any()
    else:
        raise Error("selu: only float16/32/64 dtypes supported")


# ============================================================================
# ReLU Family (#238-242)
# ============================================================================


@always_inline
def _relu_op[T: DType](x: Scalar[T]) -> Scalar[T]:
    """ReLU operation: max(0, x).

    Parameters:
        T: Data type of the scalar (float16, float32, float64, int8, etc.).

    Args:
        x: Input scalar value.

    Returns:
        max(0, x).
    """
    return max(Scalar[T](0), x)


def _tanh_op[T: DType](x: Scalar[T]) -> Scalar[T]:
    """Tanh operation for float dtypes.

    Parameters:
        T: Data type of the scalar (float16, float32, float64).

    Args:
        x: Input scalar value.

    Returns:
        tanh(x), computed using the math library function.
    """
    comptime if T == DType.float16 or T == DType.float32:
        return Scalar[T](math_tanh(Float32(x)))
    else:  # float64
        return Scalar[T](math_tanh(Float64(x)))


def _tanh_typed[dt: DType](input: Tensor[dt]) raises -> Tensor[dt]:
    """Native typed tanh -- zero dtype branches.

    Args:
        input: Input tensor (typed).

    Returns:
        Result Tensor[dt] with tanh applied element-wise.
    """
    var result = Tensor[dt](input.shape())
    var size = input.numel()
    for i in range(size):
        result._data[i] = _tanh_op[dt](input._data[i])
    return result^
