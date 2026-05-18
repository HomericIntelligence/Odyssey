"""Element-wise mathematical operations with native Tensor[dtype] implementations.

Implements mathematical functions like exp, log, sqrt, trigonometric functions, etc.
Architecture: Tensor[dtype] typed implementations are the core (zero dtype branches).
AnyTensor versions dispatch to typed implementations via ordinal-based table.

Layer 1 (outer): AnyTensor public API (exp, log, sqrt, etc.)
Layer 2: dtype dispatch table (ordinal-based)
Layer 3 (core): Tensor[dtype] native implementation (_exp_typed, etc.)
"""

from std.collections import List
from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.core.dtype_dispatch import (
    dispatch_unary,
    dispatch_binary,
    dispatch_float_unary,
    dispatch_float_binary,
)
from projectodyssey.base.dtype_ordinal import (
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
from projectodyssey.base.broadcasting import (
    broadcast_shapes,
    compute_broadcast_strides,
)
from std.math import sqrt as math_sqrt
from std.math import exp as math_exp
from std.math import log as math_log
from std.math import sin as math_sin
from std.math import cos as math_cos
from std.math import tanh as math_tanh
from std.math import ceil as math_ceil
from std.math import floor as math_floor
from std.math import trunc as math_trunc
from std.memory import UnsafePointer


@always_inline
def math_round[T: DType](x: Scalar[T]) -> Scalar[T]:
    """Round to nearest integer (banker's rounding - round half to even).

    Examples:
            -2.5 → -2.0 (nearest even)
            -1.4 → -1.0
             0.5 →  0.0 (nearest even)
             1.4 →  1.0
             2.5 →  2.0 (nearest even)
    """
    var floor_val = math_floor(x)
    var frac = x - floor_val

    # If exactly 0.5, round to nearest even integer
    if frac == Scalar[T](0.5):
        # Check if floor_val is even
        var floor_int = Int(floor_val)
        if floor_int % 2 == 0:
            return floor_val  # Already even, round down
        else:
            return floor_val + Scalar[T](1.0)  # Round up to even
    # For -0.5, need to check ceiling
    elif frac == Scalar[T](-0.5):
        var ceil_val = math_ceil(x)
        var ceil_int = Int(ceil_val)
        if ceil_int % 2 == 0:
            return ceil_val  # Round up to even
        else:
            return floor_val  # Round down to even
    # Otherwise, standard rounding
    elif frac >= Scalar[T](0.5):
        return math_ceil(x)
    else:
        return floor_val


# ============================================================================
# Unary Operations (Element-wise)
# ============================================================================


@always_inline
def _neg_op[T: DType](x: Scalar[T]) -> Scalar[T]:
    """Negation operation."""
    return -x


def negate(tensor: AnyTensor) raises -> AnyTensor:
    """Negate all elements element-wise (-x).

    Args:
            tensor: Input tensor.

    Returns:
            A new tensor with negated values.

    Examples:
    ```
            var a = full(shape, 3.0, DType.float32)
            var b = negate(a)  # All values become -3.0
    ```
    """
    return dispatch_unary[_neg_op](tensor)


@always_inline
def _abs_op[T: DType](x: Scalar[T]) -> Scalar[T]:
    """Absolute value operation."""
    if x >= Scalar[T](0):
        return x
    else:
        return -x


def abs(tensor: AnyTensor) raises -> AnyTensor:
    """Absolute value element-wise.

    Args:
            tensor: Input tensor.

    Returns:
            A new tensor with absolute values.

    Examples:
    ```
            var a = full(shape, -3.0, DType.float32)
            var b = abs(a)  # All values become 3.0
    ```
    """
    from projectodyssey.tensor.typed.elementwise import _dispatch_unary_typed

    return _dispatch_unary_typed[_abs_op](tensor)


@always_inline
def _sign_op[T: DType](x: Scalar[T]) -> Scalar[T]:
    """Sign operation: -1, 0, or 1."""
    if x > Scalar[T](0):
        return Scalar[T](1)
    elif x < Scalar[T](0):
        return Scalar[T](-1)
    else:
        return Scalar[T](0)


def sign(tensor: AnyTensor) raises -> AnyTensor:
    """Sign function element-wise (-1, 0, or 1).

    Args:
            tensor: Input tensor.

    Returns:
            A new tensor with sign values (-1 for negative, 0 for zero, 1 for positive).

    Examples:
    ```
            var a = tensor([-2.0, 0.0, 3.0])
            var b = sign(a)  # [-1.0, 0.0, 1.0]
    ```
    """
    from projectodyssey.tensor.typed.elementwise import _dispatch_unary_typed

    return _dispatch_unary_typed[_sign_op](tensor)


@always_inline
def _exp_op[T: DType](x: Scalar[T]) -> Scalar[T]:
    """Exponential operation."""
    comptime if T == DType.float16 or T == DType.float32:
        return Scalar[T](math_exp(Float32(x)))
    else:
        return Scalar[T](math_exp(Float64(x)))


def exp(tensor: AnyTensor) raises -> AnyTensor:
    """Exponential function element-wise (e^x).

    Args:
            tensor: Input tensor.

    Returns:
            A new tensor with exponential values.

    Examples:
    ```
            var a = zeros(shape, DType.float32)
            var b = exp(a)  # All values become 1.0 (e^0)
    ```
    """
    from projectodyssey.tensor.typed.elementwise import (
        _dispatch_float_unary_typed,
    )

    return _dispatch_float_unary_typed[_exp_op](tensor)


@always_inline
def _log_op[T: DType](x: Scalar[T]) -> Scalar[T]:
    """Natural logarithm operation."""
    comptime if T == DType.float16 or T == DType.float32:
        return Scalar[T](math_log(Float32(x)))
    else:
        return Scalar[T](math_log(Float64(x)))


def log(tensor: AnyTensor) raises -> AnyTensor:
    """Natural logarithm element-wise (ln(x)).

    Args:
            tensor: Input tensor (must have positive values).

    Returns:
            A new tensor with logarithm values.

    Raises:
            Error: If any value is <= 0.

    Examples:
    ```
            var a = ones(shape, DType.float32)
            var b = log(a)  # All values become 0.0 (ln(1))
    ```
    """
    from projectodyssey.tensor.typed.elementwise import (
        _dispatch_float_unary_typed,
    )

    return _dispatch_float_unary_typed[_log_op](tensor)


@always_inline
def _sqrt_op[T: DType](x: Scalar[T]) -> Scalar[T]:
    """Square root operation."""
    comptime if T == DType.float16 or T == DType.float32:
        return Scalar[T](math_sqrt(Float32(x)))
    else:
        return Scalar[T](math_sqrt(Float64(x)))


def sqrt(tensor: AnyTensor) raises -> AnyTensor:
    """Square root element-wise.

    Args:
            tensor: Input tensor (must have non-negative values).

    Returns:
            A new tensor with square root values.

    Raises:
            Error: If any value is < 0.

    Examples:
    ```
            var a = full(shape, 4.0, DType.float32)
            var b = sqrt(a)  # All values become 2.0
    ```
    """
    from projectodyssey.tensor.typed.elementwise import (
        _dispatch_float_unary_typed,
    )

    return _dispatch_float_unary_typed[_sqrt_op](tensor)


@always_inline
def _sin_op[T: DType](x: Scalar[T]) -> Scalar[T]:
    """Sine operation."""
    comptime if T == DType.float16 or T == DType.float32:
        return Scalar[T](math_sin(Float32(x)))
    else:
        return Scalar[T](math_sin(Float64(x)))


def sin(tensor: AnyTensor) raises -> AnyTensor:
    """Sine function element-wise.

    Args:
            tensor: Input tensor (values in radians).

    Returns:
            A new tensor with sine values.

    Examples:
    ```
            var a = zeros(shape, DType.float32)
            var b = sin(a)  # All values become 0.0 (sin(0))
    ```
    """
    from projectodyssey.tensor.typed.elementwise import (
        _dispatch_float_unary_typed,
    )

    return _dispatch_float_unary_typed[_sin_op](tensor)


@always_inline
def _cos_op[T: DType](x: Scalar[T]) -> Scalar[T]:
    """Cosine operation."""
    comptime if T == DType.float16 or T == DType.float32:
        return Scalar[T](math_cos(Float32(x)))
    else:
        return Scalar[T](math_cos(Float64(x)))


def cos(tensor: AnyTensor) raises -> AnyTensor:
    """Cosine function element-wise.

    Args:
            tensor: Input tensor (values in radians).

    Returns:
            A new tensor with cosine values.

    Examples:
    ```
            var a = zeros(shape, DType.float32)
            var b = cos(a)  # All values become 1.0 (cos(0))
    ```
    """
    from projectodyssey.tensor.typed.elementwise import (
        _dispatch_float_unary_typed,
    )

    return _dispatch_float_unary_typed[_cos_op](tensor)


@always_inline
def _tanh_op[T: DType](x: Scalar[T]) -> Scalar[T]:
    """Hyperbolic tangent operation."""
    comptime if T == DType.float16 or T == DType.float32:
        return Scalar[T](math_tanh(Float32(x)))
    else:
        return Scalar[T](math_tanh(Float64(x)))


def tanh(tensor: AnyTensor) raises -> AnyTensor:
    """Hyperbolic tangent function element-wise.

    Args:
            tensor: Input tensor.

    Returns:
            A new tensor with tanh values (range: -1 to 1).

    Examples:
    ```
            var a = zeros(shape, DType.float32)
            var b = tanh(a)  # All values become 0.0 (tanh(0))
    ```
    """
    from projectodyssey.tensor.typed.elementwise import (
        _dispatch_float_unary_typed,
    )

    return _dispatch_float_unary_typed[_tanh_op](tensor)


# ============================================================================
# Dtype-specialized forward pass helpers
# ============================================================================


@always_inline
def _clip_forward_impl[
    dtype: DType
](result: AnyTensor, tensor: AnyTensor, min_val: Float64, max_val: Float64):
    """Dtype-specialized clip forward: clamp values to [min, max]."""
    var min_t = Scalar[dtype](min_val)
    var max_t = Scalar[dtype](max_val)
    var in_ptr = tensor._data.bitcast[Scalar[dtype]]()
    var out_ptr = result._data.bitcast[Scalar[dtype]]()
    for i in range(tensor.numel()):
        var val = in_ptr[i]
        if val < min_t:
            out_ptr[i] = min_t
        elif val > max_t:
            out_ptr[i] = max_t
        else:
            out_ptr[i] = val


def _dispatch_clip_forward(
    result: AnyTensor, tensor: AnyTensor, min_val: Float64, max_val: Float64
) raises:
    """Runtime dispatch for clip forward pass."""
    var dtype = tensor.dtype()
    if dtype == DType.float16:
        _clip_forward_impl[DType.float16](result, tensor, min_val, max_val)
    elif dtype == DType.float32:
        _clip_forward_impl[DType.float32](result, tensor, min_val, max_val)
    elif dtype == DType.float64:
        _clip_forward_impl[DType.float64](result, tensor, min_val, max_val)
    elif dtype == DType.int8:
        _clip_forward_impl[DType.int8](result, tensor, min_val, max_val)
    elif dtype == DType.int16:
        _clip_forward_impl[DType.int16](result, tensor, min_val, max_val)
    elif dtype == DType.int32:
        _clip_forward_impl[DType.int32](result, tensor, min_val, max_val)
    elif dtype == DType.int64:
        _clip_forward_impl[DType.int64](result, tensor, min_val, max_val)
    else:
        raise Error("clip: unsupported dtype")


@always_inline
def _log10_forward_impl[
    dtype: DType
](result: AnyTensor, tensor: AnyTensor, numel: Int) raises:
    """Dtype-specialized log10 forward: log(x) / log(10)."""
    comptime ln10 = Scalar[dtype](2.302585092994046)
    var in_ptr = tensor._data.bitcast[Scalar[dtype]]()
    var out_ptr = result._data.bitcast[Scalar[dtype]]()
    for i in range(numel):
        var val = in_ptr[i]
        if val <= Scalar[dtype](0):
            raise Error("log10 requires positive values")
        comptime if dtype == DType.float16 or dtype == DType.float32:
            out_ptr[i] = Scalar[dtype](math_log(Float32(val))) / ln10
        else:
            out_ptr[i] = Scalar[dtype](math_log(Float64(val))) / ln10


def _dispatch_log10_forward(
    result: AnyTensor, tensor: AnyTensor, numel: Int
) raises:
    """Runtime dispatch for log10 forward pass."""
    var dtype = tensor.dtype()
    if dtype == DType.float16:
        _log10_forward_impl[DType.float16](result, tensor, numel)
    elif dtype == DType.float32:
        _log10_forward_impl[DType.float32](result, tensor, numel)
    elif dtype == DType.float64:
        _log10_forward_impl[DType.float64](result, tensor, numel)
    else:
        raise Error("log10: unsupported dtype (requires float type)")


@always_inline
def _log2_forward_impl[
    dtype: DType
](result: AnyTensor, tensor: AnyTensor, numel: Int) raises:
    """Dtype-specialized log2 forward: log(x) / log(2)."""
    comptime ln2 = Scalar[dtype](0.6931471805599453)
    var in_ptr = tensor._data.bitcast[Scalar[dtype]]()
    var out_ptr = result._data.bitcast[Scalar[dtype]]()
    for i in range(numel):
        var val = in_ptr[i]
        if val <= Scalar[dtype](0):
            raise Error("log2 requires positive values")
        comptime if dtype == DType.float16 or dtype == DType.float32:
            out_ptr[i] = Scalar[dtype](math_log(Float32(val))) / ln2
        else:
            out_ptr[i] = Scalar[dtype](math_log(Float64(val))) / ln2


def _dispatch_log2_forward(
    result: AnyTensor, tensor: AnyTensor, numel: Int
) raises:
    """Runtime dispatch for log2 forward pass."""
    var dtype = tensor.dtype()
    if dtype == DType.float16:
        _log2_forward_impl[DType.float16](result, tensor, numel)
    elif dtype == DType.float32:
        _log2_forward_impl[DType.float32](result, tensor, numel)
    elif dtype == DType.float64:
        _log2_forward_impl[DType.float64](result, tensor, numel)
    else:
        raise Error("log2: unsupported dtype (requires float type)")


def _logical_and_impl[
    dtype: DType
](
    result: AnyTensor,
    a: AnyTensor,
    b: AnyTensor,
    strides_a: List[Int],
    strides_b: List[Int],
    result_shape: List[Int],
    total_elems: Int,
):
    """Dtype-specialized logical AND."""
    comptime zero = Scalar[dtype](0)
    comptime one = Scalar[dtype](1)
    var a_ptr = a._data.bitcast[Scalar[dtype]]()
    var b_ptr = b._data.bitcast[Scalar[dtype]]()
    var out_ptr = result._data.bitcast[Scalar[dtype]]()

    for result_idx in range(total_elems):
        # Convert result_idx to multi-dimensional index
        var remaining = result_idx
        var idx_a = 0
        var idx_b = 0
        for d in range(len(result_shape) - 1, -1, -1):
            var coord = remaining % result_shape[d]
            remaining //= result_shape[d]
            idx_a += coord * strides_a[d]
            idx_b += coord * strides_b[d]

        var val_a = a_ptr[idx_a]
        var val_b = b_ptr[idx_b]
        var bool_result = val_a != zero and val_b != zero
        out_ptr[result_idx] = one if bool_result else zero


def _dispatch_logical_and(
    result: AnyTensor,
    a: AnyTensor,
    b: AnyTensor,
    strides_a: List[Int],
    strides_b: List[Int],
    result_shape: List[Int],
    total_elems: Int,
) raises:
    """Runtime dispatch for logical AND."""
    var dtype = a.dtype()
    if dtype == DType.float16:
        _logical_and_impl[DType.float16](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    elif dtype == DType.float32:
        _logical_and_impl[DType.float32](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    elif dtype == DType.float64:
        _logical_and_impl[DType.float64](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    elif dtype == DType.int8:
        _logical_and_impl[DType.int8](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    elif dtype == DType.int16:
        _logical_and_impl[DType.int16](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    elif dtype == DType.int32:
        _logical_and_impl[DType.int32](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    elif dtype == DType.int64:
        _logical_and_impl[DType.int64](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    else:
        raise Error("logical_and: unsupported dtype")


def _logical_or_impl[
    dtype: DType
](
    result: AnyTensor,
    a: AnyTensor,
    b: AnyTensor,
    strides_a: List[Int],
    strides_b: List[Int],
    result_shape: List[Int],
    total_elems: Int,
):
    """Dtype-specialized logical OR."""
    comptime zero = Scalar[dtype](0)
    comptime one = Scalar[dtype](1)
    var a_ptr = a._data.bitcast[Scalar[dtype]]()
    var b_ptr = b._data.bitcast[Scalar[dtype]]()
    var out_ptr = result._data.bitcast[Scalar[dtype]]()

    for result_idx in range(total_elems):
        var remaining = result_idx
        var idx_a = 0
        var idx_b = 0
        for d in range(len(result_shape) - 1, -1, -1):
            var coord = remaining % result_shape[d]
            remaining //= result_shape[d]
            idx_a += coord * strides_a[d]
            idx_b += coord * strides_b[d]

        var val_a = a_ptr[idx_a]
        var val_b = b_ptr[idx_b]
        var bool_result = val_a != zero or val_b != zero
        out_ptr[result_idx] = one if bool_result else zero


def _dispatch_logical_or(
    result: AnyTensor,
    a: AnyTensor,
    b: AnyTensor,
    strides_a: List[Int],
    strides_b: List[Int],
    result_shape: List[Int],
    total_elems: Int,
) raises:
    """Runtime dispatch for logical OR."""
    var dtype = a.dtype()
    if dtype == DType.float16:
        _logical_or_impl[DType.float16](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    elif dtype == DType.float32:
        _logical_or_impl[DType.float32](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    elif dtype == DType.float64:
        _logical_or_impl[DType.float64](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    elif dtype == DType.int8:
        _logical_or_impl[DType.int8](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    elif dtype == DType.int16:
        _logical_or_impl[DType.int16](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    elif dtype == DType.int32:
        _logical_or_impl[DType.int32](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    elif dtype == DType.int64:
        _logical_or_impl[DType.int64](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    else:
        raise Error("logical_or: unsupported dtype")


@always_inline
def _logical_not_impl[
    dtype: DType
](result: AnyTensor, tensor: AnyTensor, numel: Int):
    """Dtype-specialized logical NOT."""
    comptime zero = Scalar[dtype](0)
    comptime one = Scalar[dtype](1)
    var in_ptr = tensor._data.bitcast[Scalar[dtype]]()
    var out_ptr = result._data.bitcast[Scalar[dtype]]()
    for i in range(numel):
        var bool_result = in_ptr[i] == zero
        out_ptr[i] = one if bool_result else zero


def _dispatch_logical_not(
    result: AnyTensor, tensor: AnyTensor, numel: Int
) raises:
    """Runtime dispatch for logical NOT."""
    var dtype = tensor.dtype()
    if dtype == DType.float16:
        _logical_not_impl[DType.float16](result, tensor, numel)
    elif dtype == DType.float32:
        _logical_not_impl[DType.float32](result, tensor, numel)
    elif dtype == DType.float64:
        _logical_not_impl[DType.float64](result, tensor, numel)
    elif dtype == DType.int8:
        _logical_not_impl[DType.int8](result, tensor, numel)
    elif dtype == DType.int16:
        _logical_not_impl[DType.int16](result, tensor, numel)
    elif dtype == DType.int32:
        _logical_not_impl[DType.int32](result, tensor, numel)
    elif dtype == DType.int64:
        _logical_not_impl[DType.int64](result, tensor, numel)
    else:
        raise Error("logical_not: unsupported dtype")


def _logical_xor_impl[
    dtype: DType
](
    result: AnyTensor,
    a: AnyTensor,
    b: AnyTensor,
    strides_a: List[Int],
    strides_b: List[Int],
    result_shape: List[Int],
    total_elems: Int,
):
    """Dtype-specialized logical XOR."""
    comptime zero = Scalar[dtype](0)
    comptime one = Scalar[dtype](1)
    var a_ptr = a._data.bitcast[Scalar[dtype]]()
    var b_ptr = b._data.bitcast[Scalar[dtype]]()
    var out_ptr = result._data.bitcast[Scalar[dtype]]()

    for result_idx in range(total_elems):
        var remaining = result_idx
        var idx_a = 0
        var idx_b = 0
        for d in range(len(result_shape) - 1, -1, -1):
            var coord = remaining % result_shape[d]
            remaining //= result_shape[d]
            idx_a += coord * strides_a[d]
            idx_b += coord * strides_b[d]

        var val_a = a_ptr[idx_a]
        var val_b = b_ptr[idx_b]
        var bool_a = val_a != zero
        var bool_b = val_b != zero
        # XOR: True if exactly one is True
        var bool_result = (bool_a and not bool_b) or (not bool_a and bool_b)
        out_ptr[result_idx] = one if bool_result else zero


def _dispatch_logical_xor(
    result: AnyTensor,
    a: AnyTensor,
    b: AnyTensor,
    strides_a: List[Int],
    strides_b: List[Int],
    result_shape: List[Int],
    total_elems: Int,
) raises:
    """Runtime dispatch for logical XOR."""
    var dtype = a.dtype()
    if dtype == DType.float16:
        _logical_xor_impl[DType.float16](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    elif dtype == DType.float32:
        _logical_xor_impl[DType.float32](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    elif dtype == DType.float64:
        _logical_xor_impl[DType.float64](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    elif dtype == DType.int8:
        _logical_xor_impl[DType.int8](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    elif dtype == DType.int16:
        _logical_xor_impl[DType.int16](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    elif dtype == DType.int32:
        _logical_xor_impl[DType.int32](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    elif dtype == DType.int64:
        _logical_xor_impl[DType.int64](
            result, a, b, strides_a, strides_b, result_shape, total_elems
        )
    else:
        raise Error("logical_xor: unsupported dtype")


def clip(
    tensor: AnyTensor, min_val: Float64, max_val: Float64
) raises -> AnyTensor:
    """Clip (clamp) values to a range element-wise.

    Args:
            tensor: Input tensor.
            min_val: Minimum value.
            max_val: Maximum value.

    Returns:
            A new tensor with clipped values.

    Raises:
            Error: If min_val > max_val.

    Examples:
    ```
            var a = tensor([-5.0, 0.0, 10.0])
            var b = clip(a, 0.0, 5.0)  # [0.0, 0.0, 5.0]
    ```
    """
    if min_val > max_val:
        raise Error("clip requires min_val <= max_val")

    var result = AnyTensor(tensor.shape(), tensor.dtype())
    _dispatch_clip_forward(result, tensor, min_val, max_val)
    return result^


# ============================================================================
# Rounding operations
# ============================================================================


@always_inline
def _ceil_op[T: DType](x: Scalar[T]) -> Scalar[T]:
    """Ceiling operation."""
    comptime if T == DType.float16 or T == DType.float32:
        return Scalar[T](math_ceil(Float32(x)))
    else:
        return Scalar[T](math_ceil(Float64(x)))


def ceil(tensor: AnyTensor) raises -> AnyTensor:
    """Ceiling function element-wise (round up to nearest integer).

    Args:
            tensor: Input tensor.

    Returns:
            A new tensor with ceiling values.

    Examples:
    ```
            var a = tensor([1.2, 2.5, 3.9])
            var b = ceil(a)  # [2.0, 3.0, 4.0]
    ```
    """
    from projectodyssey.tensor.typed.elementwise import (
        _dispatch_float_unary_typed,
    )

    return _dispatch_float_unary_typed[_ceil_op](tensor)


@always_inline
def _floor_op[T: DType](x: Scalar[T]) -> Scalar[T]:
    """Floor operation."""
    comptime if T == DType.float16 or T == DType.float32:
        return Scalar[T](math_floor(Float32(x)))
    else:
        return Scalar[T](math_floor(Float64(x)))


def floor(tensor: AnyTensor) raises -> AnyTensor:
    """Floor function element-wise (round down to nearest integer).

    Args:
            tensor: Input tensor.

    Returns:
            A new tensor with floor values.

    Examples:
    ```
            var a = tensor([1.2, 2.5, 3.9])
            var b = floor(a)  # [1.0, 2.0, 3.0]
    ```
    """
    from projectodyssey.tensor.typed.elementwise import (
        _dispatch_float_unary_typed,
    )

    return _dispatch_float_unary_typed[_floor_op](tensor)


@always_inline
def _round_op[T: DType](x: Scalar[T]) -> Scalar[T]:
    """Round operation."""
    comptime if T == DType.float16 or T == DType.float32:
        return Scalar[T](math_round(Float32(x)))
    else:
        return Scalar[T](math_round(Float64(x)))


def round(tensor: AnyTensor) raises -> AnyTensor:
    """Round to nearest integer element-wise.

    Args:
            tensor: Input tensor.

    Returns:
            A new tensor with rounded values.

    Examples:
    ```
            var a = tensor([1.2, 2.5, 3.9])
            var b = round(a)  # [1.0, 2.0, 4.0] (or [1.0, 3.0, 4.0] depending on rounding mode)
    ```
    """
    from projectodyssey.tensor.typed.elementwise import (
        _dispatch_float_unary_typed,
    )

    return _dispatch_float_unary_typed[_round_op](tensor)


@always_inline
def _trunc_op[T: DType](x: Scalar[T]) -> Scalar[T]:
    """Truncate operation."""
    comptime if T == DType.float16 or T == DType.float32:
        return Scalar[T](math_trunc(Float32(x)))
    else:
        return Scalar[T](math_trunc(Float64(x)))


def trunc(tensor: AnyTensor) raises -> AnyTensor:
    """Truncate to integer element-wise (round toward zero).

    Args:
            tensor: Input tensor.

    Returns:
            A new tensor with truncated values.

    Examples:
    ```
            var a = tensor([1.9, -2.9, 3.1])
            var b = trunc(a)  # [1.0, -2.0, 3.0]
    ```
    """
    from projectodyssey.tensor.typed.elementwise import (
        _dispatch_float_unary_typed,
    )

    return _dispatch_float_unary_typed[_trunc_op](tensor)


# ============================================================================
# Logical operations
# ============================================================================


def logical_and(a: AnyTensor, b: AnyTensor) raises -> AnyTensor:
    """Logical AND element-wise with broadcasting.

    Args:
            a: First input tensor.
            b: Second input tensor.

    Returns:
            Boolean tensor (True where both are non-zero).

    Raises:
            Error: If shapes are not broadcast-compatible or dtypes don't match.

        Broadcasting:
            Shapes are broadcast to a common output shape following NumPy rules.
            Dimensions are compatible if they are equal or one is 1.

    Examples:
    ```
            var a = tensor([0.0, 1.0, 2.0])
            var b = tensor([0.0, 0.0, 1.0])
            var c = logical_and(a, b)  # [False, False, True]

            # Broadcasting example
            var x = ones([3, 1, 5], DType.float32)
            var y = ones([3, 4, 5], DType.float32)
            var z = logical_and(x, y)  # Shape (3, 4, 5)
    ```
    """
    if a.dtype() != b.dtype():
        raise Error("logical_and: tensors must have same dtype")

    # Compute broadcast shape
    var result_shape = broadcast_shapes(a.shape(), b.shape())
    var result = AnyTensor(result_shape, a.dtype())

    # Compute broadcast strides
    var strides_a = compute_broadcast_strides(a.shape(), result_shape)
    var strides_b = compute_broadcast_strides(b.shape(), result_shape)

    # Calculate total elements in result
    var total_elems = 1
    for i in range(len(result_shape)):
        total_elems *= result_shape[i]

    _dispatch_logical_and(
        result, a, b, strides_a, strides_b, result_shape, total_elems
    )
    return result^


def logical_or(a: AnyTensor, b: AnyTensor) raises -> AnyTensor:
    """Logical OR element-wise with broadcasting.

    Args:
            a: First input tensor.
            b: Second input tensor.

    Returns:
            Boolean tensor (True where either is non-zero).

    Raises:
            Error: If shapes are not broadcast-compatible or dtypes don't match.

        Broadcasting:
            Shapes are broadcast to a common output shape following NumPy rules.
            Dimensions are compatible if they are equal or one is 1.

    Examples:
    ```
            var a = tensor([0.0, 1.0, 2.0])
            var b = tensor([0.0, 0.0, 1.0])
            var c = logical_or(a, b)  # [False, True, True]

            # Broadcasting example
            var x = ones([3, 1, 5], DType.float32)
            var y = ones([3, 4, 5], DType.float32)
            var z = logical_or(x, y)  # Shape (3, 4, 5)
    ```
    """
    if a.dtype() != b.dtype():
        raise Error("logical_or: tensors must have same dtype")

    # Compute broadcast shape
    var result_shape = broadcast_shapes(a.shape(), b.shape())
    var result = AnyTensor(result_shape, a.dtype())

    # Compute broadcast strides
    var strides_a = compute_broadcast_strides(a.shape(), result_shape)
    var strides_b = compute_broadcast_strides(b.shape(), result_shape)

    # Calculate total elements in result
    var total_elems = 1
    for i in range(len(result_shape)):
        total_elems *= result_shape[i]

    _dispatch_logical_or(
        result, a, b, strides_a, strides_b, result_shape, total_elems
    )
    return result^


def logical_not(tensor: AnyTensor) raises -> AnyTensor:
    """Logical NOT element-wise.

    Args:
            tensor: Input tensor.

    Returns:
            Boolean tensor (True where input is zero).

    Examples:
    ```
            var a = tensor([0.0, 1.0, 2.0])
            var b = logical_not(a)  # [True, False, False]
    ```
    """
    var result = AnyTensor(tensor.shape(), tensor.dtype())
    _dispatch_logical_not(result, tensor, tensor.numel())
    return result^


def logical_xor(a: AnyTensor, b: AnyTensor) raises -> AnyTensor:
    """Logical XOR element-wise with broadcasting.

    Args:
            a: First input tensor.
            b: Second input tensor.

    Returns:
            Boolean tensor (True where exactly one is non-zero).

    Raises:
            Error: If shapes are not broadcast-compatible or dtypes don't match.

        Broadcasting:
            Shapes are broadcast to a common output shape following NumPy rules.
            Dimensions are compatible if they are equal or one is 1.

    Examples:
    ```
            var a = tensor([0.0, 1.0, 0.0, 1.0])
            var b = tensor([0.0, 0.0, 1.0, 1.0])
            var c = logical_xor(a, b)  # [False, True, True, False]

            # Broadcasting example
            var x = ones([3, 1, 5], DType.float32)
            var y = ones([3, 4, 5], DType.float32)
            var z = logical_xor(x, y)  # Shape (3, 4, 5)
    ```
    """
    if a.dtype() != b.dtype():
        raise Error("logical_xor: tensors must have same dtype")

    # Compute broadcast shape
    var result_shape = broadcast_shapes(a.shape(), b.shape())
    var result = AnyTensor(result_shape, a.dtype())

    # Compute broadcast strides
    var strides_a = compute_broadcast_strides(a.shape(), result_shape)
    var strides_b = compute_broadcast_strides(b.shape(), result_shape)

    # Calculate total elements in result
    var total_elems = 1
    for i in range(len(result_shape)):
        total_elems *= result_shape[i]

    _dispatch_logical_xor(
        result, a, b, strides_a, strides_b, result_shape, total_elems
    )
    return result^


# ============================================================================
# Additional transcendental functions
# ============================================================================


def log10(tensor: AnyTensor) raises -> AnyTensor:
    """Base-10 logarithm element-wise.

    Args:
            tensor: Input tensor (must have positive values).

    Returns:
            A new tensor with log10 values.

    Raises:
            Error: If any value is <= 0.

    Examples:
    ```
            var a = tensor([1.0, 10.0, 100.0])
            var b = log10(a)  # [0.0, 1.0, 2.0]
    ```
    """
    var result = AnyTensor(tensor.shape(), tensor.dtype())
    _dispatch_log10_forward(result, tensor, tensor.numel())
    return result^


def log2(tensor: AnyTensor) raises -> AnyTensor:
    """Base-2 logarithm element-wise.

    Args:
            tensor: Input tensor (must have positive values).

    Returns:
            A new tensor with log2 values.

    Raises:
            Error: If any value is <= 0.

    Examples:
    ```
            var a = tensor([1.0, 2.0, 8.0])
            var b = log2(a)  # [0.0, 1.0, 3.0]
    ```
    """
    var result = AnyTensor(tensor.shape(), tensor.dtype())
    _dispatch_log2_forward(result, tensor, tensor.numel())
    return result^


# ============================================================================
# Backward Pass (Gradient Computation)
# ============================================================================

# ============================================================================
# Dtype-specialized backward pass helpers
# ============================================================================


@always_inline
def _exp_backward_impl[
    dtype: DType
](result: AnyTensor, grad_output: AnyTensor, x: AnyTensor, numel: Int) raises:
    """Dtype-specialized exp backward: grad * exp(x)."""
    var grad_ptr = grad_output._data.bitcast[Scalar[dtype]]()
    var x_ptr = x._data.bitcast[Scalar[dtype]]()
    var out_ptr = result._data.bitcast[Scalar[dtype]]()
    for i in range(numel):
        out_ptr[i] = grad_ptr[i] * Scalar[dtype](math_exp(Float32(x_ptr[i])))


def _dispatch_exp_backward(
    result: AnyTensor, grad_output: AnyTensor, x: AnyTensor, numel: Int
) raises:
    """Runtime dispatch for exp backward pass."""
    var dtype = grad_output.dtype()
    if dtype == DType.float16:
        _exp_backward_impl[DType.float16](result, grad_output, x, numel)
    elif dtype == DType.float32:
        _exp_backward_impl[DType.float32](result, grad_output, x, numel)
    elif dtype == DType.float64:
        _exp_backward_impl[DType.float64](result, grad_output, x, numel)
    else:
        raise Error("exp_backward: unsupported dtype")


@always_inline
def _log_backward_impl[
    dtype: DType
](result: AnyTensor, grad_output: AnyTensor, x: AnyTensor, numel: Int):
    """Dtype-specialized log backward: grad / (x + epsilon)."""
    comptime epsilon = Scalar[dtype](1e-10)
    var grad_ptr = grad_output._data.bitcast[Scalar[dtype]]()
    var x_ptr = x._data.bitcast[Scalar[dtype]]()
    var out_ptr = result._data.bitcast[Scalar[dtype]]()
    for i in range(numel):
        out_ptr[i] = grad_ptr[i] / (x_ptr[i] + epsilon)


def _dispatch_log_backward(
    result: AnyTensor, grad_output: AnyTensor, x: AnyTensor, numel: Int
) raises:
    """Runtime dispatch for log backward pass."""
    var dtype = grad_output.dtype()
    if dtype == DType.float16:
        _log_backward_impl[DType.float16](result, grad_output, x, numel)
    elif dtype == DType.float32:
        _log_backward_impl[DType.float32](result, grad_output, x, numel)
    elif dtype == DType.float64:
        _log_backward_impl[DType.float64](result, grad_output, x, numel)
    else:
        raise Error("log_backward: unsupported dtype")


@always_inline
def _sqrt_backward_impl[
    dtype: DType
](result: AnyTensor, grad_output: AnyTensor, x: AnyTensor, numel: Int):
    """Dtype-specialized sqrt backward: grad / (2 * sqrt(x) + epsilon)."""
    comptime epsilon = Scalar[dtype](1e-10)
    comptime two = Scalar[dtype](2.0)
    var grad_ptr = grad_output._data.bitcast[Scalar[dtype]]()
    var x_ptr = x._data.bitcast[Scalar[dtype]]()
    var out_ptr = result._data.bitcast[Scalar[dtype]]()
    for i in range(numel):
        out_ptr[i] = grad_ptr[i] / (two * math_sqrt(x_ptr[i]) + epsilon)


def _dispatch_sqrt_backward(
    result: AnyTensor, grad_output: AnyTensor, x: AnyTensor, numel: Int
) raises:
    """Runtime dispatch for sqrt backward pass."""
    var dtype = grad_output.dtype()
    if dtype == DType.float16:
        _sqrt_backward_impl[DType.float16](result, grad_output, x, numel)
    elif dtype == DType.float32:
        _sqrt_backward_impl[DType.float32](result, grad_output, x, numel)
    elif dtype == DType.float64:
        _sqrt_backward_impl[DType.float64](result, grad_output, x, numel)
    else:
        raise Error("sqrt_backward: unsupported dtype")


def _sin_backward_impl[
    dtype: DType
](result: AnyTensor, grad_output: AnyTensor, x: AnyTensor, numel: Int) raises:
    """Dtype-specialized sin backward: grad * cos(x)."""
    var grad_ptr = grad_output._data.bitcast[Scalar[dtype]]()
    var x_ptr = x._data.bitcast[Scalar[dtype]]()
    var out_ptr = result._data.bitcast[Scalar[dtype]]()
    for i in range(numel):
        out_ptr[i] = grad_ptr[i] * Scalar[dtype](math_cos(Float32(x_ptr[i])))


def _dispatch_sin_backward(
    result: AnyTensor, grad_output: AnyTensor, x: AnyTensor, numel: Int
) raises:
    """Runtime dispatch for sin backward pass."""
    var dtype = grad_output.dtype()
    if dtype == DType.float16:
        _sin_backward_impl[DType.float16](result, grad_output, x, numel)
    elif dtype == DType.float32:
        _sin_backward_impl[DType.float32](result, grad_output, x, numel)
    elif dtype == DType.float64:
        _sin_backward_impl[DType.float64](result, grad_output, x, numel)
    else:
        raise Error("sin_backward: unsupported dtype")


def _cos_backward_impl[
    dtype: DType
](result: AnyTensor, grad_output: AnyTensor, x: AnyTensor, numel: Int) raises:
    """Dtype-specialized cos backward: grad * (-sin(x))."""
    var grad_ptr = grad_output._data.bitcast[Scalar[dtype]]()
    var x_ptr = x._data.bitcast[Scalar[dtype]]()
    var out_ptr = result._data.bitcast[Scalar[dtype]]()
    for i in range(numel):
        out_ptr[i] = grad_ptr[i] * Scalar[dtype](-math_sin(Float32(x_ptr[i])))


def _dispatch_cos_backward(
    result: AnyTensor, grad_output: AnyTensor, x: AnyTensor, numel: Int
) raises:
    """Runtime dispatch for cos backward pass."""
    var dtype = grad_output.dtype()
    if dtype == DType.float16:
        _cos_backward_impl[DType.float16](result, grad_output, x, numel)
    elif dtype == DType.float32:
        _cos_backward_impl[DType.float32](result, grad_output, x, numel)
    elif dtype == DType.float64:
        _cos_backward_impl[DType.float64](result, grad_output, x, numel)
    else:
        raise Error("cos_backward: unsupported dtype")


@always_inline
def _abs_backward_impl[
    dtype: DType
](result: AnyTensor, grad_output: AnyTensor, x: AnyTensor, numel: Int):
    """Dtype-specialized abs backward: grad * sign(x)."""
    comptime zero = Scalar[dtype](0)
    comptime one = Scalar[dtype](1)
    comptime neg_one = Scalar[dtype](-1)
    var grad_ptr = grad_output._data.bitcast[Scalar[dtype]]()
    var x_ptr = x._data.bitcast[Scalar[dtype]]()
    var out_ptr = result._data.bitcast[Scalar[dtype]]()
    for i in range(numel):
        var x_val = x_ptr[i]
        var sign_x = zero
        if x_val > zero:
            sign_x = one
        elif x_val < zero:
            sign_x = neg_one
        out_ptr[i] = grad_ptr[i] * sign_x


def _dispatch_abs_backward(
    result: AnyTensor, grad_output: AnyTensor, x: AnyTensor, numel: Int
) raises:
    """Runtime dispatch for abs backward pass."""
    var dtype = grad_output.dtype()
    if dtype == DType.float16:
        _abs_backward_impl[DType.float16](result, grad_output, x, numel)
    elif dtype == DType.float32:
        _abs_backward_impl[DType.float32](result, grad_output, x, numel)
    elif dtype == DType.float64:
        _abs_backward_impl[DType.float64](result, grad_output, x, numel)
    elif dtype == DType.int8:
        _abs_backward_impl[DType.int8](result, grad_output, x, numel)
    elif dtype == DType.int16:
        _abs_backward_impl[DType.int16](result, grad_output, x, numel)
    elif dtype == DType.int32:
        _abs_backward_impl[DType.int32](result, grad_output, x, numel)
    elif dtype == DType.int64:
        _abs_backward_impl[DType.int64](result, grad_output, x, numel)
    else:
        raise Error("abs_backward: unsupported dtype")


@always_inline
def _clip_backward_impl[
    dtype: DType
](
    result: AnyTensor,
    grad_output: AnyTensor,
    x: AnyTensor,
    min_val: Float64,
    max_val: Float64,
    numel: Int,
):
    """Dtype-specialized clip backward: grad where min <= x <= max, else 0."""
    comptime zero = Scalar[dtype](0)
    var min_t = Scalar[dtype](min_val)
    var max_t = Scalar[dtype](max_val)
    var grad_ptr = grad_output._data.bitcast[Scalar[dtype]]()
    var x_ptr = x._data.bitcast[Scalar[dtype]]()
    var out_ptr = result._data.bitcast[Scalar[dtype]]()
    for i in range(numel):
        var x_val = x_ptr[i]
        if x_val >= min_t and x_val <= max_t:
            out_ptr[i] = grad_ptr[i]
        else:
            out_ptr[i] = zero


def _dispatch_clip_backward(
    result: AnyTensor,
    grad_output: AnyTensor,
    x: AnyTensor,
    min_val: Float64,
    max_val: Float64,
    numel: Int,
) raises:
    """Runtime dispatch for clip backward pass."""
    var dtype = grad_output.dtype()
    if dtype == DType.float16:
        _clip_backward_impl[DType.float16](
            result, grad_output, x, min_val, max_val, numel
        )
    elif dtype == DType.float32:
        _clip_backward_impl[DType.float32](
            result, grad_output, x, min_val, max_val, numel
        )
    elif dtype == DType.float64:
        _clip_backward_impl[DType.float64](
            result, grad_output, x, min_val, max_val, numel
        )
    else:
        raise Error("clip_backward: unsupported dtype")


@always_inline
def _log10_backward_impl[
    dtype: DType
](result: AnyTensor, grad_output: AnyTensor, x: AnyTensor, numel: Int):
    """Dtype-specialized log10 backward: grad / (x * ln(10) + epsilon)."""
    comptime epsilon = Scalar[dtype](1e-10)
    comptime ln10 = Scalar[dtype](2.302585092994046)
    var grad_ptr = grad_output._data.bitcast[Scalar[dtype]]()
    var x_ptr = x._data.bitcast[Scalar[dtype]]()
    var out_ptr = result._data.bitcast[Scalar[dtype]]()
    for i in range(numel):
        out_ptr[i] = grad_ptr[i] / (x_ptr[i] * ln10 + epsilon)


def _dispatch_log10_backward(
    result: AnyTensor, grad_output: AnyTensor, x: AnyTensor, numel: Int
) raises:
    """Runtime dispatch for log10 backward pass."""
    var dtype = grad_output.dtype()
    if dtype == DType.float16:
        _log10_backward_impl[DType.float16](result, grad_output, x, numel)
    elif dtype == DType.float32:
        _log10_backward_impl[DType.float32](result, grad_output, x, numel)
    elif dtype == DType.float64:
        _log10_backward_impl[DType.float64](result, grad_output, x, numel)
    else:
        raise Error("log10_backward: unsupported dtype")


@always_inline
def _log2_backward_impl[
    dtype: DType
](result: AnyTensor, grad_output: AnyTensor, x: AnyTensor, numel: Int):
    """Dtype-specialized log2 backward: grad / (x * ln(2) + epsilon)."""
    comptime epsilon = Scalar[dtype](1e-10)
    comptime ln2 = Scalar[dtype](0.6931471805599453)
    var grad_ptr = grad_output._data.bitcast[Scalar[dtype]]()
    var x_ptr = x._data.bitcast[Scalar[dtype]]()
    var out_ptr = result._data.bitcast[Scalar[dtype]]()
    for i in range(numel):
        out_ptr[i] = grad_ptr[i] / (x_ptr[i] * ln2 + epsilon)


def _dispatch_log2_backward(
    result: AnyTensor, grad_output: AnyTensor, x: AnyTensor, numel: Int
) raises:
    """Runtime dispatch for log2 backward pass."""
    var dtype = grad_output.dtype()
    if dtype == DType.float16:
        _log2_backward_impl[DType.float16](result, grad_output, x, numel)
    elif dtype == DType.float32:
        _log2_backward_impl[DType.float32](result, grad_output, x, numel)
    elif dtype == DType.float64:
        _log2_backward_impl[DType.float64](result, grad_output, x, numel)
    else:
        raise Error("log2_backward: unsupported dtype")


def exp_backward(grad_output: AnyTensor, x: AnyTensor) raises -> AnyTensor:
    """Compute gradient for exponential function.

        For Y = exp(X), given ∂L/∂Y, computes:
            ∂L/∂X = ∂L/∂Y * exp(X)

    Args:
            grad_output: Gradient from upstream (∂L/∂Y).
            x: Input from forward pass.

    Returns:
            Gradient w.r.t. input (∂L/∂X).

    Examples:
    ```
            var x = ones([3, 4])
            var grad_y = ones([3, 4])
            var grad_x = exp_backward(grad_y, x)  # grad_x = grad_y * exp(x)
    ```
    """
    var result = AnyTensor(grad_output.shape(), grad_output.dtype())
    _dispatch_exp_backward(result, grad_output, x, grad_output.numel())
    return result


def log_backward(grad_output: AnyTensor, x: AnyTensor) raises -> AnyTensor:
    """Compute gradient for natural logarithm.

        For Y = log(X), given ∂L/∂Y, computes:
            ∂L/∂X = ∂L/∂Y / X

        Includes numerical stability: adds epsilon to prevent division by zero.

    Args:
            grad_output: Gradient from upstream (∂L/∂Y).
            x: Input from forward pass (must be positive).

    Returns:
            Gradient w.r.t. input (∂L/∂X).

    Examples:
            var x = full([3, 4], 2.0)
            var y = log(x)
            var grad_y = ones([3, 4])
            var grad_x = log_backward(grad_y, x)  # grad_x = grad_y / x

        Numerical Stability:
            Uses epsilon = 1e-10 to prevent division by zero.
    """
    var result = AnyTensor(grad_output.shape(), grad_output.dtype())
    _dispatch_log_backward(result, grad_output, x, grad_output.numel())
    return result


def sqrt_backward(grad_output: AnyTensor, x: AnyTensor) raises -> AnyTensor:
    """Compute gradient for square root.

        For Y = sqrt(X), given ∂L/∂Y, computes:
            ∂L/∂X = ∂L/∂Y / (2 * sqrt(X))

    Args:
            grad_output: Gradient from upstream (∂L/∂Y).
            x: Input from forward pass.

    Returns:
            Gradient w.r.t. input (∂L/∂X).

    Examples:
            var x = full([3, 4], 4.0)
            var grad_y = ones([3, 4])
            var grad_x = sqrt_backward(grad_y, x)  # grad_x = grad_y / (2 * sqrt(4.0)) = 0.25

        Numerical Stability:
            Uses epsilon = 1e-10 to prevent division by zero when sqrt(X) ≈ 0.
    """
    var result = AnyTensor(grad_output.shape(), grad_output.dtype())
    _dispatch_sqrt_backward(result, grad_output, x, grad_output.numel())
    return result


def sin_backward(grad_output: AnyTensor, x: AnyTensor) raises -> AnyTensor:
    """Compute gradient for sine function.

        For Y = sin(X), given ∂L/∂Y, computes:
            ∂L/∂X = ∂L/∂Y * cos(X)

    Args:
            grad_output: Gradient from upstream (∂L/∂Y).
            x: Input from forward pass.

    Returns:
            Gradient w.r.t. input (∂L/∂X).

    Examples:
    ```
            var x = full([3, 4], pi/2)
            var grad_y = ones([3, 4])
            var grad_x = sin_backward(grad_y, x)  # grad_x = grad_y * cos(pi/2) ≈ 0
    ```
    """
    var result = AnyTensor(grad_output.shape(), grad_output.dtype())
    _dispatch_sin_backward(result, grad_output, x, grad_output.numel())
    return result


def cos_backward(grad_output: AnyTensor, x: AnyTensor) raises -> AnyTensor:
    """Compute gradient for cosine function.

        For Y = cos(X), given ∂L/∂Y, computes:
            ∂L/∂X = ∂L/∂Y * (-sin(X))

    Args:
            grad_output: Gradient from upstream (∂L/∂Y).
            x: Input from forward pass.

    Returns:
            Gradient w.r.t. input (∂L/∂X).

    Examples:
    ```
            var x = full([3, 4], 0.0)
            var grad_y = ones([3, 4])
            var grad_x = cos_backward(grad_y, x)  # grad_x = grad_y * (-sin(0)) = 0
    ```
    """
    var result = AnyTensor(grad_output.shape(), grad_output.dtype())
    _dispatch_cos_backward(result, grad_output, x, grad_output.numel())
    return result


def abs_backward(grad_output: AnyTensor, x: AnyTensor) raises -> AnyTensor:
    """Compute gradient for absolute value.

        For Y = |X|, given ∂L/∂Y, computes:
            ∂L/∂X = ∂L/∂Y * sign(X)

        where sign(X) = 1 if X > 0, -1 if X < 0, 0 if X = 0.

        Note: Gradient at X=0 is technically undefined, we use 0 by convention.

    Args:
            grad_output: Gradient from upstream (∂L/∂Y).
            x: Input from forward pass.

    Returns:
            Gradient w.r.t. input (∂L/∂X).

    Examples:
    ```
            var x = tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
            var y = abs(x)
            var grad_y = ones([5])
            var grad_x = abs_backward(grad_y, x)  # [-1, -1, 0, 1, 1]
    ```
    """
    var result = AnyTensor(grad_output.shape(), grad_output.dtype())
    _dispatch_abs_backward(result, grad_output, x, grad_output.numel())
    return result


def clip_backward(
    grad_output: AnyTensor, x: AnyTensor, min_val: Float64, max_val: Float64
) raises -> AnyTensor:
    """Compute gradient for clip (clamp) operation.

        For Y = clip(X, min, max), given ∂L/∂Y, computes:
            ∂L/∂X = ∂L/∂Y  if min <= X <= max
            ∂L/∂X = 0       if X < min or X > max

        Gradient flows through only where input is within bounds.

    Args:
            grad_output: Gradient from upstream (∂L/∂Y).
            x: Input from forward pass.
            min_val: Minimum clip value.
            max_val: Maximum clip value.

    Returns:
            Gradient w.r.t. input (∂L/∂X).

    Examples:
    ```
            var x = tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
            var y = clip(x, -1.0, 1.0)  # [-1, -1, 0, 1, 1]
            var grad_y = ones([5])
            var grad_x = clip_backward(grad_y, x, -1.0, 1.0)  # [0, 1, 1, 1, 0]
    ```
    """
    var result = AnyTensor(grad_output.shape(), grad_output.dtype())
    _dispatch_clip_backward(
        result, grad_output, x, min_val, max_val, grad_output.numel()
    )
    return result


def log10_backward(grad_output: AnyTensor, x: AnyTensor) raises -> AnyTensor:
    """Compute gradient for base-10 logarithm.

        For Y = log10(X), given ∂L/∂Y, computes:
            ∂L/∂X = ∂L/∂Y / (X * ln(10))

    Args:
            grad_output: Gradient from upstream (∂L/∂Y).
            x: Input from forward pass (must be positive).

    Returns:
            Gradient w.r.t. input (∂L/∂X).

        Numerical Stability:
            Uses epsilon = 1e-10 to prevent division by zero.
    """
    var result = AnyTensor(grad_output.shape(), grad_output.dtype())
    _dispatch_log10_backward(result, grad_output, x, grad_output.numel())
    return result


def log2_backward(grad_output: AnyTensor, x: AnyTensor) raises -> AnyTensor:
    """Compute gradient for base-2 logarithm.

        For Y = log2(X), given ∂L/∂Y, computes:
            ∂L/∂X = ∂L/∂Y / (X * ln(2))

    Args:
            grad_output: Gradient from upstream (∂L/∂Y).
            x: Input from forward pass (must be positive).

    Returns:
            Gradient w.r.t. input (∂L/∂X).

        Numerical Stability:
            Uses epsilon = 1e-10 to prevent division by zero.
    """
    var result = AnyTensor(grad_output.shape(), grad_output.dtype())
    _dispatch_log2_backward(result, grad_output, x, grad_output.numel())
    return result
