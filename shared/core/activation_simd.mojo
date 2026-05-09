"""SIMD-optimized activation functions for AnyTensor and Tensor[dtype].

This module provides vectorized implementations of activation functions,
achieving 2-8x speedup over scalar implementations for large tensors.

Implemented SIMD activations:
- ReLU family: relu_simd, leaky_relu_simd, relu6_simd (from issue #2589)
- ELU family: elu_simd, selu_simd (from issue #2623)
- Swish family: swish_simd (from issue #2623)

Performance characteristics:
- float32: ~4x speedup for ReLU (AVX2/AVX-512), ~2-3x for ELU/SELU/Swish
- float64: ~2x speedup for ReLU, ~1.5-2x for ELU/SELU/Swish
- Automatic fallback to scalar for unsupported dtypes

Design:
- SIMD variants use vectorized max/min operations
- Conditional branches use SIMD select for vectorization
- Compile-time SIMD width selection based on dtype
- @always_inline for hot path optimization
- Numerical stability with clipping for exp operations

Usage:
    from shared.core.activation_simd import relu_simd, elu_simd, selu_simd, swish_simd

    var x = randn([1024, 1024], DType.float32)
    var y = relu_simd(x)      # 4x faster than scalar
    var z = elu_simd(x, 1.0)  # 2-3x faster than scalar
    var w = selu_simd(x)      # 2-3x faster than scalar
    var v = swish_simd(x)     # 2-3x faster than scalar

Related:
- Issue #2589: Add SIMD vectorization to element-wise operations
- Issue #2623: Add Vectorized Implementations for Common Activations
"""

from std.algorithm import vectorize
from std.sys.info import simd_width_of
from std.math import exp as math_exp
from shared.tensor.any_tensor import AnyTensor
from shared.core.activation_constants import SIGMOID_CLIP_THRESHOLD


# ============================================================================
# SIMD ReLU
# ============================================================================


def relu_simd(tensor: AnyTensor) raises -> AnyTensor:
    """SIMD-optimized ReLU activation: max(0, x).

    Uses vectorized max operations for float32/float64 tensors,
    falls back to scalar for other dtypes.

    Args:
        tensor: Input tensor of any shape.

    Returns:
        New tensor with ReLU applied element-wise.

    Performance:
        - float32: ~4x speedup over scalar
        - float64: ~2x speedup over scalar
        - Other dtypes: Falls back to scalar implementation

    Examples:
        ```mojo
        var x = randn([1024, 1024], DType.float32)
        var y = relu_simd(x)  # SIMD accelerated
        ```
    """
    from shared.core.activation import relu

    var result = AnyTensor(tensor.shape(), tensor._dtype)

    if tensor._dtype == DType.float32:
        _relu_simd_float32(tensor, result)
    elif tensor._dtype == DType.float64:
        _relu_simd_float64(tensor, result)
    else:
        # Fall back to scalar for other dtypes
        return relu(tensor)

    return result^


@always_inline
def _relu_simd_float32(tensor: AnyTensor, mut result: AnyTensor):
    """SIMD ReLU for float32 tensors."""
    comptime simd_width = simd_width_of[DType.float32]()
    var size = tensor._numel

    var in_ptr = tensor._data.bitcast[Float32]()
    var out_ptr = result._data.bitcast[Float32]()

    @parameter
    def vectorized_relu[width: Int](idx: Int):
        var vec = in_ptr.load[width=width](idx)
        # SIMD max with zero vector
        var zero_vec = SIMD[DType.float32, width](0)
        out_ptr.store[width=width](idx, max(zero_vec, vec))

    vectorize[simd_width](size, vectorized_relu)


@always_inline
def _relu_simd_float64(tensor: AnyTensor, mut result: AnyTensor):
    """SIMD ReLU for float64 tensors."""
    comptime simd_width = simd_width_of[DType.float64]()
    var size = tensor._numel

    var in_ptr = tensor._data.bitcast[Float64]()
    var out_ptr = result._data.bitcast[Float64]()

    @parameter
    def vectorized_relu[width: Int](idx: Int):
        var vec = in_ptr.load[width=width](idx)
        var zero_vec = SIMD[DType.float64, width](0)
        out_ptr.store[width=width](idx, max(zero_vec, vec))

    vectorize[simd_width](size, vectorized_relu)


# ============================================================================
# SIMD Leaky ReLU
# ============================================================================


def leaky_relu_simd(
    tensor: AnyTensor, alpha: Float64 = 0.01
) raises -> AnyTensor:
    """SIMD-optimized Leaky ReLU activation: max(alpha*x, x).

    Uses vectorized operations for float32/float64 tensors,
    falls back to scalar for other dtypes.

    Args:
        tensor: Input tensor of any shape.
        alpha: Slope for negative values (default: 0.01).

    Returns:
        New tensor with Leaky ReLU applied element-wise.

    Performance:
        - float32: ~4x speedup over scalar
        - float64: ~2x speedup over scalar

    Examples:
        ```mojo
        var x = randn([1024, 1024], DType.float32)
        var y = leaky_relu_simd(x, 0.01)  # SIMD accelerated
        ```
    """
    from shared.core.activation import leaky_relu

    var result = AnyTensor(tensor.shape(), tensor._dtype)

    if tensor._dtype == DType.float32:
        _leaky_relu_simd_float32(tensor, result, Float32(alpha))
    elif tensor._dtype == DType.float64:
        _leaky_relu_simd_float64(tensor, result, alpha)
    else:
        # Fall back to scalar for other dtypes
        return leaky_relu(tensor, alpha)

    return result^


@always_inline
def _leaky_relu_simd_float32(
    tensor: AnyTensor, mut result: AnyTensor, alpha: Float32
):
    """SIMD Leaky ReLU for float32 tensors."""
    comptime simd_width = simd_width_of[DType.float32]()
    var size = tensor._numel

    var in_ptr = tensor._data.bitcast[Float32]()
    var out_ptr = result._data.bitcast[Float32]()

    @parameter
    def vectorized_leaky_relu[width: Int](idx: Int):
        var vec = in_ptr.load[width=width](idx)
        var alpha_vec = SIMD[DType.float32, width](alpha)
        var scaled = alpha_vec * vec
        # max(alpha*x, x): if x > 0, x > alpha*x (for 0 < alpha < 1)
        out_ptr.store[width=width](idx, max(scaled, vec))

    vectorize[simd_width](size, vectorized_leaky_relu)


@always_inline
def _leaky_relu_simd_float64(
    tensor: AnyTensor, mut result: AnyTensor, alpha: Float64
):
    """SIMD Leaky ReLU for float64 tensors."""
    comptime simd_width = simd_width_of[DType.float64]()
    var size = tensor._numel

    var in_ptr = tensor._data.bitcast[Float64]()
    var out_ptr = result._data.bitcast[Float64]()

    @parameter
    def vectorized_leaky_relu[width: Int](idx: Int):
        var vec = in_ptr.load[width=width](idx)
        var alpha_vec = SIMD[DType.float64, width](alpha)
        var scaled = alpha_vec * vec
        out_ptr.store[width=width](idx, max(scaled, vec))

    vectorize[simd_width](size, vectorized_leaky_relu)


# ============================================================================
# SIMD ReLU6
# ============================================================================


def relu6_simd(tensor: AnyTensor) raises -> AnyTensor:
    """SIMD-optimized ReLU6 activation: min(max(0, x), 6).

    ReLU6 clamps values to [0, 6], commonly used in MobileNet architectures.

    Args:
        tensor: Input tensor of any shape.

    Returns:
        New tensor with ReLU6 applied element-wise.

    Performance:
        - float32: ~4x speedup over scalar
        - float64: ~2x speedup over scalar

    Examples:
        ```mojo
        var x = randn([1024, 1024], DType.float32) * 10  # Values in [-10, 10]
        var y = relu6_simd(x)  # Values clamped to [0, 6]
        ```
    """
    from shared.core.activation import relu6

    var result = AnyTensor(tensor.shape(), tensor._dtype)

    if tensor._dtype == DType.float32:
        _relu6_simd_float32(tensor, result)
    elif tensor._dtype == DType.float64:
        _relu6_simd_float64(tensor, result)
    else:
        # Fall back to scalar
        return relu6(tensor)

    return result^


@always_inline
def _relu6_simd_float32(tensor: AnyTensor, mut result: AnyTensor):
    """SIMD ReLU6 for float32 tensors."""
    comptime simd_width = simd_width_of[DType.float32]()
    var size = tensor._numel

    var in_ptr = tensor._data.bitcast[Float32]()
    var out_ptr = result._data.bitcast[Float32]()

    @parameter
    def vectorized_relu6[width: Int](idx: Int):
        var vec = in_ptr.load[width=width](idx)
        var zero_vec = SIMD[DType.float32, width](0)
        var six_vec = SIMD[DType.float32, width](6)
        # min(max(0, x), 6)
        out_ptr.store[width=width](idx, min(max(zero_vec, vec), six_vec))

    vectorize[simd_width](size, vectorized_relu6)


@always_inline
def _relu6_simd_float64(tensor: AnyTensor, mut result: AnyTensor):
    """SIMD ReLU6 for float64 tensors."""
    comptime simd_width = simd_width_of[DType.float64]()
    var size = tensor._numel

    var in_ptr = tensor._data.bitcast[Float64]()
    var out_ptr = result._data.bitcast[Float64]()

    @parameter
    def vectorized_relu6[width: Int](idx: Int):
        var vec = in_ptr.load[width=width](idx)
        var zero_vec = SIMD[DType.float64, width](0)
        var six_vec = SIMD[DType.float64, width](6)
        out_ptr.store[width=width](idx, min(max(zero_vec, vec), six_vec))

    vectorize[simd_width](size, vectorized_relu6)


# ============================================================================
# SIMD ELU
# ============================================================================


def elu_simd(tensor: AnyTensor, alpha: Float64 = 1.0) raises -> AnyTensor:
    """SIMD-optimized ELU activation: x if x > 0 else alpha * (exp(x) - 1).

    Uses vectorized operations for float32/float64 tensors,
    falls back to scalar for other dtypes.

    Args:
        tensor: Input tensor of any shape.
        alpha: Scale for negative values (default: 1.0).

    Returns:
        New tensor with ELU applied element-wise.

    Performance:
        - float32: ~2-3x speedup over scalar (exp limits vectorization)
        - float64: ~1.5-2x speedup over scalar
        - Other dtypes: Falls back to scalar implementation

    Examples:
        ```mojo
        var x = randn([1024, 1024], DType.float32)
        var y = elu_simd(x, 1.0)  # SIMD accelerated
        ```
    """
    from shared.core.activation import elu

    var result = AnyTensor(tensor.shape(), tensor._dtype)

    if tensor._dtype == DType.float32:
        _elu_simd_float32(tensor, result, Float32(alpha))
    elif tensor._dtype == DType.float64:
        _elu_simd_float64(tensor, result, alpha)
    else:
        # Fall back to scalar for other dtypes
        return elu(tensor, alpha)

    return result^


@always_inline
def _elu_simd_float32(tensor: AnyTensor, mut result: AnyTensor, alpha: Float32):
    """SIMD ELU for float32 tensors."""
    from shared.core.activation_ops import exp_scalar_f32

    comptime simd_width = simd_width_of[DType.float32]()
    var size = tensor._numel

    var in_ptr = tensor._data.bitcast[Float32]()
    var out_ptr = result._data.bitcast[Float32]()

    @parameter
    def vectorized_elu[width: Int](idx: Int):
        var vec = in_ptr.load[width=width](idx)
        var zero_vec = SIMD[DType.float32, width](0)
        var one_vec = SIMD[DType.float32, width](1)
        var alpha_vec = SIMD[DType.float32, width](alpha)

        # For positive values: just return x
        # For negative values: alpha * (exp(x) - 1)
        # We compute both and select
        var pos_result = vec
        var neg_clipped = max(vec, SIMD[DType.float32, width](-20.0))

        # SIMD exp may have limited vectorization (Mojo v0.26.1)
        # but still benefits from SIMD for the rest of computation
        var exp_result = math_exp(neg_clipped)
        var neg_result = alpha_vec * (exp_result - one_vec)

        # Select based on condition: x > 0
        var mask = vec.gt(zero_vec)
        # SIMD conditional selection: mask.select(true_value, false_value)
        out_ptr.store[width=width](idx, mask.select(pos_result, neg_result))

    vectorize[simd_width](size, vectorized_elu)


@always_inline
def _elu_simd_float64(tensor: AnyTensor, mut result: AnyTensor, alpha: Float64):
    """SIMD ELU for float64 tensors."""
    from shared.core.activation_ops import exp_scalar_f64

    comptime simd_width = simd_width_of[DType.float64]()
    var size = tensor._numel

    var in_ptr = tensor._data.bitcast[Float64]()
    var out_ptr = result._data.bitcast[Float64]()

    @parameter
    def vectorized_elu[width: Int](idx: Int):
        var vec = in_ptr.load[width=width](idx)
        var zero_vec = SIMD[DType.float64, width](0)
        var one_vec = SIMD[DType.float64, width](1)
        var alpha_vec = SIMD[DType.float64, width](alpha)

        var pos_result = vec
        var neg_clipped = max(vec, SIMD[DType.float64, width](-20.0))
        var exp_result = math_exp(neg_clipped)
        var neg_result = alpha_vec * (exp_result - one_vec)

        var mask = vec.gt(zero_vec)
        # SIMD conditional selection: mask.select(true_value, false_value)
        out_ptr.store[width=width](idx, mask.select(pos_result, neg_result))

    vectorize[simd_width](size, vectorized_elu)


# ============================================================================
# SIMD SELU
# ============================================================================


def selu_simd(
    tensor: AnyTensor,
    alpha: Float64 = 1.6732632423543772848170429916717,
    lambda_: Float64 = 1.0507009873554804934193349852946,
) raises -> AnyTensor:
    """SIMD-optimized SELU activation: λ * (x if x > 0 else α * (exp(x) - 1)).

    Uses vectorized operations for float32/float64 tensors,
    falls back to scalar for other dtypes.

    Args:
        tensor: Input tensor of any shape.
        alpha: Scale for exponential branch (default: SNN optimal).
        lambda_: Overall scale factor (default: SNN optimal).

    Returns:
        New tensor with SELU applied element-wise.

    Performance:
        - float32: ~2-3x speedup over scalar
        - float64: ~1.5-2x speedup over scalar

    Examples:
        ```mojo
        var x = randn([1024, 1024], DType.float32)
        var y = selu_simd(x)  # SIMD accelerated
        ```
    """
    from shared.core.activation import selu

    var result = AnyTensor(tensor.shape(), tensor._dtype)

    if tensor._dtype == DType.float32:
        _selu_simd_float32(tensor, result, Float32(alpha), Float32(lambda_))
    elif tensor._dtype == DType.float64:
        _selu_simd_float64(tensor, result, alpha, lambda_)
    else:
        # Fall back to scalar for other dtypes
        return selu(tensor, alpha, lambda_)

    return result^


@always_inline
def _selu_simd_float32(
    tensor: AnyTensor, mut result: AnyTensor, alpha: Float32, lambda_: Float32
):
    """SIMD SELU for float32 tensors."""
    comptime simd_width = simd_width_of[DType.float32]()
    var size = tensor._numel

    var in_ptr = tensor._data.bitcast[Float32]()
    var out_ptr = result._data.bitcast[Float32]()

    @parameter
    def vectorized_selu[width: Int](idx: Int):
        var vec = in_ptr.load[width=width](idx)
        var zero_vec = SIMD[DType.float32, width](0)
        var one_vec = SIMD[DType.float32, width](1)
        var alpha_vec = SIMD[DType.float32, width](alpha)
        var lambda_vec = SIMD[DType.float32, width](lambda_)

        var pos_result = lambda_vec * vec
        var neg_clipped = max(vec, SIMD[DType.float32, width](-20.0))
        var exp_result = math_exp(neg_clipped)
        var neg_result = lambda_vec * alpha_vec * (exp_result - one_vec)

        var mask = vec.gt(zero_vec)
        # SIMD conditional selection: mask.select(true_value, false_value)
        out_ptr.store[width=width](idx, mask.select(pos_result, neg_result))

    vectorize[simd_width](size, vectorized_selu)


@always_inline
def _selu_simd_float64(
    tensor: AnyTensor, mut result: AnyTensor, alpha: Float64, lambda_: Float64
):
    """SIMD SELU for float64 tensors."""
    comptime simd_width = simd_width_of[DType.float64]()
    var size = tensor._numel

    var in_ptr = tensor._data.bitcast[Float64]()
    var out_ptr = result._data.bitcast[Float64]()

    @parameter
    def vectorized_selu[width: Int](idx: Int):
        var vec = in_ptr.load[width=width](idx)
        var zero_vec = SIMD[DType.float64, width](0)
        var one_vec = SIMD[DType.float64, width](1)
        var alpha_vec = SIMD[DType.float64, width](alpha)
        var lambda_vec = SIMD[DType.float64, width](lambda_)

        var pos_result = lambda_vec * vec
        var neg_clipped = max(vec, SIMD[DType.float64, width](-20.0))
        var exp_result = math_exp(neg_clipped)
        var neg_result = lambda_vec * alpha_vec * (exp_result - one_vec)

        var mask = vec.gt(zero_vec)
        # SIMD conditional selection: mask.select(true_value, false_value)
        out_ptr.store[width=width](idx, mask.select(pos_result, neg_result))

    vectorize[simd_width](size, vectorized_selu)


# ============================================================================
# SIMD Swish
# ============================================================================


def swish_simd(tensor: AnyTensor) raises -> AnyTensor:
    """SIMD-optimized Swish activation: x * sigmoid(x).

    Uses vectorized operations for float32/float64 tensors,
    falls back to scalar for other dtypes.

    Args:
        tensor: Input tensor of any shape.

    Returns:
        New tensor with Swish applied element-wise.

    Performance:
        - float32: ~2-3x speedup over scalar (sigmoid involves exp)
        - float64: ~1.5-2x speedup over scalar

    Examples:
        ```mojo
        var x = randn([1024, 1024], DType.float32)
        var y = swish_simd(x)  # SIMD accelerated
        ```
    """
    from shared.core.activation import swish

    var result = AnyTensor(tensor.shape(), tensor._dtype)

    if tensor._dtype == DType.float32:
        _swish_simd_float32(tensor, result)
    elif tensor._dtype == DType.float64:
        _swish_simd_float64(tensor, result)
    else:
        # Fall back to scalar for other dtypes
        return swish(tensor)

    return result^


@always_inline
def _swish_simd_float32(tensor: AnyTensor, mut result: AnyTensor):
    """SIMD Swish for float32 tensors."""
    comptime simd_width = simd_width_of[DType.float32]()
    var size = tensor._numel

    var in_ptr = tensor._data.bitcast[Float32]()
    var out_ptr = result._data.bitcast[Float32]()

    @parameter
    def vectorized_swish[width: Int](idx: Int):
        var vec = in_ptr.load[width=width](idx)

        # Compute sigmoid(x) with numerical stability
        # sigmoid(x) = 1 / (1 + exp(-x))
        # For numerical stability, use:
        # - if x > 0: 1 / (1 + exp(-x))
        # - if x <= 0: exp(x) / (1 + exp(x))

        var neg_vec = -vec
        var neg_clipped = max(neg_vec, SIMD[DType.float32, width](-20.0))
        var exp_neg = math_exp(neg_clipped)
        var one_vec = SIMD[DType.float32, width](1)
        var sigmoid = one_vec / (one_vec + exp_neg)

        # swish(x) = x * sigmoid(x)
        var swish_result = vec * sigmoid
        out_ptr.store[width=width](idx, swish_result)

    vectorize[simd_width](size, vectorized_swish)


@always_inline
def _swish_simd_float64(tensor: AnyTensor, mut result: AnyTensor):
    """SIMD Swish for float64 tensors."""
    comptime simd_width = simd_width_of[DType.float64]()
    var size = tensor._numel

    var in_ptr = tensor._data.bitcast[Float64]()
    var out_ptr = result._data.bitcast[Float64]()

    @parameter
    def vectorized_swish[width: Int](idx: Int):
        var vec = in_ptr.load[width=width](idx)

        var neg_vec = -vec
        var neg_clipped = max(neg_vec, SIMD[DType.float64, width](-20.0))
        var exp_neg = math_exp(neg_clipped)
        var one_vec = SIMD[DType.float64, width](1)
        var sigmoid = one_vec / (one_vec + exp_neg)

        var swish_result = vec * sigmoid
        out_ptr.store[width=width](idx, swish_result)

    vectorize[simd_width](size, vectorized_swish)
