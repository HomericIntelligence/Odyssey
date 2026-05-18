"""Gradient clipping utilities for training stability.

Provides comprehensive gradient clipping functions to prevent exploding gradients
and improve training stability, especially for RNNs and deep networks.

Key Functions:
- clip_gradients_by_global_norm: Clip list of gradients by global norm
- clip_gradients_per_param: Clip each parameter's gradients independently
- clip_gradients_by_value_list: Clip list of gradients by value range
- compute_gradient_norm_list: Compute global gradient norm
- compute_gradient_statistics: Compute gradient statistics for monitoring

Re-exports from mixed_precision:
- clip_gradients_by_norm: Clip single gradient tensor by norm
- clip_gradients_by_value: Clip single gradient tensor by value

Performance:
- SIMD-vectorized inner loops for float32/float64 (4-8x speedup)
- Scalar fallback for other dtypes
- Follows patterns from activation_simd.mojo and matmul.mojo

Example:
    from projectodyssey.training.gradient_clipping import clip_gradients_by_global_norm

    # Clip gradients by global norm
    var gradients = [grad1, grad2, grad3]
    var total_norm = clip_gradients_by_global_norm(gradients, max_norm=1.0)

    if total_norm > 10.0:
        print("Warning: Large gradient norm detected:", total_norm)
"""

from std.algorithm import vectorize
from std.sys.info import simd_width_of
from projectodyssey.tensor.any_tensor import AnyTensor
from std.collections import List
from std.math import sqrt

# Re-export single-tensor clipping functions
from projectodyssey.training.mixed_precision import (
    clip_gradients_by_norm,
    clip_gradients_by_value,
)


# ============================================================================
# SIMD Helper Functions
# ============================================================================


@always_inline
def _norm_sq_simd_f32(tensor: AnyTensor) -> Float64:
    """Compute squared L2 norm using SIMD for float32 tensors."""
    comptime simd_width = simd_width_of[DType.float32]()
    var size = tensor._numel
    var ptr = tensor._data.bitcast[Float32]()
    var acc = Float64(0.0)

    @always_inline
    def vectorized_norm[width: Int](idx: Int) {var ptr, mut acc}:
        var vec = ptr.load[width=width](idx)
        var sq = vec * vec
        acc += sq.reduce_add().cast[DType.float64]()

    vectorize[simd_width](size, vectorized_norm)
    return acc


@always_inline
def _norm_sq_simd_f64(tensor: AnyTensor) -> Float64:
    """Compute squared L2 norm using SIMD for float64 tensors."""
    comptime simd_width = simd_width_of[DType.float64]()
    var size = tensor._numel
    var ptr = tensor._data.bitcast[Float64]()
    var acc = Float64(0.0)

    @always_inline
    def vectorized_norm[width: Int](idx: Int) {var ptr, mut acc}:
        var vec = ptr.load[width=width](idx)
        var sq = vec * vec
        acc += sq.reduce_add()

    vectorize[simd_width](size, vectorized_norm)
    return acc


@always_inline
def _scale_simd_f32(tensor: AnyTensor, scale: Float32):
    """Scale all elements in-place using SIMD for float32 tensors."""
    comptime simd_width = simd_width_of[DType.float32]()
    var size = tensor._numel
    var ptr = tensor._data.bitcast[Float32]()

    @always_inline
    def vectorized_scale[width: Int](idx: Int) {var ptr, var scale}:
        var vec = ptr.load[width=width](idx)
        var scale_vec = SIMD[DType.float32, width](scale)
        ptr.store[width=width](idx, vec * scale_vec)

    vectorize[simd_width](size, vectorized_scale)


@always_inline
def _scale_simd_f64(tensor: AnyTensor, scale: Float64):
    """Scale all elements in-place using SIMD for float64 tensors."""
    comptime simd_width = simd_width_of[DType.float64]()
    var size = tensor._numel
    var ptr = tensor._data.bitcast[Float64]()

    @always_inline
    def vectorized_scale[width: Int](idx: Int) {var ptr, var scale}:
        var vec = ptr.load[width=width](idx)
        var scale_vec = SIMD[DType.float64, width](scale)
        ptr.store[width=width](idx, vec * scale_vec)

    vectorize[simd_width](size, vectorized_scale)


@always_inline
def _clamp_simd_f32(tensor: AnyTensor, min_val: Float32, max_val: Float32):
    """Clamp all elements in-place using SIMD for float32 tensors."""
    comptime simd_width = simd_width_of[DType.float32]()
    var size = tensor._numel
    var ptr = tensor._data.bitcast[Float32]()

    @always_inline
    def vectorized_clamp[
        width: Int
    ](idx: Int) {var ptr, var min_val, var max_val}:
        var vec = ptr.load[width=width](idx)
        var min_vec = SIMD[DType.float32, width](min_val)
        var max_vec = SIMD[DType.float32, width](max_val)
        ptr.store[width=width](idx, max(min_vec, min(max_vec, vec)))

    vectorize[simd_width](size, vectorized_clamp)


@always_inline
def _clamp_simd_f64(tensor: AnyTensor, min_val: Float64, max_val: Float64):
    """Clamp all elements in-place using SIMD for float64 tensors."""
    comptime simd_width = simd_width_of[DType.float64]()
    var size = tensor._numel
    var ptr = tensor._data.bitcast[Float64]()

    @always_inline
    def vectorized_clamp[
        width: Int
    ](idx: Int) {var ptr, var min_val, var max_val}:
        var vec = ptr.load[width=width](idx)
        var min_vec = SIMD[DType.float64, width](min_val)
        var max_vec = SIMD[DType.float64, width](max_val)
        ptr.store[width=width](idx, max(min_vec, min(max_vec, vec)))

    vectorize[simd_width](size, vectorized_clamp)


def compute_gradient_norm_list(gradients: List[AnyTensor]) raises -> Float32:
    """Compute global L2 norm across all gradient tensors.

    Uses SIMD vectorization for float32/float64 tensors with scalar
    fallback for other dtypes.

    Args:
        gradients: List of gradient tensors.

    Returns:
        Global L2 norm of all gradients.

    Raises:
        Error: If computation fails.

    Example:
        ```mojo
        var norm = compute_gradient_norm_list(gradients)
        print("Gradient norm:", norm)
        ```
    """
    var total_norm_sq = Float64(0.0)

    for i in range(len(gradients)):
        var grad = gradients[i]

        if grad._dtype == DType.float32:
            total_norm_sq += _norm_sq_simd_f32(grad)
        elif grad._dtype == DType.float64:
            total_norm_sq += _norm_sq_simd_f64(grad)
        else:
            # Scalar fallback for other dtypes
            var numel = grad.numel()
            for j in range(numel):
                var val = grad._get_float64(j)
                total_norm_sq += val * val

    return Float32(sqrt(total_norm_sq))


def clip_gradients_by_global_norm(
    mut gradients: List[AnyTensor], max_norm: Float32
) raises -> Float32:
    """Clip gradients by global norm across all parameters.

    Computes the global L2 norm across all gradients and scales them if the
    norm exceeds max_norm. This is the most common gradient clipping method.

    Args:
        gradients: List of gradient tensors (modified in-place).
        max_norm: Maximum allowed global gradient norm.

    Returns:
        Total gradient norm before clipping.

    Raises:
        Error: If max_norm is non-positive.

    Example:
        ```mojo
        var gradients = [grad1, grad2, grad3]
        var norm = clip_gradients_by_global_norm(gradients, max_norm=1.0)

        if norm > 10.0:
            print("Large gradient detected, clipped from", norm)
        ```

    Note:
        This function modifies gradients in-place for efficiency.
        Common values: max_norm=1.0 (standard), max_norm=5.0 (aggressive).
    """
    if max_norm <= 0.0:
        raise Error("max_norm must be positive, got: " + String(max_norm))

    # Compute global gradient norm
    var total_norm = compute_gradient_norm_list(gradients)

    # Clip if needed
    if total_norm > max_norm:
        var clip_coef = Float64(max_norm) / (Float64(total_norm) + 1e-6)

        # Scale all gradients by clip coefficient using SIMD
        for i in range(len(gradients)):
            var grad = gradients[i]

            if grad._dtype == DType.float32:
                _scale_simd_f32(grad, Float32(clip_coef))
            elif grad._dtype == DType.float64:
                _scale_simd_f64(grad, clip_coef)
            else:
                var numel = grad.numel()
                for j in range(numel):
                    var val = grad._get_float64(j)
                    grad._set_float64(j, val * clip_coef)

    return total_norm


def clip_gradients_per_param(
    mut gradients: List[AnyTensor], max_norm: Float32
) raises:
    """Clip each parameter's gradients independently by their local norm.

    Unlike global norm clipping, this clips each parameter's gradient separately.
    Useful when different parameters have vastly different gradient scales.

    Args:
        gradients: List of gradient tensors (modified in-place).
        max_norm: Maximum allowed norm per parameter.

    Raises:
        Error: If max_norm is non-positive.

    Example:
        ```mojo
        var gradients = [grad1, grad2, grad3]
        clip_gradients_per_param(gradients, max_norm=1.0)
        ```

    Note:
        Less common than global norm clipping, but useful for:
        - Parameters with very different scales
        - Preventing individual parameter explosion.
    """
    if max_norm <= 0.0:
        raise Error("max_norm must be positive, got: " + String(max_norm))

    for i in range(len(gradients)):
        var grad = gradients[i]

        # Compute this parameter's gradient norm using SIMD
        var param_norm_sq: Float64
        if grad._dtype == DType.float32:
            param_norm_sq = _norm_sq_simd_f32(grad)
        elif grad._dtype == DType.float64:
            param_norm_sq = _norm_sq_simd_f64(grad)
        else:
            param_norm_sq = Float64(0.0)
            var numel = grad.numel()
            for j in range(numel):
                var val = grad._get_float64(j)
                param_norm_sq += val * val

        var param_norm = Float64(sqrt(param_norm_sq))

        # Clip if needed using SIMD
        if param_norm > Float64(max_norm):
            var clip_coef = Float64(max_norm) / (param_norm + 1e-6)

            if grad._dtype == DType.float32:
                _scale_simd_f32(grad, Float32(clip_coef))
            elif grad._dtype == DType.float64:
                _scale_simd_f64(grad, clip_coef)
            else:
                var numel = grad.numel()
                for j in range(numel):
                    var val = grad._get_float64(j)
                    grad._set_float64(j, val * clip_coef)


def clip_gradients_by_value_list(
    mut gradients: List[AnyTensor], min_value: Float32, max_value: Float32
) raises:
    """Clip all gradients by value range.

    Clamps each gradient value to [min_value, max_value].
    Simpler than norm clipping but less theoretically motivated.

    Args:
        gradients: List of gradient tensors (modified in-place).
        min_value: Minimum allowed gradient value.
        max_value: Maximum allowed gradient value.

    Raises:
        Error: If min_value >= max_value.

    Example:
        ```mojo
        var gradients = [grad1, grad2, grad3]
        clip_gradients_by_value_list(gradients, -1.0, 1.0)
        ```

    Note:
        Common for simple gradient control, but norm clipping is preferred.
    """
    if min_value >= max_value:
        raise Error(
            "min_value must be < max_value, got: "
            + String(min_value)
            + " >= "
            + String(max_value)
        )

    for i in range(len(gradients)):
        var grad = gradients[i]

        if grad._dtype == DType.float32:
            _clamp_simd_f32(grad, min_value, max_value)
        elif grad._dtype == DType.float64:
            _clamp_simd_f64(grad, Float64(min_value), Float64(max_value))
        else:
            var numel = grad.numel()
            for j in range(numel):
                var val = Float32(grad._get_float64(j))

                if val > max_value:
                    grad._set_float64(j, Float64(max_value))
                elif val < min_value:
                    grad._set_float64(j, Float64(min_value))


struct GradientStatistics:
    """Statistics about gradient magnitudes for monitoring.

    Attributes:
        global_norm: L2 norm across all gradients
        max_value: Maximum absolute gradient value
        min_value: Minimum absolute gradient value
        mean_value: Mean absolute gradient value
        num_params: Total number of parameters
        num_nan: Number of NaN values detected
        num_inf: Number of Inf values detected
    """

    var global_norm: Float32
    var max_value: Float32
    var min_value: Float32
    var mean_value: Float32
    var num_params: Int
    var num_nan: Int
    var num_inf: Int

    def __init__(
        out self,
        global_norm: Float32,
        max_value: Float32,
        min_value: Float32,
        mean_value: Float32,
        num_params: Int,
        num_nan: Int = 0,
        num_inf: Int = 0,
    ):
        """Initialize gradient statistics.

        Args:
            global_norm: Global L2 norm.
            max_value: Maximum absolute value.
            min_value: Minimum absolute value.
            mean_value: Mean absolute value.
            num_params: Total number of parameters.
            num_nan: Number of NaN values.
            num_inf: Number of Inf values.
        """
        self.global_norm = global_norm
        self.max_value = max_value
        self.min_value = min_value
        self.mean_value = mean_value
        self.num_params = num_params
        self.num_nan = num_nan
        self.num_inf = num_inf

    def is_healthy(self) -> Bool:
        """Check if gradients are healthy (no NaN/Inf).

        Returns:
            True if no NaN or Inf values detected.
        """
        return self.num_nan == 0 and self.num_inf == 0

    def print_summary(self):
        """Print gradient statistics summary."""
        print("Gradient Statistics:")
        print("  Global norm:    " + String(self.global_norm))
        print("  Max value:      " + String(self.max_value))
        print("  Min value:      " + String(self.min_value))
        print("  Mean value:     " + String(self.mean_value))
        print("  Num params:     " + String(self.num_params))
        print("  NaN count:      " + String(self.num_nan))
        print("  Inf count:      " + String(self.num_inf))


def compute_gradient_statistics(
    gradients: List[AnyTensor],
) raises -> GradientStatistics:
    """Compute comprehensive gradient statistics for monitoring.

    Useful for detecting gradient explosions, vanishing gradients, and NaN/Inf issues.

    Args:
        gradients: List of gradient tensors.

    Returns:
        GradientStatistics struct with computed metrics.

    Raises:
        Error: If computation fails.

    Example:
        ```mojo
        var stats = compute_gradient_statistics(gradients)
        stats.print_summary()

        if not stats.is_healthy():
            print("Warning: Gradient health check failed!")

        if stats.global_norm > 10.0:
            print("Warning: Large gradient norm detected!")
        ```
    """
    from std.math import isnan, isinf

    var total_norm_sq = Float64(0.0)
    var max_val = Float32(-1e9)
    var min_val = Float32(1e9)
    var sum_abs = Float64(0.0)
    var total_params = Int(0)
    var nan_count = Int(0)
    var inf_count = Int(0)

    for i in range(len(gradients)):
        var grad = gradients[i]
        var numel = grad.numel()
        total_params += numel

        for j in range(numel):
            var val = grad._get_float64(j)

            # Check for NaN/Inf
            if isnan(val):
                nan_count += 1
                continue
            if isinf(val):
                inf_count += 1
                continue

            # Accumulate statistics
            total_norm_sq += val * val
            var abs_val = val if val >= 0.0 else -val
            sum_abs += abs_val

            if abs_val > Float64(max_val):
                max_val = Float32(abs_val)
            if abs_val < Float64(min_val):
                min_val = Float32(abs_val)

    var global_norm = Float32(sqrt(total_norm_sq))
    var mean_val = Float32(
        sum_abs / Float64(total_params) if total_params > 0 else 0.0
    )

    return GradientStatistics(
        global_norm=global_norm,
        max_value=max_val,
        min_value=min_val,
        mean_value=mean_val,
        num_params=total_params,
        num_nan=nan_count,
        num_inf=inf_count,
    )
