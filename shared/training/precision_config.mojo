"""Precision configuration for multi-precision training.

This module provides a unified configuration for controlling training precision,
including compute dtype, storage dtype, and master weights management.

Key Features:
- Configurable precision modes: FP32, FP16, BF16, FP8
- Automatic gradient scaler management for reduced precision
- Master weight handling in FP32 for optimizer stability
- Tensor casting utilities for mixed precision workflows

Usage:
    # Create precision config for FP16 training
    var config = PrecisionConfig.fp16()

    # Use in training loop
    var model_input = config.cast_to_compute(input_tensor)
    var scaled_loss = config.scale_loss(loss)
    var unscaled_grads = config.unscale_gradients(grads)

See examples/mixed_precision_training.mojo for complete usage
"""

from shared.core.extensor import ExTensor, full, zeros
from shared.core.dtype_cast import cast_tensor
from shared.core.numerical_safety import has_nan, has_inf
from shared.training.mixed_precision import (
    GradientScaler,
    convert_to_fp32_master,
    update_model_from_master,
    check_gradients_finite,
    clip_gradients_by_norm,
)
from shared.training.dtype_utils import (
    float16_dtype,
    float32_dtype,
    float64_dtype,
    bfloat16_dtype,
    is_reduced_precision,
    dtype_to_string,
)


@fieldwise_init
struct PrecisionMode(Copyable, ImplicitlyCopyable, Movable, Stringable):
    """Precision mode enumeration.

    Supported modes:
    - FP32: Full precision (32-bit float).
    - FP16: Half precision (16-bit float).
    - BF16: Brain float (16-bit, wider exponent range).
    - FP8: Quarter precision (8-bit float, E4M3 format).
    """

    var value: Int
    """Internal value representing the precision mode."""

    comptime FP32 = PrecisionMode(value=0)
    comptime FP16 = PrecisionMode(value=1)
    comptime BF16 = PrecisionMode(value=2)
    comptime FP8 = PrecisionMode(value=3)

    fn __eq__(self, other: Self) -> Bool:
        return self.value == other.value

    fn __ne__(self, other: Self) -> Bool:
        return self.value != other.value

    fn __str__(self) -> String:
        if self.value == 0:
            return "fp32"
        elif self.value == 1:
            return "fp16"
        elif self.value == 2:
            return "bf16"
        elif self.value == 3:
            return "fp8"
        else:
            return "unknown"


struct PrecisionConfig(Copyable, Movable):
    """Central configuration for multi-precision training.

    Manages precision settings including:
    - Compute dtype: Precision used for forward/backward passes.
    - Storage dtype: Precision for storing model weights.
    - Master dtype: Precision for optimizer state (always FP32).
    - Gradient scaling: Automatic scaling for reduced precision.

    The config provides utilities for:
    - Casting tensors between precisions.
    - Managing gradient scaling.
    - Tracking training metrics (overflow detection).

    Attributes:
        mode: Precision mode (FP32, FP16, BF16, FP8).
        compute_dtype: DType for computations.
        storage_dtype: DType for weight storage.
        master_dtype: DType for optimizer (always float32).
        use_gradient_scaler: Whether to use gradient scaling.
        scaler: GradientScaler for loss/gradient scaling.

    Example:
        ```mojo
         Create config for mixed precision training
        var config = PrecisionConfig.fp16()

        # Cast input to compute precision
        var x = config.cast_to_compute(input)

        # Forward pass produces FP16 outputs
        var y = model.forward(x)

        # Scale loss before backward pass
        var scaled_loss = config.scale_loss(loss)
        ```
    """

    var mode: PrecisionMode
    """Precision mode (FP32, FP16, BF16, FP8)."""
    var compute_dtype: DType
    """DType for forward/backward passes."""
    var storage_dtype: DType
    """DType for model weights."""
    var master_dtype: DType
    """DType for optimizer state (always float32)."""
    var use_gradient_scaler: Bool
    """Whether to use gradient scaling."""
    var scaler: GradientScaler
    """Gradient scaler for loss/gradient scaling."""
    var _overflow_count: Int
    """Number of gradient overflows detected."""
    var _step_count: Int
    """Total number of training steps."""

    fn __init__(
        out self,
        mode: PrecisionMode,
        compute_dtype: DType,
        storage_dtype: DType,
        use_gradient_scaler: Bool = True,
        initial_scale: Float32 = 65536.0,
    ):
        """Initialize precision configuration.

        Args:
            mode: Precision mode (FP32, FP16, BF16, FP8).
            compute_dtype: DType for forward/backward passes.
            storage_dtype: DType for model weights.
            use_gradient_scaler: Enable gradient scaling (required for FP16).
            initial_scale: Initial loss scale for gradient scaler.
        """
        self.mode = mode
        self.compute_dtype = compute_dtype
        self.storage_dtype = storage_dtype
        self.master_dtype = DType.float32  # Always FP32 for optimizer stability
        self.use_gradient_scaler = use_gradient_scaler
        self.scaler = GradientScaler(initial_scale=initial_scale)
        self._overflow_count = 0
        self._step_count = 0

    @staticmethod
    fn fp32() -> PrecisionConfig:
        """Create FP32 (full precision) configuration.

        No gradient scaling needed for FP32 training.

        Returns:
            PrecisionConfig with FP32 settings.
        """
        return PrecisionConfig(
            mode=PrecisionMode.FP32,
            compute_dtype=DType.float32,
            storage_dtype=DType.float32,
            use_gradient_scaler=False,
        )

    @staticmethod
    fn fp16(initial_scale: Float32 = 65536.0) -> PrecisionConfig:
        """Create FP16 (half precision) configuration.

        Uses gradient scaling to prevent underflow in FP16.

        Args:
            initial_scale: Initial loss scale (default: 2^16).

        Returns:
            PrecisionConfig with FP16 settings.
        """
        return PrecisionConfig(
            mode=PrecisionMode.FP16,
            compute_dtype=DType.float16,
            storage_dtype=DType.float16,
            use_gradient_scaler=True,
            initial_scale=initial_scale,
        )

    @staticmethod
    fn bf16(initial_scale: Float32 = 65536.0) -> PrecisionConfig:
        """Create BF16 (brain float) configuration.

        BF16 has wider exponent range than FP16, reducing overflow risk.
        Still uses gradient scaling for safety.

        Args:
            initial_scale: Initial loss scale (default: 2^16).
                          BF16 has wider range than FP16, so slightly lower
                          initial scale can be used, but 2^16 is safe default.

        Returns:
            PrecisionConfig with BF16 settings.

        Note:
            Uses native DType.bfloat16 via bfloat16_dtype in dtype_utils.mojo.
            Not supported on Apple Silicon hardware (use FP16 instead).

        BF16 Characteristics:
            - 1 sign + 8 exponent + 7 mantissa = 16 bits.
            - Range: ~1e-38 to 3.4e38 (same as FP32).
            - Precision: ~2 decimal digits (less than FP16).
            - Better for large models due to wider exponent range.
        """
        return PrecisionConfig(
            mode=PrecisionMode.BF16,
            compute_dtype=bfloat16_dtype,
            storage_dtype=bfloat16_dtype,
            use_gradient_scaler=True,
            initial_scale=initial_scale,
        )

    @staticmethod
    fn fp8(initial_scale: Float32 = 65536.0) -> PrecisionConfig:
        """Create FP8 (quarter precision) configuration.

        FP8 has very limited range, requires aggressive scaling and monitoring.
        Uses FP16 for storage to reduce quantization noise.

        Args:
            initial_scale: Initial loss scale (default: 2^16 = 65536.0).
                          FP8 has much smaller range than FP16, so gradient
                          overflows are more likely. Higher initial scale (2^16+)
                          is recommended to prevent early overflows.

        Returns:
            PrecisionConfig with FP8 settings.

        Note:
            FP8 compute is experimental and currently uses FP16 as a fallback.
            When Mojo adds native FP8 support, this will use true FP8 compute.

        FP8 Characteristics (E4M3 format when available):
            - 1 sign + 4 exponent + 3 mantissa = 8 bits.
            - Range: ~1.5e-4 to 448 (very limited).
            - Precision: ~1 decimal digit (very low).
            - Requires: Aggressive gradient scaling, tight clipping, lower LR.
            - Use Cases: Very large models (>1GB), memory-constrained training.

        Current Implementation:
            - Compute: FP16 (fallback, will use FP8 when available).
            - Storage: FP16 (reduces quantization noise vs pure FP8).
            - Scaling: Recommended initial_scale >= 2^16 for stability.
        """
        # FP8 for compute, FP16 for storage (reduces quantization noise)
        return PrecisionConfig(
            mode=PrecisionMode.FP8,
            compute_dtype=DType.float16,  # Will use FP8 when available
            storage_dtype=DType.float16,
            use_gradient_scaler=True,
            initial_scale=initial_scale,
        )

    @staticmethod
    fn from_string(precision_str: String) raises -> PrecisionConfig:
        """Create PrecisionConfig from string name.

        Args:
            precision_str: Precision name ("fp32", "fp16", "bf16", "fp8").

        Returns:
            PrecisionConfig for the specified precision.

        Raises:
            Error: If precision_str is not recognized.
        """
        if precision_str == "fp32":
            return PrecisionConfig.fp32()
        elif precision_str == "fp16":
            return PrecisionConfig.fp16()
        elif precision_str == "bf16":
            return PrecisionConfig.bf16()
        elif precision_str == "fp8":
            return PrecisionConfig.fp8()
        else:
            raise Error(
                "Unknown precision: "
                + precision_str
                + ". Use fp32, fp16, bf16, or fp8."
            )

    fn cast_to_compute(self, tensor: ExTensor) raises -> ExTensor:
        """Cast tensor to compute precision.

        Args:
            tensor: Input tensor (any dtype).

        Returns:
            Tensor cast to compute_dtype.

        Raises:
            Error: If dtype conversion is not supported.
        """
        if tensor.dtype() == self.compute_dtype:
            return tensor
        return cast_tensor(tensor, self.compute_dtype)

    fn cast_to_storage(self, tensor: ExTensor) raises -> ExTensor:
        """Cast tensor to storage precision.

        Args:
            tensor: Input tensor (any dtype).

        Returns:
            Tensor cast to storage_dtype.

        Raises:
            Error: If dtype conversion is not supported.
        """
        if tensor.dtype() == self.storage_dtype:
            return tensor
        return cast_tensor(tensor, self.storage_dtype)

    fn cast_to_master(self, tensor: ExTensor) raises -> ExTensor:
        """Cast tensor to master (FP32) precision.

        Args:
            tensor: Input tensor (any dtype).

        Returns:
            Tensor cast to float32.

        Raises:
            Error: If dtype conversion is not supported.
        """
        return convert_to_fp32_master(tensor)

    fn scale_loss(self, loss: ExTensor) raises -> ExTensor:
        """Scale loss for mixed precision training.

        For FP32, returns loss unchanged.
        For reduced precision, applies gradient scaler.

        Args:
            loss: Unscaled loss tensor.

        Returns:
            Scaled loss tensor.

        Raises:
            Error: If loss scaling operation fails or produces invalid values.
        """
        if not self.use_gradient_scaler:
            return loss
        return self.scaler.scale_loss(loss)

    fn unscale_gradients(self, gradients: ExTensor) raises -> ExTensor:
        """Unscale gradients after backward pass.

        For FP32, returns gradients unchanged.
        For reduced precision, applies inverse of gradient scaler.

        Args:
            gradients: Scaled gradients from backward pass.

        Returns:
            Unscaled gradients ready for optimizer.

        Raises:
            Error: If gradient unscaling fails or scale is zero.
        """
        if not self.use_gradient_scaler:
            return gradients
        return self.scaler.unscale_gradients(gradients)

    fn check_gradients(self, gradients: ExTensor) raises -> Bool:
        """Check if gradients are valid (no NaN/Inf).

        Args:
            gradients: Gradient tensor to check.

        Returns:
            True if gradients are finite, False if NaN/Inf detected.

        Raises:
            Error: If gradient validation encounters non-finite values.
        """
        return check_gradients_finite(gradients)

    fn step(mut self, grads_valid: Bool):
        """Update scaler state after training step.

        Call after each training step to update gradient scaler.
        If gradients were valid, may increase scale.
        If gradients overflowed, reduces scale.

        Args:
            grads_valid: True if gradients were finite, False if overflow.
        """
        self._step_count += 1
        if not self.use_gradient_scaler:
            return

        if grads_valid:
            self.scaler.step()
        else:
            self.scaler.backoff()
            self._overflow_count += 1

    fn get_scale(self) -> Float32:
        """Get current gradient scale factor.

        Returns:
            Current loss scale value.
        """
        return self.scaler.get_scale()

    fn get_overflow_count(self) -> Int:
        """Get number of gradient overflows detected.

        Returns:
            Number of steps skipped due to overflow.
        """
        return self._overflow_count

    fn get_step_count(self) -> Int:
        """Get total number of training steps.

        Returns:
            Total number of steps (including overflows).
        """
        return self._step_count

    fn needs_master_weights(self) -> Bool:
        """Check if master weights are needed.

        Returns True for reduced precision training where
        FP32 master weights are needed for optimizer stability.

        Returns:
            True if mode is FP16, BF16, or FP8.
        """
        return self.mode != PrecisionMode.FP32

    fn clip_gradients(
        self, gradients: ExTensor, max_norm: Float32
    ) raises -> ExTensor:
        """Clip gradients by global norm.

        Useful for preventing gradient explosion in mixed precision.

        Args:
            gradients: Gradient tensor.
            max_norm: Maximum allowed gradient norm.

        Returns:
            Clipped gradients.

        Raises:
            Error: If gradient clipping fails or max_norm is non-positive.
        """
        return clip_gradients_by_norm(gradients, max_norm)

    fn print_config(self):
        """Print precision configuration summary to stdout.

        Displays the current precision settings including mode, compute/storage/master
        dtypes, and gradient scaler status.

        Example:
            ```mojo
            var config = PrecisionConfig.for_mode("fp16")
            config.print_config()
            # Output:
            # PrecisionConfig:
            #   Mode: fp16
            #   Compute dtype: float16
            #   Storage dtype: float16
            #   Master dtype: float32
            #   Gradient scaler: enabled
            #   Current scale: 65504.0
            ```
        """
        print("PrecisionConfig:")
        print("  Mode: " + String(self.mode))
        print("  Compute dtype: " + dtype_to_string(self.compute_dtype))
        print("  Storage dtype: " + dtype_to_string(self.storage_dtype))
        print("  Master dtype: " + dtype_to_string(self.master_dtype))
        print(
            "  Gradient scaler: "
            + ("enabled" if self.use_gradient_scaler else "disabled")
        )
        if self.use_gradient_scaler:
            print("  Current scale: " + String(self.get_scale()))

    fn print_stats(self):
        """Print training statistics to stdout.

        Displays step count, overflow count, overflow rate, and current gradient scale
        if gradient scaler is enabled.

        Example:
            ```mojo
            var config = PrecisionConfig.for_mode("fp16")
            # ... after training ...
            config.print_stats()
            # Output:
            # Training Stats:
            #   Total steps: 1000
            #   Overflow count: 5
            #   Overflow rate: 0.5%
            #   Current scale: 65504.0
            ```
        """
        print("Training Stats:")
        print("  Total steps: " + String(self._step_count))
        print("  Overflow count: " + String(self._overflow_count))
        if self._step_count > 0:
            var overflow_rate = (
                Float32(self._overflow_count)
                / Float32(self._step_count)
                * 100.0
            )
            print("  Overflow rate: " + String(overflow_rate) + "%")
        if self.use_gradient_scaler:
            print("  Current scale: " + String(self.get_scale()))
