"""DType aliases and utilities for mixed precision training.

Provides convenience aliases and dtype utilities for mixed precision training.

Key Differences:
- Float16 (FP16):  1 sign + 5 exponent + 10 mantissa = 16 bits
  - Range: ~6e-8 to 65504
  - Precision: ~3 decimal digits

- BFloat16 (BF16): 1 sign + 8 exponent + 7 mantissa = 16 bits
  - Range: ~1e-38 to 3.4e38 (same as FP32)
  - Precision: ~2 decimal digits

BF16 trades precision for range compared to FP16. BF16 is now natively
supported in Mojo via DType.bfloat16.

Note:
    DType.bfloat16 is NOT supported on Apple Silicon.

Usage:
    from shared.training.dtype_utils import (
        bfloat16_dtype,
        recommend_precision_dtype,
        detect_hardware_bf16_support,
    )

    # Automatic detection of BF16 support based on hardware
    var has_bf16 = detect_hardware_bf16_support()
    var dtype = recommend_precision_dtype(model_size_mb=500.0, hardware_has_bf16=has_bf16)
    var params = ExTensor.zeros((100, 100), dtype)

    # Manual control for specific cases
    var dtype = recommend_precision_dtype(model_size_mb=500.0, hardware_has_bf16=False)  # Force FP16
    var params = ExTensor.zeros((100, 100), dtype)

    # Or use BFloat16 on supported hardware (not Apple Silicon)
    var params = ExTensor.zeros((100, 100), bfloat16_dtype)  # Only on Intel/AMD
"""


# ============================================================================
# DType Aliases
# ============================================================================

comptime float16_dtype = DType.float16
"""Float16 (FP16) dtype - Half precision floating point.

Fully supported in Mojo. Use for mixed precision training
- 1 sign bit, 5 exponent bits, 10 mantissa bits
- Range: ~6e-8 to 65504
- Memory: 2 bytes
"""

comptime float32_dtype = DType.float32
"""Float32 (FP32) dtype - Single precision floating point.

Default precision for most training. Standard IEEE 754 format
- 1 sign bit, 8 exponent bits, 23 mantissa bits
- Range: ~1e-38 to 3.4e38
- Memory: 4 bytes
"""

comptime float64_dtype = DType.float64
"""Float64 (FP64) dtype - Double precision floating point.

High precision for numerical stability. Standard IEEE 754 format
- 1 sign bit, 11 exponent bits, 52 mantissa bits
- Range: ~2e-308 to 1.8e308
- Memory: 8 bytes
"""

comptime bfloat16_dtype = DType.bfloat16
"""BFloat16 (BF16) dtype - Brain floating point.

Native Mojo BFloat16 support via DType.bfloat16.

Properties:
- 1 sign bit, 8 exponent bits, 7 mantissa bits
- Range: ~1e-38 to 3.4e38 (same as FP32)
- Memory: 2 bytes
- Better for training than FP16 due to wider exponent range

Note:
    NOT supported on Apple Silicon. Use DType.float16 on Apple hardware.
"""


# ============================================================================
# DType Utility Functions
# ============================================================================


fn is_reduced_precision(dtype: DType) -> Bool:
    """Check if dtype uses reduced precision (FP16 or BF16).

        Returns True for any dtype using less than 32-bit floating point.
        Useful for conditional logic in mixed precision training.

    Args:
            dtype: DType to check.

    Returns:
            True if dtype is float16 or bfloat16, False otherwise.

        Example:
            ```mojo
            if is_reduced_precision(model.dtype()):
                # Use gradient scaling
                var scaler = GradientScaler()
            ```
    """
    return dtype == DType.float16 or dtype == DType.bfloat16


fn detect_hardware_bf16_support() -> Bool:
    """Detect if current hardware supports BF16 natively.

    Returns False for Apple Silicon (M1/M2/M3/M4) where BF16 is not supported,
    and True for other platforms (Intel/AMD x86_64) that support it.

    Note:
        This function attempts runtime detection using platform information.
        Returns False conservatively on detection failures.

    Returns:
            True if hardware supports BF16, False otherwise (including Apple Silicon).

    Example:
            ```mojo
            # Automatically detect BF16 support based on hardware
            var has_bf16 = detect_hardware_bf16_support()
            var dtype = recommend_precision_dtype(1000.0, hardware_has_bf16=has_bf16)
            ```
    """
    # Apple Silicon detection via platform module
    # BF16 is not supported on Apple Silicon (M1/M2/M3/M4)
    try:
        from python import Python
        var python = Python.import_module("platform")
        var machine = python.machine()
        var machine_str = String(machine)

        # Apple Silicon identifies as "arm64"
        if "arm64" in machine_str:
            return False  # Apple Silicon - no BF16 support

        # Other ARM platforms might support BF16, but be conservative
        if "arm" in machine_str:
            return False  # Conservative default for ARM

        # Intel/AMD x86_64 and aarch64 (non-Apple) support BF16
        if "x86_64" in machine_str or "amd64" in machine_str:
            return True
        if "aarch64" in machine_str:
            return True  # Non-Apple ARM64

        # Default to True for unknown platforms (optimistic)
        return True
    except:
        # If platform detection fails, return False (conservative)
        return False


fn is_floating_point(dtype: DType) -> Bool:
    """Check if dtype is a floating point type.

    Args:
            dtype: DType to check.

    Returns:
            True if dtype is float16, bfloat16, float32, or float64.

        Example:
            ```mojo
            if is_floating_point(tensor.dtype()):
                # Can use floating point operations
                var result = tensor / 2.0
            ```
    """
    return (
        dtype == DType.float16
        or dtype == DType.bfloat16
        or dtype == DType.float32
        or dtype == DType.float64
    )


fn get_dtype_precision_bits(dtype: DType) -> Int:
    """Get the number of mantissa bits for a floating point dtype.

        Returns the precision (mantissa bits) for floating point dtypes.
        Useful for understanding numerical precision limits.

    Args:
            dtype: DType to query.

    Returns:
            Number of mantissa bits (10 for FP16, 7 for BF16, 23 for FP32, 52 for FP64).
            Returns 0 for non-floating-point dtypes.

        Example:
            ```mojo
            var bits = get_dtype_precision_bits(DType.float16)
            print("FP16 has", bits, "mantissa bits")  # 10
            ```
    """
    if dtype == DType.float16:
        return 10  # FP16: 10 mantissa bits
    elif dtype == DType.bfloat16:
        return 7  # BF16: 7 mantissa bits
    elif dtype == DType.float32:
        return 23  # FP32: 23 mantissa bits
    elif dtype == DType.float64:
        return 52  # FP64: 52 mantissa bits
    else:
        return 0  # Not a floating point type


fn get_dtype_exponent_bits(dtype: DType) -> Int:
    """Get the number of exponent bits for a floating point dtype.

        Returns the exponent bits for floating point dtypes.
        Useful for understanding numerical range limits.

    Args:
            dtype: DType to query.

    Returns:
            Number of exponent bits (5 for FP16, 8 for BF16/FP32, 11 for FP64).
            Returns 0 for non-floating-point dtypes.

        Example:
            ```mojo
            var bits = get_dtype_exponent_bits(DType.float32)
            print("FP32 has", bits, "exponent bits")  # 8
            ```
    """
    if dtype == DType.float16:
        return 5  # FP16: 5 exponent bits (narrow range)
    elif dtype == DType.bfloat16:
        return 8  # BF16: 8 exponent bits (same range as FP32)
    elif dtype == DType.float32:
        return 8  # FP32: 8 exponent bits (wide range)
    elif dtype == DType.float64:
        return 11  # FP64: 11 exponent bits (very wide range)
    else:
        return 0  # Not a floating point type


fn dtype_to_string(dtype: DType) -> String:
    """Convert DType to human-readable string.

    Args:
            dtype: DType to convert.

    Returns:
            String representation (e.g., "float16", "bfloat16", "float32", "int32").

        Example:
            ```mojo
            var name = dtype_to_string(DType.float16)
            print("Using dtype:", name)  # "Using dtype: float16"
            ```
    """
    if dtype == DType.float16:
        return "float16"
    elif dtype == DType.bfloat16:
        return "bfloat16"
    elif dtype == DType.float32:
        return "float32"
    elif dtype == DType.float64:
        return "float64"
    elif dtype == DType.int8:
        return "int8"
    elif dtype == DType.int16:
        return "int16"
    elif dtype == DType.int32:
        return "int32"
    elif dtype == DType.int64:
        return "int64"
    elif dtype == DType.uint8:
        return "uint8"
    elif dtype == DType.uint16:
        return "uint16"
    elif dtype == DType.uint32:
        return "uint32"
    elif dtype == DType.uint64:
        return "uint64"
    elif dtype == DType.bool:
        return "bool"
    else:
        return "unknown"


fn recommend_precision_dtype(
    model_size_mb: Float64,
    hardware_has_fp16: Bool = True,
    hardware_has_bf16: Bool = False,
) -> DType:
    """Recommend optimal precision dtype based on model size and hardware.

        Provides guidance for choosing between FP32, FP16, and BF16 based on
        model characteristics and hardware capabilities.

    Args:
            model_size_mb: Model size in megabytes.
            hardware_has_fp16: Whether hardware supports FP16 acceleration.
            hardware_has_bf16: Whether hardware supports BF16 acceleration.
                Defaults to False (BF16 NOT supported on Apple Silicon M1/M2/M3).
                Pass hardware_has_bf16=True only on Intel/AMD x86_64 hardware.

    Returns:
            Recommended DType (float16, bfloat16, or float32).

        Recommendations:
            - Small models (<100MB): FP32 (speed gain minimal).
            - Medium models (100MB-1GB): FP16 if hardware supports it.
            - Large models (>1GB): BF16 if hardware supports it, else FP16.
            - No FP16 hardware: FP32 (reduced precision not worth it).

        Example:
            ```mojo
            # On Apple Silicon (default - no BF16)
            var dtype = recommend_precision_dtype(model_size_mb=500.0)
            # Returns FP16 for medium models, FP32 for small

            # On Intel/AMD with BF16 support
            var dtype = recommend_precision_dtype(model_size_mb=2000.0, hardware_has_bf16=True)
            # Returns BF16 for large models
            ```
    """
    if not hardware_has_fp16:
        # No hardware support - use FP32
        return DType.float32

    if model_size_mb < 100.0:
        # Small model - FP32 fine, speedup minimal
        return DType.float32
    elif model_size_mb < 1000.0:
        # Medium model - FP16 recommended
        return DType.float16
    else:
        # Large model - BF16 recommended if supported (not Apple Silicon)
        if hardware_has_bf16:
            return DType.bfloat16
        else:
            return DType.float16


fn print_dtype_info(dtype: DType):
    """Print detailed information about a DType.

        Displays precision, range, and memory usage for the given dtype.
        Useful for debugging and understanding dtype characteristics.

    Args:
            dtype: DType to describe.

        Example:
            ```mojo
            print_dtype_info(DType.float16)
            # Output:
            # DType: float16
            # Precision: 10 mantissa bits
            # Exponent: 5 bits
            # Range: ~6e-8 to 65504
            # Memory: 2 bytes
            ```
    """
    var name = dtype_to_string(dtype)
    print("DType: " + name)

    if is_floating_point(dtype):
        var precision = get_dtype_precision_bits(dtype)
        var exponent = get_dtype_exponent_bits(dtype)
        print("  Precision: " + String(precision) + " mantissa bits")
        print("  Exponent: " + String(exponent) + " bits")

        if dtype == DType.float16:
            print("  Range: ~6e-8 to 65504")
            print("  Memory: 2 bytes")
        elif dtype == DType.bfloat16:
            print("  Range: ~1e-38 to 3.4e38 (same as FP32)")
            print("  Memory: 2 bytes")
            print("  Note: NOT supported on Apple Silicon")
        elif dtype == DType.float32:
            print("  Range: ~1e-38 to 3.4e38")
            print("  Memory: 4 bytes")
        elif dtype == DType.float64:
            print("  Range: ~2e-308 to 1.8e308")
            print("  Memory: 8 bytes")
    else:
        if dtype == DType.int8 or dtype == DType.uint8:
            print("  Memory: 1 byte")
        elif dtype == DType.int16 or dtype == DType.uint16:
            print("  Memory: 2 bytes")
        elif dtype == DType.int32 or dtype == DType.uint32:
            print("  Memory: 4 bytes")
        elif dtype == DType.int64 or dtype == DType.uint64:
            print("  Memory: 8 bytes")
