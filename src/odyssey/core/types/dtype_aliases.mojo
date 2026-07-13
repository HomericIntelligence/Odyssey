"""Type aliases for Mojo built-in dtypes.

Provides short aliases for common dtypes used in ML training.
These aliases use Mojo's built-in DType constants for reduced precision types.

Aliases:
    BF16: BFloat16 (1 sign + 8 exponent + 7 mantissa)
    FP8: Float8 E4M3 (1 sign + 4 exponent + 3 mantissa)
    BF8: Float8 E5M2 (1 sign + 5 exponent + 2 mantissa)
    FP4: Float4 E2M1 (1 sign + 2 exponent + 1 mantissa)
    E8M0: Float8 E8M0 (8 exponent bits only, used for MXFP4 scale)

Example:
    ```mojo
    from .dtype_aliases import BF16, FP8

    # Use with Scalar
    var bf16_val = Scalar[BF16](3.14)
    var fp8_val = Scalar[FP8](2.5)

    # Use with SIMD
    var bf16_vec = SIMD[BF16, 4](1.0, 2.0, 3.0, 4.0)
    ```

Platform Note:
    DType.bfloat16 is not supported on Apple Silicon.
"""

# BFloat16 - Brain Floating Point (16-bit)
# Same exponent range as Float32, ideal for training
comptime BF16 = DType.bfloat16

# FP8 E4M3 - 8-bit float with 4 exponent + 3 mantissa bits
# NVIDIA/AMD standard for inference
comptime FP8 = DType.float8_e4m3fn

# BF8 E5M2 - 8-bit float with 5 exponent + 2 mantissa bits
# Wider range than FP8, less precision
comptime BF8 = DType.float8_e5m2

# FP4 E2M1 - 4-bit float with 2 exponent + 1 mantissa bit
# Used in blocked formats (MXFP4, NVFP4)
comptime FP4 = DType.float4_e2m1fn

# E8M0 - 8-bit exponent-only format (no mantissa, no sign)
# Used as scale factor in MXFP4 blocked format
comptime E8M0 = DType.float8_e8m0fnu
