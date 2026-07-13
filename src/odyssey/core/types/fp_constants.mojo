"""Constants for low-precision floating-point types (FP4, FP8, BF8).

This module provides centralized constants for low-precision floating-point
arithmetic, including normalization bounds, saturation thresholds, and scaling factors.

FP8 E4M3 Format:
    - 4-bit exponent, 3-bit mantissa
    - Min normal: 2^-6 = 0.015625
    - Max normal: 448.0
    - Used in NVIDIA, AMD low-precision training

FP4/E2M1 Format:
    - 2-bit exponent, 1-bit mantissa
    - Max normal: 6.0
    - Min subnormal: 0.5
    - Used in ultra-low precision quantization

BF8 E5M2 Format:
    - 5-bit exponent, 2-bit mantissa
    - Saturation: 57344.0
    - Wider exponent range than FP8

Stochastic Rounding:
    - Used for unbiased rounding in low-precision conversions
    - Scale factor: 2^24 for maximum precision in float32
"""

# FP8 E4M3 constants
comptime FP8_E4M3_MIN_NORMAL: Float32 = 0.015625  # 2^-6
comptime FP8_E4M3_MAX_NORMAL: Float32 = 448.0

# FP4/E2M1 constants (used by FP4, MXFP4, NVFP4)
comptime FP4_E2M1_MAX_NORMAL: Float32 = 6.0
comptime FP4_E2M1_MIN_SUBNORMAL: Float32 = 0.5
comptime FP4_E2M1_MANTISSA_SCALE: Float32 = 0.5

# BF8 E5M2 constants
comptime BF8_E5M2_SATURATION: Float32 = 57344.0
comptime BF8_E5M2_MANTISSA_SCALE: Float32 = 16384.0

# RNG scaling for stochastic rounding
comptime STOCHASTIC_ROUNDING_SCALE: Float32 = 16777216.0  # 2^24
