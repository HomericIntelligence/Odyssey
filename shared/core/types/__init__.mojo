"""
Types and Data Structures Module.

This module contains custom types and data structures optimized for ML workloads.
All types are implemented using Mojo's struct system for memory safety and performance.

Components:
    - BF16: Type alias for DType.bfloat16
    - FP8: Type alias for DType.float8_e4m3fn (E4M3 format)
    - BF8: Type alias for DType.float8_e5m2 (E5M2 format)
    - FP4: Type alias for DType.float4_e2m1fn (E2M1 format)
    - E8M0: Type alias for DType.float8_e8m0fnu (exponent-only scale)
    - MXFP4: Microscaling FP4 with E8M0 scaling (32-element blocks)
    - NVFP4: NVIDIA FP4 with E4M3 scaling (16-element blocks)

Note:
    For integer types, use Mojo's built-in types (Int8, Int16, etc.)
    The type aliases (BF16, FP8, BF8, FP4, E8M0) are DType values, not structs.
    Use Scalar[BF16], Scalar[FP8], etc. for scalar values.

Example:
    ```mojo
    from shared.core.types import BF16, FP8, BF8, FP4, MXFP4, NVFP4

    # Work with FP8 values using native SIMD
    var fp8_val = Scalar[FP8](3.14159)
    var float_val = Float32(fp8_val)

    # Work with BF8 values
    var bf8_val = Scalar[BF8](100.0)
    var float_val2 = Float32(bf8_val)

    # Work with BF16 values (brain floating point)
    var bf16_val = Scalar[BF16](1e30)
    var float_val3 = Float32(bf16_val)

    # Work with blocked FP4 values
    var mxfp4_val = MXFP4.from_float32(2.718)
    var nvfp4_val = NVFP4.from_float32(1.414)
    var reconstructed = mxfp4_val.to_float32()
    ```
"""

# Type alias exports (DType aliases for native Mojo types)
from .dtype_aliases import BF16, FP8, BF8, FP4, E8M0

# Blocked FP4 format exports (custom structs for microscaling)
from .mxfp4 import MXFP4, MXFP4Block
from .nvfp4 import NVFP4, NVFP4Block

# FP type constants
from .fp_constants import (
    FP8_E4M3_MIN_NORMAL,
    FP8_E4M3_MAX_NORMAL,
    FP4_E2M1_MAX_NORMAL,
    FP4_E2M1_MIN_SUBNORMAL,
    FP4_E2M1_MANTISSA_SCALE,
    BF8_E5M2_SATURATION,
    BF8_E5M2_MANTISSA_SCALE,
    STOCHASTIC_ROUNDING_SCALE,
)

# Future exports will be added here as components are implemented
# from .tensor import Tensor
# from .shape import Shape
# from .dtype import DType
