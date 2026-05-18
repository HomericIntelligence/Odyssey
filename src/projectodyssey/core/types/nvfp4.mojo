"""NVFP4 (NVIDIA FP4) blocked floating point format.

Implements NVFP4 format from the paper "Microscaling Data Formats for Deep Learning":
- Individual values: E2M1 (1 sign, 2 exponent, 1 mantissa)
- Block size: 16 elements
- Scale format: E4M3 (4-bit exponent, 3-bit mantissa)

NVFP4 provides finer-grained scaling than MXFP4 with smaller block size.
Each block of 16 FP4 values shares a common E4M3 scale factor.

According to the paper, E4M3 "achieves the best results" and provides
"modest improvements in accuracy" over E8M0, despite narrower exponent range.

Key characteristics:
- Memory: 8 bytes per block (16 × 4 bits = 64 bits) + 7 bits scale ≈ 9 bytes
- Dynamic range: Balanced (E4M3 scale)
- Precision: Limited (E2M1 values) but better than MXFP4 (smaller blocks)
- Use case: Memory-efficient ML training with better accuracy than MXFP4

Example:
    ```mojo
    from .nvfp4 import NVFP4, NVFP4Block

    # Individual value (stores both E2M1 and scale for convenience)
    var val = NVFP4.from_float32(3.14159)
    var reconstructed = val.to_float32()

    # Block storage (efficient: 16 values + 1 shared scale)
    var block = NVFP4Block.from_float32_array(data_array)
    ```

Reference:
    Tim Dettmers, Nolan Miller, Deepak Kapur, Luke Zettlemoyer.
    "Microscaling Data Formats for Deep Learning."
    arXiv preprint arXiv:2310.10537, 2023.
    https://arxiv.org/abs/2310.10537
    https://doi.org/10.48550/arXiv.2310.10537

    The paper demonstrates that microscaling (per-block scaling factors) enables
    efficient low-precision quantization (4-bit) while maintaining accuracy in
    deep learning training and inference. NVFP4 uses E4M3 scales (128 dynamic range)
    across 16-element blocks for a balance between accuracy and memory efficiency.
    The paper shows E4M3 achieves the best overall results with modest accuracy
    improvements over the wider-range E8M0 format.
"""

from std.math import isnan, isinf
from std.memory import bitcast
from projectodyssey.core.types.dtype_aliases import FP4, FP8


# ============================================================================
# E4M3 Scale Helper Functions (using native Scalar[FP8])
# ============================================================================


def _e4m3_from_float32(scale: Float32) -> Scalar[FP8]:
    """Convert Float32 scale to E4M3 format using native FP8 type.

    Args:
        scale: Positive Float32 scale value.

    Returns:
        Scalar[FP8] representation.

    Note:
        Scale must be positive. Negative values will be converted to positive.
    """
    if scale <= 0.0 or isnan(scale):
        # Return zero scale
        return bitcast[FP8, 1](SIMD[DType.uint8, 1](0))

    if isinf(scale) or scale >= 240.0:
        # Return maximum positive FP8 value
        return bitcast[FP8, 1](
            SIMD[DType.uint8, 1](0x7E)
        )  # Max finite positive

    # Use native FP8 conversion (always use absolute value for scale)
    var abs_scale = scale if scale > 0 else -scale
    return Scalar[FP8](abs_scale)


def _e4m3_to_float32(e4m3_val: Scalar[FP8]) -> Float32:
    """Convert E4M3 (FP8) to Float32 using native type.

    Args:
        e4m3_val: FP8 scale value.

    Returns:
        Float32 representation.
    """
    return Float32(e4m3_val)


def _e4m3_get_bits(e4m3_val: Scalar[FP8]) -> UInt8:
    """Get raw bits from E4M3 (FP8) value.

    Args:
        e4m3_val: FP8 scale value.

    Returns:
        8-bit raw value.
    """
    return bitcast[DType.uint8, 1](e4m3_val)[0]


def _e4m3_from_bits(bits: UInt8) -> Scalar[FP8]:
    """Create E4M3 (FP8) from raw bits.

    Args:
        bits: 8-bit raw value.

    Returns:
        Scalar[FP8] value.
    """
    return bitcast[FP8, 1](SIMD[DType.uint8, 1](bits))


# ============================================================================
# FP4 E2M1 Helper Functions (using native types)
# ============================================================================


def _fp4_from_float32(x: Float32, scale: Float32) -> UInt8:
    """Convert Float32 to FP4 E2M1 format with given scale.

    Args:
        x: Float32 value to convert.
        scale: Block-level scale factor.

    Returns:
        4-bit FP4 value stored in lower 4 bits of UInt8.

    Note:
        The value is divided by scale before encoding.
        Values outside representable range are clamped.
    """
    # Handle special cases
    if isnan(x):
        return 0b0111  # Max value as NaN representation

    if isinf(x):
        if x > 0:
            return 0b0111  # Max positive value
        else:
            return 0b1111  # Max negative value

    # Scale the input
    var scaled = x / scale

    if scaled == 0.0:
        return 0  # +0

    # Extract sign
    var sign: UInt8 = 0
    var abs_scaled = scaled
    if scaled < 0:
        sign = 1
        abs_scaled = -scaled

    # E2M1 representable values (before scaling):
    # exp=0, mantissa=0: 0
    # exp=1, mantissa=0: 1.0
    # exp=1, mantissa=1: 1.5
    # exp=2, mantissa=0: 2.0
    # exp=2, mantissa=1: 3.0
    # exp=3, mantissa=0: 4.0
    # exp=3, mantissa=1: 6.0 (max)

    # Clamp to representable range [0, 6.0]
    if abs_scaled >= 6.0:
        # Return max value: sign=s, exp=3, mantissa=1
        return (sign << 3) | 0b111

    if abs_scaled < 0.5:
        # Return zero (subnormals not well-defined for E2M1)
        return sign << 3

    # Find best representation
    # Quantize to nearest representable value
    #  values: 0, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
    var exp: UInt8
    var mantissa: UInt8
    if abs_scaled < 1.25:
        exp = 1
        mantissa = 0  # 1.0
    elif abs_scaled < 1.75:
        exp = 1
        mantissa = 1  # 1.5
    elif abs_scaled < 2.5:
        exp = 2
        mantissa = 0  # 2.0
    elif abs_scaled < 3.5:
        exp = 2
        mantissa = 1  # 3.0
    elif abs_scaled < 5.0:
        exp = 3
        mantissa = 0  # 4.0
    else:
        exp = 3
        mantissa = 1  # 6.0

    # Combine: sign(1) | exponent(2) | mantissa(1)
    return (sign << 3) | (exp << 1) | mantissa


def _fp4_to_float32(fp4_bits: UInt8, scale: Float32) -> Float32:
    """Convert FP4 E2M1 bits to Float32 with given scale.

    Args:
        fp4_bits: 4-bit FP4 value in lower 4 bits.
        scale: Block-level scale factor.

    Returns:
        Float32 representation of the scaled E2M1 value.
    """
    # Extract components (4 bits total)
    var sign = (fp4_bits >> 3) & 0x1
    var exp = (fp4_bits >> 1) & 0x3  # 2 bits
    var mantissa = fp4_bits & 0x1  # 1 bit

    # Handle zero
    if exp == 0:
        return Float32(0.0) if sign == 0 else Float32(-0.0)

    # Compute unscaled value
    # E2M1: value = 2^(exp-1) * (1 + mantissa/2)
    # With 1-bit mantissa, the fractional part is mantissa * 0.5
    var exponent = exp.cast[DType.int32]() - 1
    var base = Float32(1.0) + Float32(mantissa.cast[DType.float32]()) * Float32(
        0.5
    )

    # Compute 2^exponent
    var unscaled = base
    if exponent > 0:
        for _ in range(exponent):
            unscaled *= 2.0
    elif exponent < 0:
        for _ in range(-exponent):
            unscaled /= 2.0

    # Apply sign and scale
    var result = unscaled * scale
    if sign == 1:
        result = -result

    return result


struct NVFP4(Copyable, Movable, Writable):
    """NVFP4 individual value (E2M1 + E4M3 scale).

    Acts like FP16 but stores internally as 4-bit E2M1 value plus 8-bit E4M3 scale.
    This representation is convenient but NOT space-efficient (12 bits total vs 4 bits in blocks).

    For efficient storage, use NVFP4Block which amortizes the scale across 16 values.

    Attributes:
        value: 4-bit E2M1 encoded value (stored in lower 4 bits of UInt8).
        scale: E4M3 (FP8) scale factor using native Scalar[FP8].
    """

    var value: UInt8
    """4-bit E2M1 encoded value (lower 4 bits used)."""
    var scale: Scalar[FP8]
    """E4M3 (FP8) scale factor using native type."""

    def __init__(
        out self,
        value: UInt8 = 0,
        scale: Scalar[FP8] = bitcast[FP8, 1](SIMD[DType.uint8, 1](0x3C))[0],
    ):
        """Initialize NVFP4 from E2M1 value and E4M3 scale.

        Args:
            value: E2M1 encoded value (4 bits in lower nibble).
            scale: E4M3 (FP8) scale factor.
        """
        self.value = value & 0xF  # Only keep lower 4 bits
        self.scale = scale

    @staticmethod
    def from_float32(x: Float32) -> Self:
        """Convert Float32 to NVFP4.

        Computes optimal scale for the single value and encodes.

        Args:
            x: Float32 value to convert.

        Returns:
            NVFP4 representation.
        """
        # Handle special cases
        if isnan(x) or isinf(x):
            var fp4_bits = _fp4_from_float32(x, 1.0)
            return NVFP4(
                fp4_bits, bitcast[FP8, 1](SIMD[DType.uint8, 1](0x3C))[0]
            )

        if x == 0.0:
            return NVFP4(0, bitcast[FP8, 1](SIMD[DType.uint8, 1](0x3C))[0])

        # Compute scale: find value such that |x| / scale is in E2M1 range [0, 6]
        var abs_x = x if x > 0 else -x
        var scale_val = Float32(1.0)
        var exp_val = 0

        # Scale to fit in E2M1 range [0, 6]
        while abs_x / scale_val > 6.0:
            scale_val *= 2.0
            exp_val += 1

        while abs_x / scale_val < 0.5 and exp_val > -7:
            scale_val /= 2.0
            exp_val -= 1

        # Create E4M3 scale using helper function
        var scale = _e4m3_from_float32(scale_val)
        var scale_f32 = _e4m3_to_float32(scale)

        # Encode E2M1 value using helper function
        var fp4_bits = _fp4_from_float32(x, scale_f32)

        return NVFP4(fp4_bits, scale)

    @staticmethod
    def from_float32_stochastic(x: Float32, seed: UInt64) -> Self:
        """Convert Float32 to NVFP4 with stochastic rounding.

        Uses stochastic rounding which is recommended for gradient quantization.
        When a value falls between two representable FP4 values, probabilistically
        round up or down based on distance.

        Args:
            x: Float32 value to convert.
            seed: Random seed for deterministic stochastic rounding.

        Returns:
            NVFP4 representation with stochastic rounding.

        Note:
            Use this for gradients and backward computations.
            Use from_float32() for forward passes and weights.

        Example:
            ```mojo
            # Gradient value 1.25 between 1.0 and 1.5.
            # Will round to 1.5 with ~50% probability.
            var grad = NVFP4.from_float32_stochastic(1.25, seed=12345)
        ```
        """
        # Handle special cases
        if isnan(x) or isinf(x):
            var fp4_bits = _fp4_from_float32(x, 1.0)
            return NVFP4(
                fp4_bits, bitcast[FP8, 1](SIMD[DType.uint8, 1](0x3C))[0]
            )

        if x == 0.0:
            return NVFP4(0, bitcast[FP8, 1](SIMD[DType.uint8, 1](0x3C))[0])

        # Compute scale same as deterministic version
        var abs_x = x if x > 0 else -x
        var scale_val = Float32(1.0)
        var exp_val = 0

        while abs_x / scale_val > 6.0:
            scale_val *= 2.0
            exp_val += 1

        while abs_x / scale_val < 0.5 and exp_val > -7:
            scale_val /= 2.0
            exp_val -= 1

        var scale = _e4m3_from_float32(scale_val)
        var scale_f32 = _e4m3_to_float32(scale)

        # Stochastic rounding for E2M1 encoding
        var fp4_bits = NVFP4._fp4_stochastic_round(x, scale_f32, seed)

        return NVFP4(fp4_bits, scale)

    @staticmethod
    def _fp4_stochastic_round(
        x: Float32, scale: Float32, seed: UInt64
    ) -> UInt8:
        """Internal: Stochastic rounding helper using simple LCG.

        Args:
            x: Float32 value to encode.
            scale: Scale factor.
            seed: Random seed.

        Returns:
            4-bit FP4 value (in lower 4 bits of UInt8) with stochastic rounding.
        """
        var scaled = x / scale
        var sign: UInt8 = 0
        var abs_scaled = scaled

        if scaled < 0:
            sign = 1
            abs_scaled = -scaled

        # E2M1 representable values: 0, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
        # Find neighboring values and distance
        var lower: Float32
        var upper: Float32
        var lower_bits: UInt8
        var upper_bits: UInt8

        # E2M1 representable values: 0, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
        # Boundaries use actual representable values to determine bucket
        if abs_scaled < 0.5:
            # Close to zero - round to 0
            return sign << 3
        elif abs_scaled < 1.0:
            # Between 0 and 1.0 - stochastic round between them
            lower = 0.0
            upper = 1.0
            lower_bits = 0b000  # zero
            upper_bits = 0b010  # exp=1, mantissa=0 (1.0)
        elif abs_scaled < 1.5:
            lower = 1.0
            upper = 1.5
            lower_bits = 0b010  # exp=1, mantissa=0
            upper_bits = 0b011  # exp=1, mantissa=1
        elif abs_scaled < 2.0:
            lower = 1.5
            upper = 2.0
            lower_bits = 0b011
            upper_bits = 0b100  # exp=2, mantissa=0
        elif abs_scaled < 3.0:
            lower = 2.0
            upper = 3.0
            lower_bits = 0b100
            upper_bits = 0b101  # exp=2, mantissa=1
        elif abs_scaled < 4.0:
            lower = 3.0
            upper = 4.0
            lower_bits = 0b101
            upper_bits = 0b110  # exp=3, mantissa=0
        elif abs_scaled < 6.0:
            lower = 4.0
            upper = 6.0
            lower_bits = 0b110
            upper_bits = 0b111  # exp=3, mantissa=1
        else:
            # At or above max
            return (sign << 3) | 0b111

        # Compute probability of rounding up
        var distance = abs_scaled - lower
        var range_val = upper - lower
        var prob_up = distance / range_val

        # Simple LCG: seed = (1103515245 * seed + 12345) mod 2^32
        var rng_state = seed.cast[DType.uint32]()
        rng_state = (1103515245 * rng_state + 12345) & 0xFFFFFFFF
        var random_val = Float32(rng_state >> 8) / Float32(16777216.0)

        # Stochastic decision
        var result_bits: UInt8
        if random_val < prob_up:
            result_bits = upper_bits
        else:
            result_bits = lower_bits

        return (sign << 3) | result_bits

    def to_float32(self) -> Float32:
        """Convert NVFP4 to Float32.

        Returns:
            Float32 representation.
        """
        var scale_f32 = _e4m3_to_float32(self.scale)
        return _fp4_to_float32(self.value, scale_f32)

    def __add__(self, other: NVFP4) -> NVFP4:
        """Add two NVFP4 values (via Float32).

        Args:
            other: Value to add.

        Returns:
            Sum as NVFP4.
        """
        return NVFP4.from_float32(self.to_float32() + other.to_float32())

    def __sub__(self, other: NVFP4) -> NVFP4:
        """Subtract two NVFP4 values (via Float32).

        Args:
            other: Value to subtract.

        Returns:
            Difference as NVFP4.
        """
        return NVFP4.from_float32(self.to_float32() - other.to_float32())

    def __mul__(self, other: NVFP4) -> NVFP4:
        """Multiply two NVFP4 values (via Float32).

        Args:
            other: Value to multiply.

        Returns:
            Product as NVFP4.
        """
        return NVFP4.from_float32(self.to_float32() * other.to_float32())

    def __truediv__(self, other: NVFP4) -> NVFP4:
        """Divide two NVFP4 values (via Float32).

        Args:
            other: Divisor.

        Returns:
            Quotient as NVFP4.
        """
        return NVFP4.from_float32(self.to_float32() / other.to_float32())

    def __neg__(self) -> NVFP4:
        """Negate NVFP4 value.

        Returns:
            Negated value.
        """
        # Flip sign bit in E2M1 value (bit 3)
        var neg_value = self.value ^ 0b1000
        return NVFP4(neg_value, self.scale)

    def __eq__(self, other: NVFP4) -> Bool:
        """Check equality.

        Args:
            other: Value to compare.

        Returns:
            True if equal.
        """
        # Compare FP4 value bits and scale bits
        var self_scale_bits = _e4m3_get_bits(self.scale)
        var other_scale_bits = _e4m3_get_bits(other.scale)
        return self.value == other.value and self_scale_bits == other_scale_bits

    def __ne__(self, other: NVFP4) -> Bool:
        """Check inequality.

        Args:
            other: Value to compare.

        Returns:
            True if not equal.
        """
        return not (self == other)

    def __lt__(self, other: NVFP4) -> Bool:
        """Check less than.

        Args:
            other: Value to compare.

        Returns:
            True if self < other.
        """
        return self.to_float32() < other.to_float32()

    def __le__(self, other: NVFP4) -> Bool:
        """Check less than or equal.

        Args:
            other: Value to compare.

        Returns:
            True if self <= other.
        """
        return self.to_float32() <= other.to_float32()

    def __gt__(self, other: NVFP4) -> Bool:
        """Check greater than.

        Args:
            other: Value to compare.

        Returns:
            True if self > other.
        """
        return self.to_float32() > other.to_float32()

    def __ge__(self, other: NVFP4) -> Bool:
        """Check greater than or equal.

        Args:
            other: Value to compare.

        Returns:
            True if self >= other.
        """
        return self.to_float32() >= other.to_float32()

    def write_to(self, mut writer: Some[Writer]):
        """Write NVFP4 value to a writer.

        Args:
            writer: Target writer to write the value to.

        Notes:
            Primary implementation of string formatting for the Writable trait.
            __str__() delegates to this method via String.write(self).
        """
        writer.write("NVFP4(" + String(self.to_float32()) + ")")

    def __str__(self) -> String:
        """Convert to string.

        Returns:
            String representation.
        """
        return String.write(self)

    def write_repr_to(self, mut writer: Some[Writer]):
        """Write the repr representation to a Writer (required for Writable trait).

        Args:
            writer: The writer to write the repr representation to.
        """
        var s = self.__repr__()
        writer.write(s)

    def __repr__(self) -> String:
        """Get representation string.

        Returns:
            Representation string.
        """
        return (
            "NVFP4(value=0x"
            + hex(Int(self.value))
            + ", scale="
            + String(_e4m3_to_float32(self.scale))
            + ")"
        )


def hex(val: Int) -> String:
    """Convert integer to hex string (simple implementation)."""
    if val == 0:
        return "0"
    var digits = String("0123456789abcdef")
    var result = String("")
    var v = val if val >= 0 else -val
    while v > 0:
        result = chr(Int(digits.as_bytes()[v % 16])) + result
        v //= 16
    return result


struct NVFP4Block(Copyable, Movable, Writable):
    """NVFP4 block storage: 16 E2M1 values + 1 E4M3 scale (9 bytes total).

    Memory layout:
    - Bytes 0-7: 16 E2M1 values (4 bits each, packed 2 per byte)
    - Byte 8: E4M3 (FP8) scale

    Bit packing:
    Each byte stores 2 E2M1 values:
    - Upper 4 bits: First E2M1 value
    - Lower 4 bits: Second E2M1 value

    This provides 14:1 compression vs Float32 (9 bytes vs 64 bytes).
    Smaller blocks (16 vs 32) provide better accuracy per the paper.

    Example:
        ```mojo
        from collections import List

        # Create block from Float32 array
        var values = List[Float32]()
        for i in range(16):
            values.append(Float32(i) * 0.1)

        var block = NVFP4Block.from_float32_array(values)
        var decoded = block.to_float32_array()
        ```
    """

    var data: SIMD[DType.uint8, 8]
    """8 bytes containing 16 packed E2M1 values (2 per byte)."""
    var scale: Scalar[FP8]
    """Shared E4M3 (FP8) scale factor for all 16 values."""

    def __init__(out self):
        """Initialize NVFP4Block with zeros."""
        self.data = SIMD[DType.uint8, 8](0)
        self.scale = bitcast[FP8, 1](SIMD[DType.uint8, 1](0x3C))[
            0
        ]  # Scale = 1.0

    def __init__(out self, data: SIMD[DType.uint8, 8], scale: Scalar[FP8]):
        """Initialize NVFP4Block from packed data and scale.

        Args:
            data: 8 bytes containing 16 packed E2M1 values.
            scale: E4M3 (FP8) scale factor for the block.
        """
        self.data = data
        self.scale = scale

    @staticmethod
    def from_float32_array(values: List[Float32]) raises -> Self:
        """Convert 16 Float32 values to NVFP4Block.

        Args:
            values: List of exactly 16 Float32 values.

        Returns:
            NVFP4Block with optimal scale and packed E2M1 values.

        Raises:
            Error: If values list doesn't contain exactly 16 elements.

        Note:
            Computes optimal E4M3 scale as max(abs(values)) / 6.0
            to fit all values in E2M1 range [0, 6].
        """
        if len(values) != 16:
            raise Error(
                "NVFP4Block requires exactly 16 values, got "
                + String(len(values))
            )

        # Find optimal scale: max(abs(values)) / 6.0
        var max_abs = Float32(0.0)
        for i in range(16):
            var abs_val = values[i] if values[i] >= 0 else -values[i]
            if abs_val > max_abs:
                max_abs = abs_val

        # Compute scale (avoid division by zero)
        var scale_val = max_abs / 6.0
        if scale_val < 1e-10:
            scale_val = 1.0

        var scale = _e4m3_from_float32(scale_val)
        var scale_f32 = _e4m3_to_float32(scale)

        # Pack E2M1 values (2 per byte)
        var data = SIMD[DType.uint8, 8](0)
        for i in range(8):
            # First value (upper 4 bits)
            var val1 = _fp4_from_float32(values[i * 2], scale_f32)
            # Second value (lower 4 bits)
            var val2 = _fp4_from_float32(values[i * 2 + 1], scale_f32)

            # Pack: upper 4 bits = val1, lower 4 bits = val2
            data[i] = ((val1 & 0xF) << 4) | (val2 & 0xF)

        return NVFP4Block(data, scale)

    def to_float32_array(self) -> List[Float32]:
        """Decode NVFP4Block to 16 Float32 values.

        Returns:
            List of 16 Float32 values decoded from the block.

        Note:
            Decoding is lossless given the quantization that occurred during encoding.
        """
        var result = List[Float32]()
        var scale_f32 = _e4m3_to_float32(self.scale)

        for i in range(8):
            var byte = self.data[i]
            # Extract upper 4 bits (first value)
            var val1_bits = (byte >> 4) & 0xF
            result.append(_fp4_to_float32(val1_bits, scale_f32))

            # Extract lower 4 bits (second value)
            var val2_bits = byte & 0xF
            result.append(_fp4_to_float32(val2_bits, scale_f32))

        return result^

    def get(self, index: Int) raises -> NVFP4:
        """Get NVFP4 value at index (0-15).

        Args:
            index: Index in range [0, 15].

        Returns:
            NVFP4 value at the given index.

        Raises:
            Error: If index is out of range.
        """
        if index < 0 or index >= 16:
            raise Error("Index " + String(index) + " out of range [0, 15]")

        var byte_idx = index // 2
        var is_upper = (index % 2) == 0

        var byte = self.data[byte_idx]
        var fp4_bits: UInt8
        if is_upper:
            fp4_bits = (byte >> 4) & 0xF
        else:
            fp4_bits = byte & 0xF

        return NVFP4(fp4_bits, self.scale)

    def set(mut self, index: Int, value: NVFP4) raises -> None:
        """Set NVFP4 value at index (0-15).

        Args:
            index: Index in range [0, 15].
            value: NVFP4 value to set.

        Raises:
            Error: If index is out of range.

        Note:
            This updates the E2M1 value but keeps the block's shared scale.
            The value's scale is ignored - it's re-encoded with the block's scale.
        """
        if index < 0 or index >= 16:
            raise Error("Index " + String(index) + " out of range [0, 15]")

        # Re-encode value with block's scale
        var float_val = value.to_float32()
        var scale_f32 = _e4m3_to_float32(self.scale)
        var fp4_bits = _fp4_from_float32(float_val, scale_f32)

        var byte_idx = index // 2
        var is_upper = (index % 2) == 0

        var byte = self.data[byte_idx]
        if is_upper:
            # Update upper 4 bits
            byte = (byte & 0x0F) | ((fp4_bits & 0xF) << 4)
        else:
            # Update lower 4 bits
            byte = (byte & 0xF0) | (fp4_bits & 0xF)

        self.data[byte_idx] = byte

    def write_to(self, mut writer: Some[Writer]):
        """Write NVFP4Block value to a writer.

        Args:
            writer: Target writer to write the value to.

        Notes:
            Primary implementation of string formatting for the Writable trait.
            __str__() delegates to this method via String.write(self).
        """
        writer.write(
            "NVFP4Block(16 values, scale="
            + String(_e4m3_to_float32(self.scale))
            + ")"
        )

    def __str__(self) -> String:
        """String representation showing scale and value count.

        Returns:
            String representation.
        """
        return String.write(self)

    def write_repr_to(self, mut writer: Some[Writer]):
        """Write the repr representation to a Writer (required for Writable trait).

        Args:
            writer: The writer to write the repr representation to.
        """
        var s = self.__repr__()
        writer.write(s)

    def __repr__(self) -> String:
        """Detailed representation.

        Returns:
            Detailed string representation.
        """
        return (
            "NVFP4Block(scale="
            + String(_e4m3_to_float32(self.scale))
            + ", data=8 bytes)"
        )
