"""MXFP4 (Microscaling FP4) blocked floating point format.

Implements MXFP4 format from the paper "Microscaling Data Formats for Deep Learning":
- Individual values: E2M1 (1 sign, 2 exponent, 1 mantissa)
- Block size: 32 elements
- Scale format: E8M0 (8-bit exponent-only, no mantissa or sign)

MXFP4 provides wide dynamic range through exponent-only scaling.
Each block of 32 FP4 values shares a common E8M0 scale factor.

Key characteristics:
- Memory: 16 bytes per block (32 × 4 bits = 128 bits) + 1 byte scale = 17 bytes
- Dynamic range: Wide (E8M0 scale)
- Precision: Limited (E2M1 values)
- Use case: Memory-efficient ML training and inference

Example:
    from .mxfp4 import MXFP4, MXFP4Block

    # Individual value (stores both E2M1 and scale for convenience)
    var val = MXFP4.from_float32(3.14159)
    var reconstructed = val.to_float32()

    # Block storage (efficient: 32 values + 1 shared scale)
    var block = MXFP4Block.from_float32_array(data_array)
    ```

Reference:
    Tim Dettmers, Nolan Miller, Deepak Kapur, Luke Zettlemoyer
    "Microscaling Data Formats for Deep Learning."
    arXiv preprint arXiv:2310.10537, 2023
    https://arxiv.org/abs/2310.10537
    https://doi.org/10.48550/arXiv.2310.10537

    The paper demonstrates that microscaling (per-block scaling factors) enables
    efficient low-precision quantization (4-bit) while maintaining accuracy in
    deep learning training and inference. MXFP4 uses E8M0 scales (256 dynamic range)
    across 32-element blocks for maximum efficiency.
"""

from math import isnan, isinf
from memory import bitcast
from .dtype_aliases import FP4, E8M0


# ============================================================================
# E8M0 Scale Helper Functions (using native Scalar[E8M0])
# ============================================================================


fn _e8m0_from_float32(scale: Float32) -> Scalar[E8M0]:
    """Convert Float32 scale to E8M0 format using manual exponent extraction.

    Args:
        scale: Positive Float32 scale value.

    Returns:
        Scalar[E8M0] representation (power of 2 closest to scale).

    Note:
        E8M0 is exponent-only (no mantissa), so it can only represent powers of 2.
        The conversion extracts the Float32 exponent and rounds to nearest power of 2.
        Scale must be positive. Negative or zero values return minimum scale.
    """
    if scale <= 0.0 or isnan(scale):
        # Return minimum scale by bitcasting exponent 0
        return bitcast[E8M0, 1](SIMD[DType.uint8, 1](0))

    if isinf(scale):
        # Return maximum scale by bitcasting exponent 255
        return bitcast[E8M0, 1](SIMD[DType.uint8, 1](255))

    # E8M0 can only represent powers of 2, so we extract the exponent from Float32
    # Float32 format: 1 sign + 8 exponent + 23 mantissa, exponent bias = 127
    var bits = bitcast[DType.uint32, 1](SIMD[DType.float32, 1](scale))
    var float_exp = ((bits[0] >> 23) & 0xFF).cast[DType.uint8]()

    # Adjust for mantissa: if mantissa >= 0.5, round exponent up
    var mantissa = bits[0] & 0x7FFFFF  # 23-bit mantissa
    if mantissa >= 0x400000:  # >= 0.5 in binary fraction
        if float_exp < 255:
            float_exp += 1

    return bitcast[E8M0, 1](SIMD[DType.uint8, 1](float_exp))


fn _e8m0_to_float32(e8m0_val: Scalar[E8M0]) -> Float32:
    """Convert E8M0 to Float32 using manual exponent reconstruction.

    Args:
        e8m0_val: E8M0 scale value.

    Returns:
        Float32 representation (power of 2).

    Note:
        E8M0 is exponent-only, so the result is always a power of 2.
        Constructs Float32 with the E8M0 exponent and zero mantissa.
    """
    # Get the 8-bit exponent from E8M0
    var exp = bitcast[DType.uint8, 1](e8m0_val)[0]

    # Handle special cases
    if exp == 0:
        return Float32(0.0)  # Minimum scale (or could return subnormal)
    if exp == 255:
        return Float32(1.0) / Float32(0.0)  # Infinity

    # Construct Float32: sign=0, exponent=exp, mantissa=0
    # Float32 bits: 0 (sign) | exp (8 bits) | 0...0 (23 bits mantissa)
    var float_bits = exp.cast[DType.uint32]() << 23
    return bitcast[DType.float32, 1](SIMD[DType.uint32, 1](float_bits))[0]


fn _e8m0_get_exponent(e8m0_val: Scalar[E8M0]) -> UInt8:
    """Get raw exponent bits from E8M0 value.

    Args:
        e8m0_val: E8M0 scale value.

    Returns:
        8-bit exponent value.
    """
    return bitcast[DType.uint8, 1](e8m0_val)[0]


fn _e8m0_from_exponent(exponent: UInt8) -> Scalar[E8M0]:
    """Create E8M0 from raw exponent bits.

    Args:
        exponent: 8-bit exponent value (bias = 127).

    Returns:
        Scalar[E8M0] value.
    """
    return bitcast[E8M0, 1](SIMD[DType.uint8, 1](exponent))


# ============================================================================
# FP4 E2M1 Helper Functions (using native Scalar[FP4])
# ============================================================================


fn _fp4_from_float32(x: Float32, scale: Float32) -> UInt8:
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
    # Representable values: 0, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
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


fn _fp4_to_float32(fp4_bits: UInt8, scale: Float32) -> Float32:
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


struct MXFP4(Copyable, Movable, Representable, Stringable):
    """MXFP4 individual value (E2M1 + E8M0 scale).

    Acts like FP16 but stores internally as 4-bit E2M1 value plus 8-bit E8M0 scale.
    This representation is convenient but NOT space-efficient (12 bits total vs 4 bits in blocks).

    For efficient storage, use MXFP4Block which amortizes the scale across 32 values.

    Attributes:
        value: 4-bit E2M1 encoded value (stored in lower 4 bits of UInt8).
        scale: 8-bit E8M0 scale factor (native Scalar[E8M0]).
    """

    var value: UInt8
    """4-bit E2M1 encoded value (lower 4 bits)."""
    var scale: Scalar[E8M0]
    """8-bit E8M0 scale factor."""

    fn __init__(
        out self, value: UInt8 = 0, scale: Scalar[E8M0] = _e8m0_from_exponent(127)
    ):
        """Initialize MXFP4 from E2M1 value and E8M0 scale.

        Args:
            value: E2M1 encoded value (4-bit value in lower bits).
            scale: E8M0 scale factor.
        """
        self.value = value & 0xF  # Mask to 4 bits
        self.scale = scale

    @staticmethod
    fn from_float32(x: Float32) -> Self:
        """Convert Float32 to MXFP4.

        Computes optimal scale for the single value and encodes.

        Args:
            x: Float32 value to convert.

        Returns:
            MXFP4 representation.
        """
        # Handle special cases
        if isnan(x) or isinf(x):
            return MXFP4(_fp4_from_float32(x, 1.0), _e8m0_from_exponent(127))

        if x == 0.0:
            return MXFP4(0, _e8m0_from_exponent(127))

        # Compute scale: find 2^k such that |x| / 2^k is in E2M1 range [0, 6]
        var abs_x = x if x > 0 else -x
        var scale_val = Float32(1.0)
        var exp_val = 0

        # Scale to fit in E2M1 range [0, 6]
        while abs_x / scale_val > 6.0:
            scale_val *= 2.0
            exp_val += 1

        while abs_x / scale_val < 0.5 and exp_val > -127:
            scale_val /= 2.0
            exp_val -= 1

        # Create E8M0 scale
        var scale = _e8m0_from_float32(scale_val)

        # Encode E2M1 value
        var value = _fp4_from_float32(x, _e8m0_to_float32(scale))

        return MXFP4(value, scale)

    @staticmethod
    fn from_float32_stochastic(x: Float32, seed: UInt64) -> Self:
        """Convert Float32 to MXFP4 with stochastic rounding.

        Uses stochastic rounding which is recommended for gradient quantization.
        When a value falls between two representable FP4 values, probabilistically
        round up or down based on distance.

        Args:
            x: Float32 value to convert.
            seed: Random seed for deterministic stochastic rounding.

        Returns:
            MXFP4 representation with stochastic rounding.

        Note:
            Use this for gradients and backward computations.
            Use from_float32() for forward passes and weights.

        Example:
            ```mojo
             Gradient value 1.25 between 1.0 and 1.5.
            # Will round to 1.5 with ~50% probability.
            var grad = MXFP4.from_float32_stochastic(1.25, seed=12345)
        ```
        """
        # Handle special cases
        if isnan(x) or isinf(x):
            return MXFP4(_fp4_from_float32(x, 1.0), _e8m0_from_exponent(127))

        if x == 0.0:
            return MXFP4(0, _e8m0_from_exponent(127))

        # Compute scale same as deterministic version
        var abs_x = x if x > 0 else -x
        var scale_val = Float32(1.0)
        var exp_val = 0

        while abs_x / scale_val > 6.0:
            scale_val *= 2.0
            exp_val += 1

        while abs_x / scale_val < 0.5 and exp_val > -127:
            scale_val /= 2.0
            exp_val -= 1

        var scale = _e8m0_from_float32(scale_val)
        var scale_f32 = _e8m0_to_float32(scale)

        # Stochastic rounding for E2M1 encoding
        var value = MXFP4._fp4_stochastic_round(x, scale_f32, seed)

        return MXFP4(value, scale)

    @staticmethod
    fn _fp4_stochastic_round(x: Float32, scale: Float32, seed: UInt64) -> UInt8:
        """Internal: Stochastic rounding helper using simple LCG.

        Args:
            x: Float32 value to encode.
            scale: Scale factor.
            seed: Random seed.

        Returns:
            4-bit FP4 value with stochastic rounding (stored in lower 4 bits of UInt8).
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

    fn to_float32(self) -> Float32:
        """Convert MXFP4 to Float32.

        Returns:
            Float32 representation.
        """
        return _fp4_to_float32(self.value, _e8m0_to_float32(self.scale))

    fn __add__(self, other: MXFP4) -> MXFP4:
        """Add two MXFP4 values (via Float32).

        Args:
            other: Value to add.

        Returns:
            Sum as MXFP4.
        """
        return MXFP4.from_float32(self.to_float32() + other.to_float32())

    fn __sub__(self, other: MXFP4) -> MXFP4:
        """Subtract two MXFP4 values (via Float32).

        Args:
            other: Value to subtract.

        Returns:
            Difference as MXFP4.
        """
        return MXFP4.from_float32(self.to_float32() - other.to_float32())

    fn __mul__(self, other: MXFP4) -> MXFP4:
        """Multiply two MXFP4 values (via Float32).

        Args:
            other: Value to multiply.

        Returns:
            Product as MXFP4.
        """
        return MXFP4.from_float32(self.to_float32() * other.to_float32())

    fn __truediv__(self, other: MXFP4) -> MXFP4:
        """Divide two MXFP4 values (via Float32).

        Args:
            other: Divisor.

        Returns:
            Quotient as MXFP4.
        """
        return MXFP4.from_float32(self.to_float32() / other.to_float32())

    fn __neg__(self) -> MXFP4:
        """Negate MXFP4 value.

        Returns:
            Negated value.
        """
        # Flip sign bit in E2M1 value
        var neg_value = self.value ^ 0b1000
        return MXFP4(neg_value, self.scale)

    fn __eq__(self, other: MXFP4) -> Bool:
        """Check equality.

        Args:
            other: Value to compare.

        Returns:
            True if equal.
        """
        return self.value == other.value and _e8m0_get_exponent(
            self.scale
        ) == _e8m0_get_exponent(other.scale)

    fn __ne__(self, other: MXFP4) -> Bool:
        """Check inequality.

        Args:
            other: Value to compare.

        Returns:
            True if not equal.
        """
        return not (self == other)

    fn __lt__(self, other: MXFP4) -> Bool:
        """Check less than.

        Args:
            other: Value to compare.

        Returns:
            True if self < other.
        """
        return self.to_float32() < other.to_float32()

    fn __le__(self, other: MXFP4) -> Bool:
        """Check less than or equal.

        Args:
            other: Value to compare.

        Returns:
            True if self <= other.
        """
        return self.to_float32() <= other.to_float32()

    fn __gt__(self, other: MXFP4) -> Bool:
        """Check greater than.

        Args:
            other: Value to compare.

        Returns:
            True if self > other.
        """
        return self.to_float32() > other.to_float32()

    fn __ge__(self, other: MXFP4) -> Bool:
        """Check greater than or equal.

        Args:
            other: Value to compare.

        Returns:
            True if self >= other.
        """
        return self.to_float32() >= other.to_float32()

    fn __str__(self) -> String:
        """Convert to string.

        Returns:
            String representation.
        """
        return "MXFP4(" + String(self.to_float32()) + ")"

    fn __repr__(self) -> String:
        """Get representation string.

        Returns:
            Representation string.
        """
        return (
            "MXFP4(value=0x"
            + hex(self.value)
            + ", scale="
            + String(_e8m0_to_float32(self.scale))
            + ")"
        )


struct MXFP4Block(Copyable, Movable, Representable, Stringable):
    """MXFP4 block storage: 32 E2M1 values + 1 E8M0 scale (17 bytes total).

    Memory layout:
    - Bytes 0-15: 32 E2M1 values (4 bits each, packed 2 per byte).
    - Byte 16: E8M0 scale (8-bit exponent).

    Bit packing:
    Each byte stores 2 E2M1 values:
    - Upper 4 bits: First E2M1 value.
    - Lower 4 bits: Second E2M1 value.

    This provides 16:1 compression vs Float32 (17 bytes vs 128 bytes).

    Example:
        ```
        from collections import List

        # Create block from Float32 array
        var values : List[Float32]()
        for i in range(32):
            values.append(Float32(i) * 0.1)

        var block = MXFP4Block.from_float32_array(values)
        var decoded = block.to_float32_array()
        ```
    """

    var data: SIMD[DType.uint8, 16]
    """16 bytes containing 32 packed E2M1 values (2 per byte)."""
    var scale: Scalar[E8M0]
    """Shared E8M0 scale factor for all 32 values."""

    fn __init__(out self):
        """Initialize MXFP4Block with zeros."""
        self.data = SIMD[DType.uint8, 16](0)
        self.scale = _e8m0_from_exponent(127)  # Scale = 1.0

    fn __init__(out self, data: SIMD[DType.uint8, 16], scale: Scalar[E8M0]):
        """Initialize MXFP4Block from packed data and scale.

        Args:
            data: 16 bytes containing 32 packed E2M1 values.
            scale: E8M0 scale factor for the block.
        """
        self.data = data
        self.scale = scale

    @staticmethod
    fn from_float32_array(values: List[Float32]) raises -> Self:
        """Convert 32 Float32 values to MXFP4Block.

        Args:
            values: List of exactly 32 Float32 values.

        Returns:
            MXFP4Block with optimal scale and packed E2M1 values.

        Raises:
            Error: If values list doesn't contain exactly 32 elements.

        Note:
        ```
            Computes optimal E8M0 scale as max(abs(values)) / 6.0.
            to fit all values in E2M1 range [0, 6].
        ```
        """
        if len(values) != 32:
            raise Error(
                "MXFP4Block requires exactly 32 values, got "
                + String(len(values))
            )

        # Find optimal scale: max(abs(values)) / 6.0
        var max_abs = Float32(0.0)
        for i in range(32):
            var abs_val = values[i] if values[i] >= 0 else -values[i]
            if abs_val > max_abs:
                max_abs = abs_val

        # Compute scale (avoid division by zero)
        # Zero/near-zero blocks use scale=1.0 fallback - tested in test_mxfp4_block.mojo
        var scale_val = max_abs / 6.0
        if scale_val < 1e-10:
            scale_val = 1.0

        var scale = _e8m0_from_float32(scale_val)
        var scale_f32 = _e8m0_to_float32(scale)

        # Pack E2M1 values (2 per byte)
        var data = SIMD[DType.uint8, 16](0)
        for i in range(16):
            # First value (upper 4 bits)
            var val1 = _fp4_from_float32(values[i * 2], scale_f32)
            # Second value (lower 4 bits)
            var val2 = _fp4_from_float32(values[i * 2 + 1], scale_f32)

            # Pack: upper 4 bits = val1, lower 4 bits = val2
            data[i] = ((val1 & 0xF) << 4) | (val2 & 0xF)

        return MXFP4Block(data, scale)

    fn to_float32_array(self) -> List[Float32]:
        """Decode MXFP4Block to 32 Float32 values.

        Returns:
            List of 32 Float32 values decoded from the block.

        Note:
            Decoding is lossless given the quantization that occurred during encoding.
        """
        var result = List[Float32]()
        var scale_f32 = _e8m0_to_float32(self.scale)

        for i in range(16):
            var byte = self.data[i]
            # Extract upper 4 bits (first value)
            var val1_bits = (byte >> 4) & 0xF
            result.append(_fp4_to_float32(val1_bits, scale_f32))

            # Extract lower 4 bits (second value)
            var val2_bits = byte & 0xF
            result.append(_fp4_to_float32(val2_bits, scale_f32))

        return result^

    fn get(self, index: Int) raises -> MXFP4:
        """Get MXFP4 value at index (0-31).

        Args:
            index: Index in range [0, 31].

        Returns:
            MXFP4 value at the given index.

        Raises:
            Error: If index is out of range.
        """
        if index < 0 or index >= 32:
            raise Error("Index " + String(index) + " out of range [0, 31]")

        var byte_idx = index // 2
        var is_upper = (index % 2) == 0

        var byte = self.data[byte_idx]
        var fp4_bits: UInt8
        if is_upper:
            fp4_bits = (byte >> 4) & 0xF
        else:
            fp4_bits = byte & 0xF

        return MXFP4(fp4_bits, self.scale)

    fn set(mut self, index: Int, value: MXFP4) raises -> None:
        """Set MXFP4 value at index (0-31).

        Args:
            index: Index in range [0, 31].
            value: MXFP4 value to set.

        Raises:
            Error: If index is out of range.

        Note:
            This updates the E2M1 value but keeps the block's shared scale.
            The value's scale is ignored - it's re-encoded with the block's scale.
        """
        if index < 0 or index >= 32:
            raise Error("Index " + String(index) + " out of range [0, 31]")

        # Re-encode value with block's scale
        var float_val = value.to_float32()
        var fp4_bits = _fp4_from_float32(
            float_val, _e8m0_to_float32(self.scale)
        )

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

    fn __str__(self) -> String:
        """String representation showing scale and value count.

        Returns:
            String representation.
        """
        return (
            "MXFP4Block(32 values, scale="
            + String(_e8m0_to_float32(self.scale))
            + ")"
        )

    fn __repr__(self) -> String:
        """Detailed representation.

        Returns:
            Detailed string representation.
        """
        return (
            "MXFP4Block(scale="
            + String(_e8m0_to_float32(self.scale))
            + ", data=16 bytes)"
        )
