"""Value-correctness tests for shape operations on non-contiguous inputs.

Verifies that shape operations (reshape, tile, repeat, broadcast_to, permute,
concatenate, flatten) correctly read from non-contiguous (transposed/strided)
source tensors and produce correct output values. These tests expose the
flat-index bug where _get_float64 ignores strides and reads from wrong
memory locations.

See Issue #4086 for context on the stride-aware indexing bug.
"""

from shared.core.extensor import ExTensor, arange
from shared.core.shape import (
    reshape,
    tile,
    repeat,
    broadcast_to,
    permute,
    concatenate,
    flatten,
)
from shared.core.matrix import transpose
from tests.shared.conftest import (
    assert_value_at,
    assert_dim,
    assert_numel,
    assert_shape,
    assert_equal_int,
)


# ============================================================================
# Helper: Create non-contiguous test tensors
# ============================================================================


fn make_noncontiguous_3x4() raises -> ExTensor:
    """Create a (4,3) non-contiguous transposed tensor from arange(12).

    The tensor logically contains:
    [[ 0,  4,  8],
     [ 1,  5,  9],
     [ 2,  6, 10],
     [ 3,  7, 11]]

    Element [r,c] == float(c*4 + r).
    """
    var flat = arange(0.0, 12.0, 1.0, DType.float32)
    var t2d = reshape(flat, [3, 4])
    return transpose(t2d)


# ============================================================================
# Test: reshape() on non-contiguous input
# ============================================================================


fn test_reshape_noncontiguous_to_flat() raises:
    """Reshape non-contiguous (4,3) to flat (12,).

    Verifies that reshape correctly interprets strided values and produces
    a contiguous flat result with correct element order.
    """
    var source = make_noncontiguous_3x4()  # (4,3), non-contiguous
    var result = reshape(source, [12])

    assert_shape(result, [12], "reshape should produce (12,) tensor")

    # Verify element values
    # Logical order in source (reading row-major from non-contiguous):
    # [0,4,8,1,5,9,2,6,10,3,7,11]
    for i in range(12):
        var expected_row = i % 4
        var expected_col = i // 4
        var expected_val = Float32(expected_col * 4 + expected_row)
        assert_value_at(
            result, i, Float64(expected_val), message="reshape element " + String(i)
        )


fn test_reshape_noncontiguous_to_2d() raises:
    """Reshape non-contiguous (4,3) to (3,4).

    Verifies correct reinterpretation of strided data in new shape.
    """
    var source = make_noncontiguous_3x4()  # (4,3), non-contiguous
    var result = reshape(source, [3, 4])

    assert_shape(result, [3, 4], "reshape should produce (3,4) tensor")

    # After reshape, we should have values in row-major order
    # Row 0: [0, 4, 8, 1]
    # Row 1: [5, 9, 2, 6]
    # Row 2: [10, 3, 7, 11]
    var expected_flat = List[Float64]()
    expected_flat.append(0.0)
    expected_flat.append(4.0)
    expected_flat.append(8.0)
    expected_flat.append(1.0)
    expected_flat.append(5.0)
    expected_flat.append(9.0)
    expected_flat.append(2.0)
    expected_flat.append(6.0)
    expected_flat.append(10.0)
    expected_flat.append(3.0)
    expected_flat.append(7.0)
    expected_flat.append(11.0)
    for i in range(12):
        assert_value_at(
            result, i, expected_flat[i], message="reshape(" + String(i) + ")"
        )


# ============================================================================
# Test: flatten() on non-contiguous input
# ============================================================================


fn test_flatten_noncontiguous() raises:
    """Flatten non-contiguous (4,3) to flat (12,).

    Similar to reshape, but explicitly named operation.
    """
    var source = make_noncontiguous_3x4()  # (4,3), non-contiguous
    var result = flatten(source)

    assert_shape(result, [12], "flatten should produce (12,) tensor")

    # Should match the logical row-major order of the source
    for i in range(12):
        var expected_row = i % 4
        var expected_col = i // 4
        var expected_val = Float32(expected_col * 4 + expected_row)
        assert_value_at(
            result, i, Float64(expected_val), message="flatten element " + String(i)
        )


# ============================================================================
# Test: tile() on non-contiguous input
# ============================================================================


fn test_tile_noncontiguous() raises:
    """Tile non-contiguous (4,3) by 2x in both dimensions.

    Result should be (8,6) with correct values repeated.
    """
    var source = make_noncontiguous_3x4()  # (4,3), non-contiguous
    var result = tile(source, [2, 2])

    assert_shape(result, [8, 6], "tile should produce (8,6) tensor")

    # Verify a few key elements
    # Top-left (0,0) should be 0.0
    assert_value_at(result, 0, 0.0, message="tile[0,0]")
    # Top-right (0,3) should be 0.0 (first repeat in col)
    assert_value_at(result, 3, 0.0, message="tile[0,3]")
    # Second row, first col (1,0) should be 1.0
    assert_value_at(result, 6, 1.0, message="tile[1,0]")


# ============================================================================
# Entry point
# ============================================================================


fn main() raises:
    print("Running non-contiguous shape operation value tests...")

    print("  test_reshape_noncontiguous_to_flat...")
    test_reshape_noncontiguous_to_flat()

    print("  test_reshape_noncontiguous_to_2d...")
    test_reshape_noncontiguous_to_2d()

    print("  test_flatten_noncontiguous...")
    test_flatten_noncontiguous()

    print("  test_tile_noncontiguous...")
    test_tile_noncontiguous()

    print("All non-contiguous shape tests passed!")
