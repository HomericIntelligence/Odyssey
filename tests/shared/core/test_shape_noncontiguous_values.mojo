# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
"""Value-correctness tests for shape ops on non-contiguous inputs. Closes #4086.

Tests that reshape, tile, repeat, broadcast_to, permute, concatenate, and
flatten produce correct element values when called on non-contiguous
(transposed/strided) tensors. Exposes flat-index bugs where _get_float64(i)
ignores _strides and reads wrong memory offsets.

Non-contiguous setup pattern (transpose_view):
    arange(12) reshaped (3,4) -> transpose_view -> shape (4,3), strides [1,4]
    Flat memory: [0,1,2,3,4,5,6,7,8,9,10,11]
    Logical [r,c] = flat_mem[r*1 + c*4] = flat_mem[r + 4c]

    Row-major read of logical (4,3) tensor:
      [0,4,8,  1,5,9,  2,6,10,  3,7,11]
"""

# Import AnyTensor and shape operations
from shared.tensor.any_tensor import AnyTensor, arange
from shared.core.shape import (
    reshape,
    flatten,
    concatenate,
    permute,
    broadcast_to,
    tile,
    repeat,
)
from shared.core.matrix import transpose_view

# Import test helpers
from tests.shared.conftest import (
    assert_value_at,
    assert_dim,
    assert_numel,
    assert_shape,
)


# ============================================================================
# Helper: create standard non-contiguous test tensor
# ============================================================================


def make_noncontiguous_4x3() raises -> AnyTensor:
    """Return a (4,3) non-contiguous tensor via transpose_view of arange(12) reshaped (3,4).

    Flat memory order: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    Strides after transpose_view: [1, 4] (non-C-order, is_contiguous() == False)
    Logical element [r, c] = c*4 + r

    Row-major traversal of logical shape (4,3):
      [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
    """
    var flat = arange(0.0, 12.0, 1.0, DType.float32)
    var shape: List[Int] = [3, 4]
    var t2d = flat.reshape(shape)
    return transpose_view(t2d)  # shape (4,3), non-contiguous


# ============================================================================
# Tests
# ============================================================================


def test_reshape_noncontiguous_values() raises:
    """Verify reshape() on non-contiguous (4,3) input produces correct flat 1D values.

    Expected: row-major read of logical (4,3) = [0,4,8, 1,5,9, 2,6,10, 3,7,11]
    """
    var t_nc = make_noncontiguous_4x3()
    var new_shape: List[Int] = [12]
    var result = reshape(t_nc, new_shape)

    assert_dim(result, 1, "reshape: result should be 1D")
    assert_numel(result, 12, "reshape: result should have 12 elements")

    # Expected: row-major traversal of logical (4,3) tensor
    var expected: List[Float64] = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
    for i in range(12):
        assert_value_at(result, i, expected[i])


def test_flatten_noncontiguous_values() raises:
    """Verify flatten() on non-contiguous (4,3) input produces correct 1D values.

    Expected: same as reshape to 1D = [0,4,8, 1,5,9, 2,6,10, 3,7,11]
    """
    var t_nc = make_noncontiguous_4x3()
    var result = flatten(t_nc)

    assert_dim(result, 1, "flatten: result should be 1D")
    assert_numel(result, 12, "flatten: result should have 12 elements")

    var expected: List[Float64] = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
    for i in range(12):
        assert_value_at(result, i, expected[i])


def test_permute_noncontiguous_values() raises:
    """Verify permute([1,0]) on non-contiguous (4,3) input gives correct (3,4) values.

    Permute [1,0] on logical (4,3) transposes back to (3,4).
    Result[r,c] = t_nc[c,r] = c*4 + r = original arange(12) row-major order.
    Expected flat: [0,1,2,3, 4,5,6,7, 8,9,10,11]
    """
    var t_nc = make_noncontiguous_4x3()
    var dims: List[Int] = [1, 0]
    var result = permute(t_nc, dims)

    var result_shape = result.shape()
    if result_shape[0] != 3 or result_shape[1] != 4:
        raise Error(
            "permute: expected shape (3,4), got ("
            + String(result_shape[0])
            + ","
            + String(result_shape[1])
            + ")"
        )
    assert_numel(result, 12, "permute: result should have 12 elements")

    # Result should be original arange(12) in row-major (3,4) order
    for i in range(12):
        assert_value_at(result, i, Float64(i))


def test_concatenate_noncontiguous_values() raises:
    """Verify concatenate() of two non-contiguous (4,3) tensors along axis=0 gives correct (8,3) values.

    Each block should be the row-major read of logical (4,3):
      rows 0-3: [0,4,8], [1,5,9], [2,6,10], [3,7,11]
    Total 24 elements: first 12 from t_nc, second 12 same.
    """
    var t_nc = make_noncontiguous_4x3()
    var tensors: List[AnyTensor] = [t_nc, t_nc]
    var result = concatenate(tensors, axis=0)

    var result_shape = result.shape()
    if result_shape[0] != 8 or result_shape[1] != 3:
        raise Error(
            "concatenate: expected shape (8,3), got ("
            + String(result_shape[0])
            + ","
            + String(result_shape[1])
            + ")"
        )
    assert_numel(result, 24, "concatenate: result should have 24 elements")

    # Each half: row-major read of logical (4,3) = [0,4,8, 1,5,9, 2,6,10, 3,7,11]
    var expected_half: List[Float64] = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
    for half in range(2):
        for i in range(12):
            assert_value_at(result, half * 12 + i, expected_half[i])


def test_broadcast_to_noncontiguous_values() raises:
    """Verify broadcast_to() on non-contiguous (3,1) column tensor gives correct (3,4) values.

    Setup: arange(3) reshaped (1,3) -> transpose_view -> (3,1), strides [1,3]
    Logical values: column [0], [1], [2]
    broadcast_to (3,4): each row i repeated 4 times
    Expected flat: [0,0,0,0, 1,1,1,1, 2,2,2,2]
    """
    var flat = arange(0.0, 3.0, 1.0, DType.float32)
    var row_shape: List[Int] = [1, 3]
    var row = flat.reshape(row_shape)
    var col_nc = transpose_view(row)  # shape (3,1), non-contiguous, strides [1,3]

    var target_shape: List[Int] = [3, 4]
    var result = broadcast_to(col_nc, target_shape)

    var result_shape = result.shape()
    if result_shape[0] != 3 or result_shape[1] != 4:
        raise Error(
            "broadcast_to: expected shape (3,4), got ("
            + String(result_shape[0])
            + ","
            + String(result_shape[1])
            + ")"
        )
    assert_numel(result, 12, "broadcast_to: result should have 12 elements")

    # Each row i should have value i repeated 4 times
    for row_idx in range(3):
        for col_idx in range(4):
            var flat_idx = row_idx * 4 + col_idx
            assert_value_at(result, flat_idx, Float64(row_idx))


def test_tile_noncontiguous_values() raises:
    """Verify tile() on non-contiguous (2,3) input tiled [1,2] gives correct (2,6) values.

    Setup: arange(6) reshaped (3,2) -> transpose_view -> (2,3), strides [1,2]
    Logical (2,3): [r,c] = flat_mem[r + 2c]
      row 0: [0,2,4]  row 1: [1,3,5]
    tile([1,2]) -> (2,6) by doubling columns:
      row 0: [0,2,4,0,2,4]  row 1: [1,3,5,1,3,5]
    Expected flat: [0,2,4,0,2,4, 1,3,5,1,3,5]
    """
    var flat = arange(0.0, 6.0, 1.0, DType.float32)
    var base_shape: List[Int] = [3, 2]
    var t2d = flat.reshape(base_shape)
    var t_nc = transpose_view(t2d)  # shape (2,3), strides [1,2], non-contiguous

    var reps: List[Int] = [1, 2]
    var result = tile(t_nc, reps)

    var result_shape = result.shape()
    if result_shape[0] != 2 or result_shape[1] != 6:
        raise Error(
            "tile: expected shape (2,6), got ("
            + String(result_shape[0])
            + ","
            + String(result_shape[1])
            + ")"
        )
    assert_numel(result, 12, "tile: result should have 12 elements")

    # Row 0: [0,2,4,0,2,4], Row 1: [1,3,5,1,3,5]
    var expected: List[Float64] = [0, 2, 4, 0, 2, 4, 1, 3, 5, 1, 3, 5]
    for i in range(12):
        assert_value_at(result, i, expected[i])


def test_repeat_noncontiguous_values() raises:
    """Verify repeat() on non-contiguous (2,3) input repeated 2x along axis=0 gives correct (4,3) values.

    Same non-contiguous (2,3) as tile test (strides [1,2]):
      row 0: [0,2,4]  row 1: [1,3,5]
    repeat(2, axis=0) -> each row appears twice:
      [0,2,4], [0,2,4], [1,3,5], [1,3,5]
    Expected flat: [0,2,4,0,2,4, 1,3,5,1,3,5]
    """
    var flat = arange(0.0, 6.0, 1.0, DType.float32)
    var base_shape: List[Int] = [3, 2]
    var t2d = flat.reshape(base_shape)
    var t_nc = transpose_view(t2d)  # shape (2,3), strides [1,2], non-contiguous

    var result = repeat(t_nc, 2, 0)

    var result_shape = result.shape()
    if result_shape[0] != 4 or result_shape[1] != 3:
        raise Error(
            "repeat: expected shape (4,3), got ("
            + String(result_shape[0])
            + ","
            + String(result_shape[1])
            + ")"
        )
    assert_numel(result, 12, "repeat: result should have 12 elements")

    # Rows 0,1 are repeats of t_nc row 0: [0,2,4]; rows 2,3 of t_nc row 1: [1,3,5]
    var expected: List[Float64] = [0, 2, 4, 0, 2, 4, 1, 3, 5, 1, 3, 5]
    for i in range(12):
        assert_value_at(result, i, expected[i])


# ============================================================================
# Main test runner
# ============================================================================


def main() raises:
    """Run value-correctness tests for shape ops on non-contiguous inputs."""
    print("Running shape op value-correctness tests on non-contiguous inputs...")

    try:
        test_reshape_noncontiguous_values()
        print("  PASS: reshape() non-contiguous value correctness")
    except e:
        print("  FAIL: reshape() FAILED:", String(e))

    try:
        test_flatten_noncontiguous_values()
        print("  PASS: flatten() non-contiguous value correctness")
    except e:
        print("  FAIL: flatten() FAILED:", String(e))

    try:
        test_permute_noncontiguous_values()
        print("  PASS: permute() non-contiguous value correctness")
    except e:
        print("  FAIL: permute() FAILED:", String(e))

    try:
        test_concatenate_noncontiguous_values()
        print("  PASS: concatenate() non-contiguous value correctness")
    except e:
        print("  FAIL: concatenate() FAILED:", String(e))

    try:
        test_broadcast_to_noncontiguous_values()
        print("  PASS: broadcast_to() non-contiguous value correctness")
    except e:
        print("  FAIL: broadcast_to() FAILED:", String(e))

    try:
        test_tile_noncontiguous_values()
        print("  PASS: tile() non-contiguous value correctness")
    except e:
        print("  FAIL: tile() FAILED:", String(e))

    try:
        test_repeat_noncontiguous_values()
        print("  PASS: repeat() non-contiguous value correctness")
    except e:
        print("  FAIL: repeat() FAILED:", String(e))

    print("Shape op non-contiguous value-correctness tests completed.")
