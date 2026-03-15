"""Regression tests for concatenate() with non-contiguous tensors.

Issue #4083: The axis-0 concatenate() path had a flat-index bug in its
non-contiguous else-branch that ignored strides, producing wrong results
when concatenating non-contiguous tensors (e.g., transposed views).

These tests verify:
- Contiguous concatenation still works correctly after the fix.
- Non-contiguous tensors are read with stride-aware byte offsets.
- Incompatible shapes raise the expected error.
"""

from shared.core import ExTensor, zeros, concatenate
from tests.shared.conftest import (
    assert_numel,
    assert_dim,
    assert_value_at,
    assert_true,
)


fn test_concat_contiguous_axis0() raises:
    """Baseline: concatenating contiguous tensors along axis=0 still works.

    Creates two 2x3 float32 tensors filled with 0.5 and 1.0 respectively,
    concatenates them along axis=0, and verifies the result shape and values.
    """
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)

    var a = zeros(shape, DType.float32)
    for i in range(6):
        a._set_float64(i, 0.5)

    var b = zeros(shape, DType.float32)
    for i in range(6):
        b._set_float64(i, 1.0)

    var tensors: List[ExTensor] = []
    tensors.append(a)
    tensors.append(b)

    var result = concatenate(tensors, axis=0)

    assert_dim(result, 2, "Result should be 2D")
    assert_numel(result, 12, "Result should have 12 elements (4x3)")

    # First half from a (0.5)
    for i in range(6):
        assert_value_at(result, i, 0.5, 1e-6, "First half should be 0.5")

    # Second half from b (1.0)
    for i in range(6):
        assert_value_at(result, 6 + i, 1.0, 1e-6, "Second half should be 1.0")


fn test_concat_noncontiguous_axis0() raises:
    """Regression: concatenating a non-contiguous tensor along axis=0.

    Creates a 2x3 float32 tensor with values [0.0, 0.5, 1.0, -0.5, -1.0, 1.5],
    then manually sets strides to [1, 2] (column-major) so it becomes
    non-contiguous. Concatenates it with a contiguous 2x3 tensor filled
    with 0.0 along axis=0 and verifies that the stride-aware path reads
    the correct element values instead of the flat-indexed (wrong) ones.

    Column-major stride [1, 2] on a 2x3 shape means element (row, col) lives
    at byte offset (row*1 + col*2) * dtype_size. The C-order flat scan visits
    elements in row-major order: (0,0), (0,1), (0,2), (1,0), (1,1), (1,2),
    mapping to memory positions 0, 2, 4, 1, 3, 5 (element indices).
    """
    # Allocate a 2x3 float32 tensor and fill flat memory with known values.
    # Memory layout (element indices 0..5): 0.0, 0.5, 1.0, -0.5, -1.0, 1.5
    var shape = List[Int]()
    shape.append(2)
    shape.append(3)
    var t = zeros(shape, DType.float32)
    t._set_float64(0, 0.0)
    t._set_float64(1, 0.5)
    t._set_float64(2, 1.0)
    t._set_float64(3, -0.5)
    t._set_float64(4, -1.0)
    t._set_float64(5, 1.5)

    # Override strides to column-major [1, 2]: non-contiguous
    t._strides[0] = 1
    t._strides[1] = 2
    assert_true(not t.is_contiguous(), "Tensor should be non-contiguous after stride override")

    # Contiguous pad tensor filled with 0.0
    var pad = zeros(shape, DType.float32)
    var tensors: List[ExTensor] = []
    tensors.append(t)
    tensors.append(pad)

    var result = concatenate(tensors, axis=0)

    assert_dim(result, 2, "Result should be 2D")
    assert_numel(result, 12, "Result should have 12 elements")

    # With column-major strides [1, 2] and shape (2, 3), C-order traversal:
    #   flat i=0: coords (0,0) -> mem[0*1 + 0*2] = mem[0] = 0.0
    #   flat i=1: coords (0,1) -> mem[0*1 + 1*2] = mem[2] = 1.0
    #   flat i=2: coords (0,2) -> mem[0*1 + 2*2] = mem[4] = -1.0
    #   flat i=3: coords (1,0) -> mem[1*1 + 0*2] = mem[1] = 0.5
    #   flat i=4: coords (1,1) -> mem[1*1 + 1*2] = mem[3] = -0.5
    #   flat i=5: coords (1,2) -> mem[1*1 + 2*2] = mem[5] = 1.5
    assert_value_at(result, 0, 0.0, 1e-6, "result[0] should be 0.0")
    assert_value_at(result, 1, 1.0, 1e-6, "result[1] should be 1.0")
    assert_value_at(result, 2, -1.0, 1e-6, "result[2] should be -1.0")
    assert_value_at(result, 3, 0.5, 1e-6, "result[3] should be 0.5")
    assert_value_at(result, 4, -0.5, 1e-6, "result[4] should be -0.5")
    assert_value_at(result, 5, 1.5, 1e-6, "result[5] should be 1.5")

    # Pad half should all be 0.0
    for i in range(6):
        assert_value_at(result, 6 + i, 0.0, 1e-6, "Pad half should be 0.0")


fn test_concat_incompatible_shapes_raises() raises:
    """Error path: concatenating tensors with incompatible shapes raises Error."""
    var shape_a = List[Int]()
    shape_a.append(2)
    shape_a.append(3)
    var a = zeros(shape_a, DType.float32)

    var shape_b = List[Int]()
    shape_b.append(2)
    shape_b.append(4)  # Different column count — incompatible for axis=0 concat
    var b = zeros(shape_b, DType.float32)

    var tensors: List[ExTensor] = []
    tensors.append(a)
    tensors.append(b)

    var raised = False
    try:
        _ = concatenate(tensors, axis=0)
    except:
        raised = True

    assert_true(raised, "Should raise Error for incompatible shapes")


fn main() raises:
    """Run all concatenate non-contiguous regression tests."""
    print("Running concatenate non-contiguous regression tests...")

    try:
        test_concat_contiguous_axis0()
        print("  PASS test_concat_contiguous_axis0")
    except e:
        print("  FAIL test_concat_contiguous_axis0:", String(e))

    try:
        test_concat_noncontiguous_axis0()
        print("  PASS test_concat_noncontiguous_axis0")
    except e:
        print("  FAIL test_concat_noncontiguous_axis0:", String(e))

    try:
        test_concat_incompatible_shapes_raises()
        print("  PASS test_concat_incompatible_shapes_raises")
    except e:
        print("  FAIL test_concat_incompatible_shapes_raises:", String(e))

    print("Done.")
