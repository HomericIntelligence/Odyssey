"""Typed Tensor[dtype] Strassen helper cores.

Internal module -- not part of the public API.
"""

from std.collections import List
from shared.tensor.tensor import Tensor
from shared.tensor.any_tensor import AnyTensor


def _extract_quadrants_typed[
    dtype: DType
](
    src: Tensor[dtype], n: Int, n_half: Int
) raises -> Tuple[
    Tensor[dtype],
    Tensor[dtype],
    Tensor[dtype],
    Tensor[dtype],
]:
    """Extract four quadrants from a square matrix (typed core)."""
    var q_shape = List[Int]()
    q_shape.append(n_half)
    q_shape.append(n_half)

    var q11 = Tensor[dtype](q_shape)
    var q12 = Tensor[dtype](q_shape)
    var q21 = Tensor[dtype](q_shape)
    var q22 = Tensor[dtype](q_shape)

    var src_ptr = src._data
    var q11_ptr = q11._data
    var q12_ptr = q12._data
    var q21_ptr = q21._data
    var q22_ptr = q22._data

    for i in range(n_half):
        for j in range(n_half):
            q11_ptr.store(i * n_half + j, src_ptr.load(i * n + j))
            q12_ptr.store(i * n_half + j, src_ptr.load(i * n + (j + n_half)))
            q21_ptr.store(i * n_half + j, src_ptr.load((i + n_half) * n + j))
            q22_ptr.store(
                i * n_half + j,
                src_ptr.load((i + n_half) * n + (j + n_half)),
            )

    return (q11^, q12^, q21^, q22^)


def _combine_quadrants_typed[
    dtype: DType
](
    c11: Tensor[dtype],
    c12: Tensor[dtype],
    c21: Tensor[dtype],
    c22: Tensor[dtype],
    n: Int,
    n_half: Int,
) raises -> Tensor[dtype]:
    """Combine four quadrants into a full matrix (typed core)."""
    var c_shape = List[Int]()
    c_shape.append(n)
    c_shape.append(n)
    var result = Tensor[dtype](c_shape)

    var c_ptr = result._data
    var c11_ptr = c11._data
    var c12_ptr = c12._data
    var c21_ptr = c21._data
    var c22_ptr = c22._data

    for i in range(n_half):
        for j in range(n_half):
            c_ptr.store(i * n + j, c11_ptr.load(i * n_half + j))
            c_ptr.store(
                i * n + (j + n_half), c12_ptr.load(i * n_half + j)
            )
            c_ptr.store(
                (i + n_half) * n + j, c21_ptr.load(i * n_half + j)
            )
            c_ptr.store(
                (i + n_half) * n + (j + n_half),
                c22_ptr.load(i * n_half + j),
            )

    return result^


def _matmul_strassen_copy_result[
    dtype: DType
](src: AnyTensor, mut dst: AnyTensor, m: Int, n: Int) raises:
    """Copy Strassen result using typed pointers (zero bitcasts)."""
    var src_typed = src.as_tensor[dtype]()
    var dst_typed = dst.as_tensor[dtype]()
    var src_ptr = src_typed._data
    var dst_ptr = dst_typed._data
    for i in range(m):
        for j in range(n):
            var idx = i * n + j
            dst_ptr.store(idx, src_ptr.load(idx))
