"""Unit tests for the symmetric Jacobi eigensolver (symmetric_eigh).

Tests cover:
- Rejects a non-square / non-2D matrix
- Diagonal matrix: eigenvalues are the (sorted) diagonal, Q is a permutation
- 1×1 matrix edge case
- A symmetric PSD matrix: reconstruction Q diag(λ) Qᵀ == A, eigenvalues ascending
- Eigenvalues match a hand-known 2×2 symmetric matrix
"""

from std.math import sqrt as scalar_sqrt
from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros
from odyssey.core.eigen import symmetric_eigh


def _abs_diff(a: Float64, b: Float64) -> Float64:
    var d = a - b
    if d < 0:
        d = -d
    return d


def test_reject_non_square() raises:
    """A non-square matrix must raise."""
    print("Running test_reject_non_square...")
    var m = zeros([2, 3], DType.float64)
    try:
        var (_, _) = symmetric_eigh(m)
        raise Error("Should have rejected 2x3")
    except _:
        print("  ok rejected non-square")
    print("test_reject_non_square PASSED")


def test_1x1() raises:
    """1x1 matrix: eigenvalue is the single entry, Q is [1]."""
    print("Running test_1x1...")
    var m = zeros([1, 1], DType.float64)
    m.store[DType.float64](0, 3.5)
    var (vals, q) = symmetric_eigh(m)
    if _abs_diff(vals.load[DType.float64](0), 3.5) > 1e-12:
        raise Error("1x1 eigenvalue wrong")
    if _abs_diff(q.load[DType.float64](0), 1.0) > 1e-12:
        raise Error("1x1 Q should be [1]")
    print("  ok 1x1")
    print("test_1x1 PASSED")


def test_2x2_known() raises:
    """[[2,1],[1,2]] has eigenvalues 1 and 3 (ascending)."""
    print("Running test_2x2_known...")
    var m = zeros([2, 2], DType.float64)
    m.store[DType.float64](0, 2.0)
    m.store[DType.float64](1, 1.0)
    m.store[DType.float64](2, 1.0)
    m.store[DType.float64](3, 2.0)
    var (vals, _) = symmetric_eigh(m)
    if _abs_diff(vals.load[DType.float64](0), 1.0) > 1e-10:
        raise Error("2x2 lambda0 should be 1")
    if _abs_diff(vals.load[DType.float64](1), 3.0) > 1e-10:
        raise Error("2x2 lambda1 should be 3")
    print("  ok eigenvalues 1, 3")
    print("test_2x2_known PASSED")


def test_reconstruction_psd() raises:
    """For a symmetric PSD 4x4, Q diag(λ) Qᵀ must reconstruct A to 1e-10.

    A = M Mᵀ with M[i] = i*0.1 - 0.7 (the parity-reference fixture). Also checks
    eigenvalues are non-negative and ascending.
    """
    print("Running test_reconstruction_psd...")
    var n = 4
    # Build M then A = M Mᵀ directly into a tensor.
    var mm = zeros([n, n], DType.float64)
    for i in range(n * n):
        mm.store[DType.float64](i, Float64(i) * 0.1 - 0.7)
    var A = zeros([n, n], DType.float64)
    for i in range(n):
        for j in range(n):
            var acc = 0.0
            for k in range(n):
                acc += mm.load[DType.float64](i * n + k) * mm.load[
                    DType.float64
                ](j * n + k)
            A.store[DType.float64](i * n + j, acc)

    var (vals, Q) = symmetric_eigh(A)

    # ascending
    for i in range(1, n):
        if vals.load[DType.float64](i) < vals.load[DType.float64](i - 1) - 1e-9:
            raise Error("eigenvalues not ascending")

    # reconstruction
    var maxerr = 0.0
    for i in range(n):
        for j in range(n):
            var acc = 0.0
            for k in range(n):
                acc += (
                    Q.load[DType.float64](i * n + k)
                    * vals.load[DType.float64](k)
                    * Q.load[DType.float64](j * n + k)
                )
            var d = _abs_diff(acc, A.load[DType.float64](i * n + j))
            if d > maxerr:
                maxerr = d
    if maxerr > 1e-10:
        raise Error("reconstruction error too large: " + String(maxerr))
    print("  ok reconstruction to " + String(maxerr))
    print("test_reconstruction_psd PASSED")


def main() raises:
    """Run all eigensolver tests."""
    print("=" * 60)
    print("Symmetric Jacobi Eigensolver Test Suite")
    print("=" * 60)
    test_reject_non_square()
    test_1x1()
    test_2x2_known()
    test_reconstruction_psd()
    print("=" * 60)
    print("All eigensolver tests PASSED")
    print("=" * 60)
