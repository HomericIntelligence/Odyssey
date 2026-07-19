"""Unit tests for the permutation-equivariant Deep Sets block.

Tests cover:
- Construction + shape mapping ([batch, set, dim] -> [batch, set, out]).
- Parameter collection (3 tensors: lam, gam, bias).
- Input validation (rejects non-positive dims; rejects non-rank-3 / wrong last
  dim input).
- Numerical parity with a PyTorch reference (per-element Lambda + sum-pooled
  Gamma + bias, ReLU) on fixed ramp weights/input (tolerance 1e-5).
- THE DEFINING PROPERTY: permutation equivariance — permuting the set elements
  of the input must permute the output rows identically, to tight tolerance.
- Package-path import: `from odyssey.core.layers import DeepSetsEquivariant`
  must resolve (guards docstring-without-export regressions).
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros
from odyssey.core.layers.deepsets import DeepSetsEquivariant

# Package-path import: MUST resolve via the layers package __init__ export.
from odyssey.core.layers import DeepSetsEquivariant as DeepSetsFromPackage


def test_shape_mapping() raises:
    """Maps [batch, set, dim] -> [batch, set, out]."""
    print("Running test_shape_mapping...")
    var layer = DeepSetsEquivariant[DType.float32](6, 5)
    var x = zeros([2, 4, 6], DType.float32)
    var y = layer.forward(x)
    if y.shape()[0] != 2 or y.shape()[1] != 4 or y.shape()[2] != 5:
        raise Error("DeepSets must map [2,4,6] -> [2,4,5]")
    print("  ok shape [2,4,6] -> [2,4,5]")
    print("test_shape_mapping PASSED")


def test_parameter_count() raises:
    """`parameters()` returns 3 tensors (lam, gam, bias)."""
    print("Running test_parameter_count...")
    var layer = DeepSetsEquivariant[DType.float32](6, 5)
    var params = layer.parameters()
    if len(params) != 3:
        raise Error("DeepSets should expose 3 parameter tensors")
    print("  ok 3 parameter tensors")
    print("test_parameter_count PASSED")


def test_reject_bad_dims() raises:
    """Non-positive dim / out_features must raise."""
    print("Running test_reject_bad_dims...")
    try:
        var _ = DeepSetsEquivariant[DType.float32](0, 5)
        raise Error("Should have rejected dim = 0")
    except _:
        print("  ok rejected dim = 0")
    try:
        var _ = DeepSetsEquivariant[DType.float32](6, 0)
        raise Error("Should have rejected out_features = 0")
    except _:
        print("  ok rejected out_features = 0")
    print("test_reject_bad_dims PASSED")


def test_reject_bad_input_rank() raises:
    """Non-rank-3 input and wrong last dim must raise."""
    print("Running test_reject_bad_input_rank...")
    var layer = DeepSetsEquivariant[DType.float32](6, 5)
    try:
        var bad = zeros([2, 6], DType.float32)  # rank-2
        var _ = layer.forward(bad)
        raise Error("Should have rejected rank-2 input")
    except _:
        print("  ok rejected rank-2 input")
    try:
        var wrong = zeros([2, 4, 7], DType.float32)  # last dim != 6
        var _ = layer.forward(wrong)
        raise Error("Should have rejected last-dim mismatch")
    except _:
        print("  ok rejected wrong last dim")
    print("test_reject_bad_input_rank PASSED")


def test_package_path_import() raises:
    """`DeepSetsEquivariant` must be importable from the layers package."""
    print("Running test_package_path_import...")
    var layer = DeepSetsFromPackage[DType.float32](6, 5)
    var x = zeros([2, 4, 6], DType.float32)
    var y = layer.forward(x)
    if y.shape()[2] != 5:
        raise Error("package-path import produced wrong shape")
    print("  ok imported via odyssey.core.layers package path")
    print("test_package_path_import PASSED")


def test_parity_with_pytorch() raises:
    """Forward must match a PyTorch reference to 1e-5.

    Fixed ramp weights/input and the reference output are transcribed from
    parity_refs/deepsets_parity_reference.py (torch, float64, batch=2,
    set_size=4, dim=6, out=5; sum pool, ReLU). Odyssey lam/gam use (dim, out)
    layout, matching the reference's torch matmul `X @ W`.
    """
    print("Running test_parity_with_pytorch...")

    # Reference output (flattened [B*S*O] = 2*4*5 = 40), from the generator.
    var ref_vals = List[Float64]()
    ref_vals.append(0.045000000000000005)
    ref_vals.append(0.040220000000000006)
    ref_vals.append(0.03544)
    ref_vals.append(0.03065999999999999)
    ref_vals.append(0.02587999999999999)
    ref_vals.append(0.03600000000000001)
    ref_vals.append(0.034820000000000004)
    ref_vals.append(0.033639999999999996)
    ref_vals.append(0.032459999999999996)
    ref_vals.append(0.03127999999999999)
    ref_vals.append(0.027000000000000017)
    ref_vals.append(0.02942000000000002)
    ref_vals.append(0.03184000000000001)
    ref_vals.append(0.03426)
    ref_vals.append(0.03667999999999999)
    ref_vals.append(0.018000000000000023)
    ref_vals.append(0.024020000000000017)
    ref_vals.append(0.030040000000000008)
    ref_vals.append(0.03606000000000001)
    ref_vals.append(0.04207999999999999)
    ref_vals.append(0.0)
    ref_vals.append(0.0)
    ref_vals.append(0.03688000000000001)
    ref_vals.append(0.08682000000000001)
    ref_vals.append(0.13676000000000002)
    ref_vals.append(0.0)
    ref_vals.append(0.0)
    ref_vals.append(0.035080000000000014)
    ref_vals.append(0.08862000000000003)
    ref_vals.append(0.14216000000000004)
    ref_vals.append(0.0)
    ref_vals.append(0.0)
    ref_vals.append(0.03328000000000002)
    ref_vals.append(0.09042000000000003)
    ref_vals.append(0.14756000000000002)
    ref_vals.append(0.0)
    ref_vals.append(0.0)
    ref_vals.append(0.03148000000000002)
    ref_vals.append(0.09222000000000002)
    ref_vals.append(0.15296)

    var layer = DeepSetsEquivariant[DType.float64](6, 5)
    # lam[i] = i*0.01 - 0.15 (dim*out = 30 entries)
    for i in range(30):
        layer.lam.store[DType.float64](i, Float64(i) * 0.01 - 0.15)
    # gam[i] = i*0.007 - 0.10
    for i in range(30):
        layer.gam.store[DType.float64](i, Float64(i) * 0.007 - 0.10)
    # bias[i] = i*0.02 - 0.04
    for i in range(5):
        layer.bias.store[DType.float64](i, Float64(i) * 0.02 - 0.04)

    # X[i] = i*0.01 - 0.2 (batch*set*dim = 48 entries)
    var x = zeros([2, 4, 6], DType.float64)
    for i in range(48):
        x.store[DType.float64](i, Float64(i) * 0.01 - 0.2)

    var y = layer.forward(x)
    for i in range(40):
        var d = y.load[DType.float64](i) - ref_vals[i]
        if d < 0:
            d = -d
        if d > 1e-5:
            raise Error("DeepSets parity mismatch at index " + String(i))
    print("  ok matches PyTorch reference to 1e-5")
    print("test_parity_with_pytorch PASSED")


def test_permutation_equivariance() raises:
    """DEFINING PROPERTY: permuting set elements permutes output rows identically.

    Build a single-batch input, run forward to get `y`. Then apply a nontrivial
    permutation of the set axis to the SAME input, run forward again, and assert
    the second output equals `y` with its rows permuted the same way, to tight
    tolerance. (Uses float64 so the equality is exact up to fp round-off, not a
    loose shape check.)
    """
    print("Running test_permutation_equivariance...")

    var S = 4
    var D = 6
    var O = 5
    var layer = DeepSetsEquivariant[DType.float64](D, O)
    # Nontrivial ramp weights so the map is not degenerate.
    for i in range(D * O):
        layer.lam.store[DType.float64](i, Float64(i) * 0.013 - 0.08)
    for i in range(D * O):
        layer.gam.store[DType.float64](i, Float64(i) * 0.009 - 0.05)
    for i in range(O):
        layer.bias.store[DType.float64](i, Float64(i) * 0.01 - 0.02)

    # Base input [1, S, D], ramp values.
    var x = zeros([1, S, D], DType.float64)
    for i in range(S * D):
        x.store[DType.float64](i, Float64(i) * 0.017 - 0.25)

    var y = layer.forward(x)  # [1, S, O]

    # Nontrivial permutation of the set axis: perm = [2, 0, 3, 1].
    var perm = List[Int]()
    perm.append(2)
    perm.append(0)
    perm.append(3)
    perm.append(1)

    var xp = zeros([1, S, D], DType.float64)
    for new_pos in range(S):
        var src = perm[new_pos]
        for d in range(D):
            xp.store[DType.float64](
                new_pos * D + d, x.load[DType.float64](src * D + d)
            )

    var yp = layer.forward(xp)  # [1, S, O]

    # Equivariance: yp[new_pos] must equal y[perm[new_pos]] (rows permuted the
    # same way the input rows were).
    for new_pos in range(S):
        var src = perm[new_pos]
        for o in range(O):
            var got = yp.load[DType.float64](new_pos * O + o)
            var want = y.load[DType.float64](src * O + o)
            var d = got - want
            if d < 0:
                d = -d
            if d > 1e-9:
                raise Error(
                    "equivariance broken at set pos "
                    + String(new_pos)
                    + ", out "
                    + String(o)
                )
    print("  ok permuting input set permutes output rows identically")
    print("test_permutation_equivariance PASSED")


def main() raises:
    """Run all DeepSets equivariant-block tests."""
    print("=" * 60)
    print("DeepSetsEquivariant (permutation-equivariant block) Test Suite")
    print("=" * 60)
    test_shape_mapping()
    test_parameter_count()
    test_reject_bad_dims()
    test_reject_bad_input_rank()
    test_package_path_import()
    test_parity_with_pytorch()
    test_permutation_equivariance()
    print("=" * 60)
    print("All DeepSetsEquivariant tests PASSED")
    print("=" * 60)
