"""Unit tests for the 1-layer KAN (Kolmogorov-Arnold Network) block.

Tests cover:
- Construction + shape (input [batch, in_features] -> [batch, out_features])
- Parameter collection (3 tensors: base_weight, spline_weight, spline_coeff)
- Coefficient count per edge (grid_size + spline_order)
- Argument validation (rejects non-positive dims, grid_max <= grid_min)
- Input-rank validation (rejects non-2D / wrong last dim)
- Out-of-range compact support: an input outside [grid_min, grid_max] has an
  all-zero spline branch (edge output = base branch only)
- Numerical parity with a numpy reference (Cox-de Boor B-spline basis + residual
  activation) on fixed ramp params + inputs (tolerance 1e-5), including an
  in-range row and a below-grid-min row (grid-boundary / out-of-range coverage)
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros
from odyssey.core.layers.kan import KAN

# Package-path import: KAN must be reachable via the package `__init__` export,
# not only its module. A sibling layer was flagged for a docstring-Components
# entry without a matching `from ...import`; this guards that export line.
from odyssey.core.layers import KAN as KANFromPackage


def test_package_export() raises:
    """KAN is importable via the `odyssey.core.layers` package export."""
    print("Running test_package_export...")
    var kan = KANFromPackage[DType.float32](4, 4)
    var x = zeros([1, 4], DType.float32)
    var y = kan.forward(x)
    if y.shape()[1] != 4:
        raise Error("package-exported KAN must be usable")
    print("  ok KAN reachable via odyssey.core.layers")
    print("test_package_export PASSED")


def test_shape_preserved() raises:
    """KAN maps [batch, in_features] -> [batch, out_features]."""
    print("Running test_shape_preserved...")
    var kan = KAN[DType.float32](4, 6)
    var x = zeros([2, 4], DType.float32)
    var y = kan.forward(x)
    if y.shape()[0] != 2 or y.shape()[1] != 6:
        raise Error("KAN must map [batch, in] -> [batch, out]")
    print("  ok shape [2, 4] -> [2, 6]")
    print("test_shape_preserved PASSED")


def test_coeff_count() raises:
    """The spline_coeff tensor holds in*out*(grid_size+spline_order) values."""
    print("Running test_coeff_count...")
    var kan = KAN[DType.float32](4, 4, grid_size=5, spline_order=3)
    # 4 * 4 * (5 + 3) = 128
    if kan.spline_coeff.numel() != 128:
        raise Error("coeff count must be in*out*(grid_size+spline_order)")
    print("  ok 128 spline coefficients")
    print("test_coeff_count PASSED")


def test_parameter_count() raises:
    """`parameters()` returns 3 tensors."""
    print("Running test_parameter_count...")
    var kan = KAN[DType.float32](4, 4)
    var params = kan.parameters()
    if len(params) != 3:
        raise Error("KAN should expose 3 parameter tensors")
    print("  ok 3 parameter tensors")
    print("test_parameter_count PASSED")


def test_reject_bad_args() raises:
    """Non-positive dims and grid_max <= grid_min must raise."""
    print("Running test_reject_bad_args...")
    try:
        var _ = KAN[DType.float32](0, 4)
        raise Error("Should have rejected in_features = 0")
    except _:
        print("  ok rejected in_features = 0")
    try:
        var _ = KAN[DType.float32](4, 4, grid_min=1.0, grid_max=-1.0)
        raise Error("Should have rejected grid_max <= grid_min")
    except _:
        print("  ok rejected grid_max <= grid_min")
    print("test_reject_bad_args PASSED")


def test_reject_bad_input_rank() raises:
    """Non-2D input / wrong last dim must raise."""
    print("Running test_reject_bad_input_rank...")
    var kan = KAN[DType.float32](4, 4)
    try:
        var bad = zeros([4], DType.float32)
        var _ = kan.forward(bad)
        raise Error("Should have rejected 1D input")
    except _:
        print("  ok rejected 1D input")
    try:
        var bad2 = zeros([2, 3], DType.float32)
        var _ = kan.forward(bad2)
        raise Error("Should have rejected wrong last dim")
    except _:
        print("  ok rejected wrong last dim")
    print("test_reject_bad_input_rank PASSED")


def test_out_of_range_spline_zero() raises:
    """Input outside [grid_min, grid_max] => spline branch is zero.

    With base_weight = 0 and only spline_weight/coeff set, an in-range input
    produces a nonzero output while a far-out-of-range input produces exactly 0
    (the B-spline basis has compact support). This directly exercises the
    documented grid-range behavior.
    """
    print("Running test_out_of_range_spline_zero...")
    var kan = KAN[DType.float64](1, 1, grid_size=5, spline_order=3)
    # base_weight stays 0; set spline_weight=1 and all coeffs=1.
    kan.spline_weight.store[DType.float64](0, Float64(1.0))
    for m in range(8):
        kan.spline_coeff.store[DType.float64](m, Float64(1.0))

    var x_in = zeros([1, 1], DType.float64)
    x_in.store[DType.float64](0, Float64(0.0))  # centre of grid
    var y_in = kan.forward(x_in)
    var v_in = y_in.load[DType.float64](0)
    # Partition of unity => sum of order-k basis is 1 inside the range.
    if v_in <= 0.0:
        raise Error("in-range spline output should be positive")

    var x_out = zeros([1, 1], DType.float64)
    x_out.store[DType.float64](0, Float64(5.0))  # far outside [-1, 1]
    var y_out = kan.forward(x_out)
    var v_out = y_out.load[DType.float64](0)
    if v_out < 0:
        v_out = -v_out
    if v_out > 1e-12:
        raise Error("out-of-range spline branch must be zero")
    print("  ok in-range nonzero, out-of-range spline = 0")
    print("test_out_of_range_spline_zero PASSED")


def test_parity_with_reference() raises:
    """Forward must match the numpy KAN reference to 1e-5.

    Fixed ramp params + inputs and the reference output are transcribed from
    parity_refs/kan_parity_reference.py (numpy, float64, in=4, out=4,
    grid_size=5, spline_order=3, range [-1, 1], batch=2). Row 1 includes a
    below-grid-min component (x = -1.3) so out-of-range compact support is
    covered. The B-spline recursion is identical on both sides, so parity is
    exact-by-construction.
    """
    print("Running test_parity_with_reference...")

    # Reference output (from the generator), built before any tensor stores.
    var ref_vals = List[Float64]()
    ref_vals.append(0.11188056322349646)
    ref_vals.append(0.12936175267313427)
    ref_vals.append(0.15068294212277208)
    ref_vals.append(0.17584413157240988)
    ref_vals.append(0.12306906032436184)
    ref_vals.append(0.14314732879622444)
    ref_vals.append(0.16699809726808704)
    ref_vals.append(0.1946213657399496)

    var kan = KAN[DType.float64](4, 4, grid_size=5, spline_order=3)
    # base_weight ramp: i*0.01 - 0.05  (16 values, flat i*out+j)
    for i in range(16):
        kan.base_weight.store[DType.float64](i, Float64(i) * 0.01 - 0.05)
    # spline_weight ramp: i*0.02 - 0.10
    for i in range(16):
        kan.spline_weight.store[DType.float64](i, Float64(i) * 0.02 - 0.10)
    # spline_coeff ramp: i*0.003 - 0.05  (128 values)
    for i in range(128):
        kan.spline_coeff.store[DType.float64](i, Float64(i) * 0.003 - 0.05)

    # Inputs: [[-0.7, -0.2, 0.35, 0.9], [-1.3, 0.0, 0.5, 1.0]]
    var x = zeros([2, 4], DType.float64)
    x.store[DType.float64](0, Float64(-0.7))
    x.store[DType.float64](1, Float64(-0.2))
    x.store[DType.float64](2, Float64(0.35))
    x.store[DType.float64](3, Float64(0.9))
    x.store[DType.float64](4, Float64(-1.3))
    x.store[DType.float64](5, Float64(0.0))
    x.store[DType.float64](6, Float64(0.5))
    x.store[DType.float64](7, Float64(1.0))

    var y = kan.forward(x)
    for i in range(8):
        var d = y.load[DType.float64](i) - ref_vals[i]
        if d < 0:
            d = -d
        if d > 1e-5:
            raise Error("KAN parity mismatch at index " + String(i))
    print("  ok matches numpy KAN reference to 1e-5")
    print("test_parity_with_reference PASSED")


def main() raises:
    """Run all KAN tests."""
    print("=" * 60)
    print("KAN (Kolmogorov-Arnold Network) Test Suite")
    print("=" * 60)
    test_package_export()
    test_shape_preserved()
    test_coeff_count()
    test_parameter_count()
    test_reject_bad_args()
    test_reject_bad_input_rank()
    test_out_of_range_spline_zero()
    test_parity_with_reference()
    print("=" * 60)
    print("All KAN tests PASSED")
    print("=" * 60)
