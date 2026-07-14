"""Unit tests for the LayerNorm layer (torch.nn.LayerNorm convention).

Tests cover:
- Construction + validation (positive num_features)
- forward() preserves shape (batch, features)
- parameter collection (2 tensors: gamma, beta)
- Numerical parity with torch.nn.LayerNorm on a fixed ramp input (1e-5), for
  both the default affine (gamma=1, beta=0) and a custom affine
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros
from odyssey.core.layers.layernorm import LayerNorm


def _abs_diff(a: Float64, b: Float64) -> Float64:
    var d = a - b
    if d < 0:
        d = -d
    return d


def _seed_ramp(
    mut t: AnyTensor, count: Int, scale: Float64, off: Float64
) raises:
    """Seed a tensor's flat buffer with value[i] = i*scale + off."""
    for i in range(count):
        t.store[DType.float64](i, Float64(i) * scale + off)


def test_shape() raises:
    """forward preserves (batch, features)."""
    print("Running test_shape...")
    var ln = LayerNorm[DType.float32](4)
    var x = zeros([2, 4], DType.float32)
    var out = ln.forward(x)
    if out.shape()[0] != 2 or out.shape()[1] != 4:
        raise Error("LayerNorm forward must preserve (batch, features)")
    print("  ok (2, 4)")
    print("test_shape PASSED")


def test_reject_bad_sizes() raises:
    """Non-positive num_features must raise."""
    print("Running test_reject_bad_sizes...")
    try:
        var _ = LayerNorm[DType.float32](0)
        raise Error("Should have rejected num_features = 0")
    except e:
        print("  ok rejected num_features = 0")
    print("test_reject_bad_sizes PASSED")


def test_parameter_count() raises:
    """parameters() returns 2 tensors (gamma, beta)."""
    print("Running test_parameter_count...")
    var ln = LayerNorm[DType.float32](4)
    if len(ln.parameters()) != 2:
        raise Error("LayerNorm should expose 2 parameter tensors")
    print("  ok 2 parameter tensors")
    print("test_parameter_count PASSED")


def test_parity_with_pytorch() raises:
    """forward must match torch.nn.LayerNorm on a fixed ramp input to 1e-5.

    Reference values from parity_refs/layernorm_parity_reference.py (batch=2,
    features=4, eps=1e-5). Two cases: default affine (gamma=1, beta=0) and a
    custom affine.
    """
    print("Running test_parity_with_pytorch...")

    var ref_default = List[Float64]()
    ref_default.append(-1.341581162)
    ref_default.append(-0.447193721)
    ref_default.append(0.447193721)
    ref_default.append(1.341581162)
    ref_default.append(-1.341581162)
    ref_default.append(-0.447193721)
    ref_default.append(0.447193721)
    ref_default.append(1.341581162)

    var ref_custom = List[Float64]()
    ref_custom.append(-1.156185872)
    ref_custom.append(-0.608992151)
    ref_custom.append(0.832589011)
    ref_custom.append(3.168557615)
    ref_custom.append(-1.156185872)
    ref_custom.append(-0.608992151)
    ref_custom.append(0.832589011)
    ref_custom.append(3.168557615)

    var x = zeros([2, 4], DType.float64)
    _seed_ramp(x, 8, 0.3, -1.0)

    # Default affine.
    var ln = LayerNorm[DType.float64](4, 1e-5)
    var out_d = ln.forward(x)
    for i in range(8):
        if _abs_diff(out_d.load[DType.float64](i), ref_default[i]) > 1e-5:
            raise Error("LayerNorm default parity mismatch at " + String(i))

    # Custom affine.
    var ln2 = LayerNorm[DType.float64](4, 1e-5)
    _seed_ramp(ln2.gamma, 4, 0.5, 0.75)
    _seed_ramp(ln2.beta, 4, 0.1, -0.15)
    var out_c = ln2.forward(x)
    for i in range(8):
        if _abs_diff(out_c.load[DType.float64](i), ref_custom[i]) > 1e-5:
            raise Error("LayerNorm custom parity mismatch at " + String(i))

    print("  ok matches torch.nn.LayerNorm (default + custom affine) to 1e-5")
    print("test_parity_with_pytorch PASSED")


def main() raises:
    """Run all LayerNorm tests."""
    print("=" * 60)
    print("LayerNorm Layer Test Suite")
    print("=" * 60)
    test_shape()
    test_reject_bad_sizes()
    test_parameter_count()
    test_parity_with_pytorch()
    print("=" * 60)
    print("All LayerNorm tests PASSED")
    print("=" * 60)
