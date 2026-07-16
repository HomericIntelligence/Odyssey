"""Unit tests for the Transformer FeedForward (FFN) block.

Tests cover:
- Construction + shape preservation (input (..., d_model) -> (..., d_model))
- Default inner dimension (d_ff defaults to 4 * d_model)
- Parameter collection (4 tensors: fc1.weight/bias, fc2.weight/bias)
- d_model validation (rejects non-positive d_model)
- Numerical parity with a PyTorch reference (Linear -> exact GELU -> Linear)
  on fixed ramp weights + input (tolerance 1e-5)
- ReLU variant (use_gelu=False) runs
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros
from odyssey.core.layers.feedforward import FeedForward


def test_shape_preserved() raises:
    """FFN maps (batch, d_model) -> (batch, d_model)."""
    print("Running test_shape_preserved...")
    var ffn = FeedForward[DType.float32](8, 32)
    var x = zeros([2, 8], DType.float32)
    var y = ffn.forward(x)
    if y.shape()[0] != 2 or y.shape()[1] != 8:
        raise Error("FFN must preserve (batch, d_model) shape")
    print("  ok shape preserved (2, 8)")
    print("test_shape_preserved PASSED")


def test_default_inner_dim() raises:
    """d_ff defaults to 4 * d_model when not given."""
    print("Running test_default_inner_dim...")
    var ffn = FeedForward[DType.float32](16)
    if ffn.d_ff != 64:
        raise Error("default d_ff should be 4 * d_model")
    print("  ok default d_ff = 4 * d_model = 64")
    print("test_default_inner_dim PASSED")


def test_parameter_count() raises:
    """parameters() returns 4 tensors (two Linear layers)."""
    print("Running test_parameter_count...")
    var ffn = FeedForward[DType.float32](8, 16)
    var params = ffn.parameters()
    if len(params) != 4:
        raise Error("FFN should expose 4 parameter tensors")
    print("  ok 4 parameter tensors")
    print("test_parameter_count PASSED")


def test_reject_bad_d_model() raises:
    """Non-positive d_model must raise."""
    print("Running test_reject_bad_d_model...")
    try:
        var _ = FeedForward[DType.float32](0)
        raise Error("Should have rejected d_model = 0")
    except e:
        print("  ok rejected d_model = 0")
    print("test_reject_bad_d_model PASSED")


def test_parity_with_pytorch() raises:
    """Forward must match a PyTorch Linear->GELU(exact)->Linear reference.

    Fixed ramp weights/input and the reference output are transcribed from
    parity_refs/feedforward_parity_reference.py (torch, float64, d_model=4,
    d_ff=8, batch=2, exact GELU). Odyssey Linear uses W of shape (in, out); the
    reference sets torch's (out, in) weight to the transpose so the two match.
    """
    print("Running test_parity_with_pytorch...")

    # Build the reference output list first, before any tensor stores.
    var ref = List[Float64]()
    ref.append(-0.045003)
    ref.append(-0.014435)
    ref.append(0.016133)
    ref.append(0.046701)
    ref.append(-0.038288)
    ref.append(-0.007493)
    ref.append(0.023302)
    ref.append(0.054096)

    var ffn = FeedForward[DType.float64](4, 8)
    for i in range(32):
        ffn.fc1.weight.store[DType.float64](i, Float64(i) * 0.01 - 0.15)
    for i in range(8):
        ffn.fc1.bias.store[DType.float64](i, Float64(i) * 0.02 - 0.08)
    for i in range(32):
        ffn.fc2.weight.store[DType.float64](i, Float64(i) * 0.005 - 0.08)
    for i in range(4):
        ffn.fc2.bias.store[DType.float64](i, Float64(i) * 0.03 - 0.05)

    var x = zeros([2, 4], DType.float64)
    for i in range(8):
        x.store[DType.float64](i, Float64(i) * 0.1 - 0.3)

    var y = ffn.forward(x)
    for i in range(8):
        var d = y.load[DType.float64](i) - ref[i]
        if d < 0:
            d = -d
        if d > 1e-5:
            raise Error("FFN parity mismatch at index " + String(i))
    print("  ok matches PyTorch Linear->GELU->Linear to 1e-5")
    print("test_parity_with_pytorch PASSED")


def test_relu_variant_runs() raises:
    """use_gelu=False (ReLU activation) runs and preserves shape."""
    print("Running test_relu_variant_runs...")
    var ffn = FeedForward[DType.float32](8, 16, use_gelu=False)
    var x = zeros([2, 8], DType.float32)
    var y = ffn.forward(x)
    if y.shape()[0] != 2 or y.shape()[1] != 8:
        raise Error("ReLU-variant FFN must preserve shape")
    print("  ok ReLU variant runs")
    print("test_relu_variant_runs PASSED")


def main() raises:
    """Run all FeedForward tests."""
    print("=" * 60)
    print("FeedForward (Transformer FFN) Test Suite")
    print("=" * 60)
    test_shape_preserved()
    test_default_inner_dim()
    test_parameter_count()
    test_reject_bad_d_model()
    test_parity_with_pytorch()
    test_relu_variant_runs()
    print("=" * 60)
    print("All FeedForward tests PASSED")
    print("=" * 60)
