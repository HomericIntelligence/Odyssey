"""Backward-pass correctness tests for the GoogLeNet CIFAR-10 example.

Two independent layers of evidence for the hand-written 222-parameter backward:

1. Direct unit tests of `concatenate_depthwise_backward` — the highest-risk
   primitive (4-way channel split). These are fully deterministic (sentinel
   tensors, no RNG) and assert the split is a TRUE value-exact inverse of the
   forward concat, not merely shape-correct. A shape-correct-but-value-wrong
   split (e.g. if `split_with_indices` had been used) would fail here even
   though the convergence loop below might still trend downhill.

2. A convergence test: several SGD-momentum steps on a fixed synthetic batch
   spanning all 10 classes, asserting the loss is finite at every step, that
   BOTH the last layer (fc) and the first layer (initial conv) weights change
   — so gradients demonstrably flow end-to-end — and that the final loss drops
   below 0.95 * the first (the same threshold the ResNet-18 sibling test uses)
   — not just a bare `last < first` that a rounding jitter could satisfy.
"""

from std.math import isnan, isinf
from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros
from odyssey.data import one_hot_encode
from model import (
    GoogLeNet,
    concatenate_depthwise,
    concatenate_depthwise_backward,
)
from train import compute_gradients, initialize_velocities


def assert_true(cond: Bool, msg: String) raises:
    if not cond:
        raise Error("assertion failed: " + msg)


def _snapshot(t: AnyTensor) raises -> Float32:
    # Safe scalar read via the sanctioned load API.
    return Float32(t.load[DType.float32](0))


def _make_sentinel(
    batch: Int, c: Int, hw: Int, base: Float32
) raises -> AnyTensor:
    """Build a (batch, c, 1, hw)-shaped tensor with a UNIQUE value per element.

    value(b, ch, i) = base + b*1000 + ch*10 + i, so any misrouting of a
    channel block (wrong offset, wrong batch stride) produces a detectable
    mismatch rather than an accidental collision.
    """
    var t = zeros([batch, c, 1, hw], DType.float32)
    var d = t._data.bitcast[Float32]()
    for b in range(batch):
        for ch in range(c):
            for i in range(hw):
                var idx = ((b * c + ch) * hw) + i
                d[idx] = (
                    base + Float32(b) * 1000.0 + Float32(ch) * 10.0 + Float32(i)
                )
    return t^


def test_concat_backward_split_exact() raises:
    """Backward returns each input's channel block VERBATIM (value-exact split).

    Build 4 sentinel tensors, concat them, then split the concatenated tensor
    back via the backward. Because concat copies gradients through unchanged
    (identity on values), each returned g_k must equal the ORIGINAL t_k
    element-for-element.
    """
    var batch = 2
    var hw = 3
    var c1 = 1
    var c2 = 2
    var c3 = 1
    var c4 = 2
    var t1 = _make_sentinel(batch, c1, hw, base=1.0)
    var t2 = _make_sentinel(batch, c2, hw, base=2.0)
    var t3 = _make_sentinel(batch, c3, hw, base=3.0)
    var t4 = _make_sentinel(batch, c4, hw, base=4.0)

    var cat = concatenate_depthwise(t1, t2, t3, t4)
    assert_true(
        cat.shape()[1] == c1 + c2 + c3 + c4, "concat channel count wrong"
    )

    var grads = concatenate_depthwise_backward(cat, c1, c2, c3, c4)
    var g1 = grads[0]
    var g2 = grads[1]
    var g3 = grads[2]
    var g4 = grads[3]

    _assert_tensor_equal(g1, t1, "g1 != t1")
    _assert_tensor_equal(g2, t2, "g2 != t2")
    _assert_tensor_equal(g3, t3, "g3 != t3")
    _assert_tensor_equal(g4, t4, "g4 != t4")


def _assert_tensor_equal(a: AnyTensor, b: AnyTensor, msg: String) raises:
    assert_true(a._numel == b._numel, msg + " (numel mismatch)")
    var ad = a._data.bitcast[Float32]()
    var bd = b._data.bitcast[Float32]()
    for i in range(a._numel):
        if ad[i] != bd[i]:
            raise Error(
                msg
                + " at index "
                + String(i)
                + ": "
                + String(ad[i])
                + " != "
                + String(bd[i])
            )


def test_concat_backward_roundtrip() raises:
    """Re-concatenating the split pieces reconstructs the concatenated tensor.

    Splitting a concatenated tensor and re-concatenating the pieces must be an
    exact round trip — a stronger check that the split preserves the per-batch
    channel-block layout across all 4 branches.
    """
    var batch = 2
    var hw = 4
    var c1 = 2
    var c2 = 1
    var c3 = 2
    var c4 = 1
    var t1 = _make_sentinel(batch, c1, hw, base=10.0)
    var t2 = _make_sentinel(batch, c2, hw, base=20.0)
    var t3 = _make_sentinel(batch, c3, hw, base=30.0)
    var t4 = _make_sentinel(batch, c4, hw, base=40.0)

    var cat = concatenate_depthwise(t1, t2, t3, t4)
    var parts = concatenate_depthwise_backward(cat, c1, c2, c3, c4)
    var recat = concatenate_depthwise(parts[0], parts[1], parts[2], parts[3])
    _assert_tensor_equal(recat, cat, "round-trip concat(split(cat)) != cat")


def test_concat_backward_rejects_non_float32() raises:
    """Wrong dtype fails loudly (float32-only bitcast contract)."""
    var bad = zeros([1, 4, 1, 1], DType.float64)
    var raised = False
    try:
        _ = concatenate_depthwise_backward(bad, 1, 1, 1, 1)
    except:
        raised = True
    assert_true(raised, "expected non-float32 grad_output to raise")


def test_backward_converges() raises:
    """Reduce loss by >5% with finite losses and both end-layers updating.

    Runs several SGD-momentum steps: asserts finite loss each step, that the
    final loss is < 0.95 * the first, and that BOTH fc (last) and initial-conv
    (first) weights changed — proving gradients reach both ends of the net.
    """
    var num_classes = 10
    var batch = 10  # one sample per class, interleaved
    var model = GoogLeNet(num_classes=num_classes)
    var velocities = initialize_velocities(model)

    # Synthetic images (batch, 3, 32, 32): class-correlated signal so the task
    # is learnable but not trivial. Deterministic by construction.
    var images = zeros([batch, 3, 32, 32], DType.float32)
    var img_d = images._data.bitcast[Float32]()
    for s in range(batch):
        var cls = s  # sample s has class s -> every batch sees all classes
        for i in range(3 * 32 * 32):
            img_d[s * (3 * 32 * 32) + i] = (
                Float32(cls) * 0.05 + Float32(i % 5) * 0.01
            )

    var labels_raw = zeros([batch], DType.uint8)
    var lbl_d = labels_raw._data.bitcast[UInt8]()
    for s in range(batch):
        lbl_d[s] = UInt8(s)
    var labels = one_hot_encode(labels_raw, num_classes)

    var lr = Float32(0.01)
    var momentum = Float32(0.9)

    # Snapshot one weight from the LAST layer (fc) and one from the FIRST
    # (initial conv) so the check proves gradients flow end-to-end. A backward
    # bug that zeroes early-stage gradients while still updating fc would pass
    # an fc-only check but fail the initial-conv one.
    var fc_before = _snapshot(model.fc_weights)
    var conv_before = _snapshot(model.initial_conv_weights)

    var first_loss = Float32(0.0)
    var last_loss = Float32(0.0)
    var steps = 15
    for step in range(steps):
        var loss = compute_gradients(
            model, images, labels, lr, momentum, velocities
        )
        assert_true(not isnan(loss), "loss is NaN at step " + String(step + 1))
        assert_true(not isinf(loss), "loss is Inf at step " + String(step + 1))
        if step == 0:
            first_loss = loss
        last_loss = loss
        print(
            "  Step "
            + String(step + 1)
            + "/"
            + String(steps)
            + ", Loss: "
            + String(loss)
        )

    assert_true(
        _snapshot(model.fc_weights) != fc_before,
        "fc_weights did not change — SGD update not applied to last layer",
    )
    assert_true(
        _snapshot(model.initial_conv_weights) != conv_before,
        (
            "initial_conv_weights did not change — gradient did not reach the"
            " first layer"
        ),
    )

    print("first=" + String(first_loss) + " last=" + String(last_loss))
    # Hard floor first (clear signal if loss went UP), then the 5% threshold
    # the ResNet-18 sibling test also enforces.
    assert_true(
        last_loss < first_loss,
        "Loss did NOT decrease: first="
        + String(first_loss)
        + " last="
        + String(last_loss),
    )
    assert_true(
        last_loss < first_loss * Float32(0.95),
        "Loss decrease < 5%: first="
        + String(first_loss)
        + " last="
        + String(last_loss),
    )
    print("GOOGLENET_BWD_CONVERGES: PASS")


def main() raises:
    print("test_concat_backward_split_exact...", end="")
    test_concat_backward_split_exact()
    print(" PASS")
    print("test_concat_backward_roundtrip...", end="")
    test_concat_backward_roundtrip()
    print(" PASS")
    print("test_concat_backward_rejects_non_float32...", end="")
    test_concat_backward_rejects_non_float32()
    print(" PASS")
    print("test_backward_converges...")
    test_backward_converges()
    print("test_backward_converges PASS")
    print("ALL GOOGLENET BACKWARD TESTS PASSED")
