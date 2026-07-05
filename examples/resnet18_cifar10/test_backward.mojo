"""Backward-pass integration tests for ResNet-18 CIFAR-10 example (#5515)."""

from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.tensor.tensor_creation import zeros, randn
from std.math import isnan, isinf
from model import (
    ResNet18,
    ResNet18Velocities,
    initialize_velocities,
)
from train import train_step, train_epoch
from projectodyssey.data.formats.idx_loader import one_hot_encode


def assert_true(cond: Bool, msg: String) raises:
    if not cond:
        raise Error("assertion failed: " + msg)


def _make_non_uniform_batch() raises -> Tuple[AnyTensor, AnyTensor]:
    var img_shape = [4, 3, 32, 32]
    var images = randn(img_shape, DType.float32, seed=42)
    # One-hot encoded labels: cross_entropy requires targets to have the
    # same shape as logits (batch=4, num_classes=10) — see loss.mojo:309.
    var lbl_shape = [4, 10]
    var labels = zeros(lbl_shape, DType.float32)
    labels.set(0 * 10 + 3, Float32(1.0))  # class 3 for sample 0
    labels.set(1 * 10 + 7, Float32(1.0))  # class 7 for sample 1
    labels.set(2 * 10 + 1, Float32(1.0))  # class 1 for sample 2
    labels.set(3 * 10 + 9, Float32(1.0))  # class 9 for sample 3
    return (images^, labels^)


def _snapshot(t: AnyTensor) raises -> Float32:
    # Safe read via load[dtype] — sanctioned API (any_tensor.mojo:1479)
    return Float32(t.load[DType.float32](0))


def test_train_step_runs_without_error() raises:
    var model = ResNet18(num_classes=10)
    var vel = initialize_velocities(model)
    var pair = _make_non_uniform_batch()
    var loss = train_step(model, pair[0], pair[1], vel, lr=0.01, momentum=0.9)
    assert_true(not isnan(loss), "loss is NaN")
    assert_true(not isinf(loss), "loss is Inf")


def test_every_stage_receives_nonzero_gradient() raises:
    var model = ResNet18(num_classes=10)
    var vel = initialize_velocities(model)

    # One representative param per stage/block group (identity vs projection):
    var b_init = _snapshot(model.conv1_kernel)  # initial conv
    var b_s1i = _snapshot(model.s1b1_conv1_kernel)  # stage 1 identity
    var b_s2p = _snapshot(model.s2b1_proj_kernel)  # stage 2 projection
    var b_s2i = _snapshot(model.s2b2_conv1_kernel)  # stage 2 identity
    var b_s3p = _snapshot(model.s3b1_proj_kernel)  # stage 3 projection
    var b_s3i = _snapshot(model.s3b2_conv1_kernel)  # stage 3 identity
    var b_s4p = _snapshot(model.s4b1_proj_kernel)  # stage 4 projection
    var b_s4i = _snapshot(model.s4b2_conv1_kernel)  # stage 4 identity
    var b_fc = _snapshot(model.fc_weights)  # FC

    var pair = _make_non_uniform_batch()
    _ = train_step(model, pair[0], pair[1], vel, lr=0.01, momentum=0.9)

    assert_true(
        _snapshot(model.conv1_kernel) != b_init, "initial conv unchanged"
    )
    assert_true(
        _snapshot(model.s1b1_conv1_kernel) != b_s1i, "stage1 identity unchanged"
    )
    assert_true(
        _snapshot(model.s2b1_proj_kernel) != b_s2p,
        "stage2 projection unchanged",
    )
    assert_true(
        _snapshot(model.s2b2_conv1_kernel) != b_s2i, "stage2 identity unchanged"
    )
    assert_true(
        _snapshot(model.s3b1_proj_kernel) != b_s3p,
        "stage3 projection unchanged",
    )
    assert_true(
        _snapshot(model.s3b2_conv1_kernel) != b_s3i, "stage3 identity unchanged"
    )
    assert_true(
        _snapshot(model.s4b1_proj_kernel) != b_s4p,
        "stage4 projection unchanged",
    )
    assert_true(
        _snapshot(model.s4b2_conv1_kernel) != b_s4i, "stage4 identity unchanged"
    )
    assert_true(_snapshot(model.fc_weights) != b_fc, "fc unchanged")


def _build_separable_batch(
    samples_per_class: Int, num_classes: Int, seed: Int
) raises -> Tuple[AnyTensor, AnyTensor]:
    """Build (images, one_hot_labels) with strong per-class channel bias.

    Bias = 2.0 * c/num_classes - 1.0 ∈ [-1.0, +0.8], added to every pixel.
    Dominates N(0,1) noise → linearly separable at conv1+BN.
    """
    var total = num_classes * samples_per_class
    var img_shape = List[Int]()
    img_shape.append(total)
    img_shape.append(3)
    img_shape.append(32)
    img_shape.append(32)
    var images = randn(img_shape, DType.float32, seed=seed)
    var pixels_per_sample = 3 * 32 * 32

    var lbl_int_shape = List[Int]()
    lbl_int_shape.append(total)
    var labels_int = zeros(lbl_int_shape, DType.uint8)

    for c in range(num_classes):
        var bias = Float32(2.0) * Float32(c) / Float32(num_classes) - Float32(
            1.0
        )
        for s in range(samples_per_class):
            var idx = c * samples_per_class + s
            labels_int.set(idx, UInt8(c))
            var base = idx * pixels_per_sample
            for k in range(pixels_per_sample):
                var cur = images.load[DType.float32](base + k)
                images.store[DType.float32](base + k, cur + bias)

    var labels_one_hot = one_hot_encode(labels_int, num_classes=num_classes)
    return (images, labels_one_hot)


def test_loss_decreases_over_steps() raises:
    """One epoch on separable synthetic data reduces loss by >=5%.

    Data: N=100 (10 classes × 10 samples), per-class channel bias ± ~1.0 on
    top of N(0,1) noise → linearly separable at conv1+BN. 10 batches × 10.
    """
    var num_classes = 10
    var samples_per_class = 10
    var batch_size = 10
    var pair = _build_separable_batch(samples_per_class, num_classes, seed=42)
    var images = pair[0]
    var labels = pair[1]

    var model = ResNet18(num_classes=num_classes)
    var velocities = initialize_velocities(model)

    var loss_history: List[Float32] = []
    _ = train_epoch(
        model,
        images,
        labels,
        batch_size,
        Float32(0.01),
        Float32(0.9),
        velocities,
        loss_history,
        1,
        1,
    )

    assert_true(
        len(loss_history) >= 2, "Need >=2 batches to test loss decrease"
    )
    var first = loss_history[0]
    var last = loss_history[len(loss_history) - 1]

    # Hard floor first — clearer failure signal if training goes UP
    assert_true(
        last < first,
        "Loss did NOT decrease at all: first="
        + String(first)
        + " last="
        + String(last),
    )
    # Issue-required threshold: loss[final] < loss[0] * 0.95
    assert_true(
        last < first * Float32(0.95),
        "Loss decrease < 5%: first=" + String(first) + " last=" + String(last),
    )


def main() raises:
    print("test_train_step_runs_without_error...", end="")
    test_train_step_runs_without_error()
    print(" PASS")
    print("test_every_stage_receives_nonzero_gradient...", end="")
    test_every_stage_receives_nonzero_gradient()
    print(" PASS")
    print("test_loss_decreases_over_steps...", end="")
    test_loss_decreases_over_steps()
    print(" PASS")
