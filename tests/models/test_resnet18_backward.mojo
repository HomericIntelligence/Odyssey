"""Backward-pass integration tests for ResNet-18 CIFAR-10 example (#5515)."""

from tests.projectodyssey.conftest import assert_true
from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.tensor.tensor_creation import zeros, randn
from math import isnan, isinf
from examples.resnet18_cifar10.model import (
    ResNet18,
    ResNet18Velocities,
    initialize_velocities,
)
from examples.resnet18_cifar10.train import train_step


def _make_non_uniform_batch() raises -> Tuple[AnyTensor, AnyTensor]:
    var img_shape = List[Int](4, 3, 32, 32)
    var images = randn(img_shape, DType.float32, seed=42)
    # One-hot encoded labels: cross_entropy requires targets to have the
    # same shape as logits (batch=4, num_classes=10) — see loss.mojo:309.
    var lbl_shape = List[Int](4, 10)
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


def main() raises:
    print("test_train_step_runs_without_error...", end="")
    test_train_step_runs_without_error()
    print(" PASS")
    print("test_every_stage_receives_nonzero_gradient...", end="")
    test_every_stage_receives_nonzero_gradient()
    print(" PASS")
