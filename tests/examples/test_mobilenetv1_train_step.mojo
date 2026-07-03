"""Functional smoke test for MobileNetV1 train_step (issue #5525).

Exercises the real training machinery from the example package and verifies
the four acceptance criteria from #5525:

  1. ``initialize_velocities(model)`` returns exactly 110 velocity buffers
     (initial 4 + 13 blocks x 8 + fc 2).
  2. Two successive ``compute_gradients`` calls on a fixed batch produce
     finite, strictly-positive losses that strictly decrease.
  3. At least one model parameter changes after a training step.
  4. A post-training forward pass returns logits of shape (2, 10).

Runs as: pixi run mojo -I src -I . ./tests/examples/test_mobilenetv1_train_step.mojo

The example modules are imported as package submodules (examples has an
__init__.mojo, as does examples/mobilenetv1_cifar10), so the compiler resolves
them via the repository-root include path used by the test group runner.
"""

from math import isnan, isinf

from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.tensor.tensor_creation import ones, zeros

from examples.mobilenetv1_cifar10.model import MobileNetV1
from examples.mobilenetv1_cifar10.train import (
    compute_gradients,
    initialize_velocities,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_batch() raises -> Tuple[AnyTensor, AnyTensor]:
    """Build a small fixed batch: ones input + deterministic one-hot labels.

    Returns:
        (input, labels_onehot) where
            input:          (2, 3, 32, 32) filled with 1.0
            labels_onehot:  (2, 10), sample 0 -> class 0, sample 1 -> class 1.
    """
    var batch_size = 2
    var num_classes = 10

    var input_shape: List[Int] = [batch_size, 3, 32, 32]
    var input = ones(input_shape, DType.float32)

    var labels_shape: List[Int] = [batch_size, num_classes]
    var labels_onehot = zeros(labels_shape, DType.float32)
    # Sample 0 -> class 0, sample 1 -> class 1 (rows sum to 1 for cross-entropy).
    # AnyTensor.store[dtype](index, value) writes to
    # self._data.bitcast[Scalar[dtype]]()[index] (see any_tensor.mojo), the
    # same flat-index bitcast pattern production code uses for element access
    # (e.g. loss_t._data.bitcast[Float32]()[0] in train.mojo). Tensors are
    # stored contiguously in row-major order, so `row * num_classes + col`
    # is the correct flat index for element (row, col) of a (rows, cols)
    # tensor.
    labels_onehot.store[DType.float32](0 * num_classes + 0, Float32(1.0))
    labels_onehot.store[DType.float32](1 * num_classes + 1, Float32(1.0))

    return (input^, labels_onehot^)


def _assert_true(cond: Bool, message: String) raises:
    if not cond:
        raise Error("Assertion failed: " + message)


# ---------------------------------------------------------------------------
# Acceptance-criteria tests
# ---------------------------------------------------------------------------


def test_velocity_count_is_110() raises:
    """Criterion 1: velocity buffer count == 110."""
    var model = MobileNetV1(num_classes=10)
    var velocities = initialize_velocities(model)
    _assert_true(
        len(velocities) == 110,
        "expected 110 velocity buffers, got " + String(len(velocities)),
    )


def test_losses_finite_positive_and_decreasing() raises:
    """Criterion 2 + 3: two losses finite/positive & strictly decreasing,
    and at least one parameter changes after a step."""
    var model = MobileNetV1(num_classes=10)
    var velocities = initialize_velocities(model)

    var batch = _make_batch()
    var input = batch[0]
    var labels_onehot = batch[1]

    var learning_rate = Float32(0.05)
    var momentum = Float32(0.0)

    # Snapshot a parameter element before any update (fc_bias starts at 0).
    var fc_bias_before = model.fc_bias.load[DType.float32](0)

    # First training step.
    var loss1 = compute_gradients(
        model, input, labels_onehot, learning_rate, momentum, velocities
    )

    # Parameter must have changed (criterion 3).
    var fc_bias_after = model.fc_bias.load[DType.float32](0)
    _assert_true(
        fc_bias_after != fc_bias_before,
        "expected fc_bias[0] to change after a training step (before="
        + String(fc_bias_before)
        + ", after="
        + String(fc_bias_after)
        + ")",
    )

    # Second training step on a fresh batch to avoid use-after-move
    var batch2 = _make_batch()
    var input2 = batch2[0]
    var labels_onehot2 = batch2[1]
    var loss2 = compute_gradients(
        model, input2, labels_onehot2, learning_rate, momentum, velocities
    )

    # Both losses finite and strictly positive (criterion 2).
    _assert_true(
        not isnan(loss1) and not isinf(loss1),
        "loss1 must be finite, got " + String(loss1),
    )
    _assert_true(
        not isnan(loss2) and not isinf(loss2),
        "loss2 must be finite, got " + String(loss2),
    )
    _assert_true(
        loss1 > Float32(0.0), "loss1 must be positive, got " + String(loss1)
    )
    _assert_true(
        loss2 > Float32(0.0), "loss2 must be positive, got " + String(loss2)
    )

    # Strictly decreasing.
    _assert_true(
        loss2 < loss1,
        "expected loss2 < loss1, got loss1="
        + String(loss1)
        + ", loss2="
        + String(loss2),
    )


def test_post_train_forward_shape() raises:
    """Criterion 4: forward returns (2, 10) logits after training."""
    var model = MobileNetV1(num_classes=10)
    var velocities = initialize_velocities(model)

    # Training step with first batch
    var batch1 = _make_batch()
    var input1 = batch1[0]
    var labels_onehot1 = batch1[1]
    _ = compute_gradients(
        model, input1, labels_onehot1, Float32(0.05), Float32(0.0), velocities
    )

    # Forward pass with separate batch to avoid use-after-move
    var batch2 = _make_batch()
    var input2 = batch2[0]
    var logits = model.forward(input2, training=False)
    var shape = logits.shape()
    _assert_true(
        len(shape) == 2,
        "expected rank-2 logits, got rank " + String(len(shape)),
    )
    _assert_true(
        shape[0] == 2,
        "expected batch dimension 2, got " + String(shape[0]),
    )
    _assert_true(
        shape[1] == 10,
        "expected 10 classes, got " + String(shape[1]),
    )


def main() raises:
    print("Running test_mobilenetv1_train_step tests...")

    test_velocity_count_is_110()
    print("[PASS] test_velocity_count_is_110")

    test_losses_finite_positive_and_decreasing()
    print("[PASS] test_losses_finite_positive_and_decreasing")

    test_post_train_forward_shape()
    print("[PASS] test_post_train_forward_shape")

    print("PASSED")
