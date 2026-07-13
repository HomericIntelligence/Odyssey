"""BatchNorm running-stat persistence regression test for GoogLeNet (#5575).

A training forward (`training=True`) must write the EMA-updated running_mean /
running_var back onto the model, so a subsequent inference forward
(`training=False`) uses the accumulated statistics rather than the init values
(mean=0, var=1). Before the fix, both `GoogLeNet.forward`'s stem BN and every
BN inside `InceptionModule.forward` / `inception_forward_cached` discarded
batch_norm2d's updated stats via `out, _, _ = batch_norm2d(...)`, so inference
silently used stale init values and produced wrong results.

Same class of bug (and fix) as MobileNetV1 #5537.
"""

from model import GoogLeNet
from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros


def assert_true(cond: Bool, msg: String) raises:
    if not cond:
        raise Error("assertion failed: " + msg)


def _running_var_sig(t: AnyTensor) raises -> Float32:
    """Sum of running_var — a scalar signature that is num_channels at init.

    running_var is initialized to all-ones, so its sum equals the channel
    count. A training forward pushes it toward the batch variance, changing the
    sum; this gives a single value to compare before/after.
    """
    var d = t._data.bitcast[Float32]()
    var s = Float32(0.0)
    for i in range(t._numel):
        s += d[i]
    return s


def _running_mean_sig(t: AnyTensor) raises -> Float32:
    """Sum of running_mean — 0 at init (all-zeros), nonzero after training."""
    var d = t._data.bitcast[Float32]()
    var s = Float32(0.0)
    for i in range(t._numel):
        s += d[i]
    return s


def test_bn_running_stats_persist_after_training() raises:
    """A training forward updates the model's persistent BN running stats."""
    var model = GoogLeNet(num_classes=10)

    # Non-trivial, non-uniform input so batch statistics differ from init.
    var x = zeros([2, 3, 32, 32], DType.float32)
    var xd = x._data.bitcast[Float32]()
    for i in range(2 * 3 * 32 * 32):
        xd[i] = Float32(i % 7) * 0.1 - 0.3

    # Snapshot the initial (stem) BN stats (init: mean sum = 0, var sum = C).
    var mean_before = _running_mean_sig(model.initial_bn_running_mean)
    var var_before = _running_var_sig(model.initial_bn_running_var)

    _ = model.forward(x, training=True)

    var mean_after = _running_mean_sig(model.initial_bn_running_mean)
    var var_after = _running_var_sig(model.initial_bn_running_var)

    # Init running_mean is all-zeros; a training forward must move it.
    assert_true(
        mean_before == Float32(0.0),
        "precondition: initial running_mean must start at 0, got "
        + String(mean_before),
    )
    assert_true(
        mean_after != mean_before,
        "initial_bn_running_mean was NOT updated by training forward (still "
        + String(mean_after)
        + ") — stats discarded (#5575)",
    )
    assert_true(
        var_after != var_before,
        "initial_bn_running_var was NOT updated by training forward (still "
        + String(var_after)
        + ") — stats discarded (#5575)",
    )


def test_bn_stats_persist_in_inception_modules() raises:
    """The Inception modules also persist their BN running stats.

    Checks the FIRST (inception_3a) AND LAST (inception_5b) module, so this
    proves the shared InceptionModule.forward path persists for every module,
    not just one.
    """
    var model = GoogLeNet(num_classes=10)
    var x = zeros([2, 3, 32, 32], DType.float32)
    var xd = x._data.bitcast[Float32]()
    for i in range(2 * 3 * 32 * 32):
        xd[i] = Float32(i % 5) * 0.2 - 0.4

    var i3a_before = _running_mean_sig(model.inception_3a.bn1x1_1_running_mean)
    var i5b_before = _running_mean_sig(model.inception_5b.bn1x1_1_running_mean)

    _ = model.forward(x, training=True)

    assert_true(
        _running_mean_sig(model.inception_3a.bn1x1_1_running_mean)
        != i3a_before,
        "inception_3a (first) BN running_mean not persisted (#5575)",
    )
    assert_true(
        _running_mean_sig(model.inception_5b.bn1x1_1_running_mean)
        != i5b_before,
        "inception_5b (last) BN running_mean not persisted (#5575)",
    )


def test_inference_forward_leaves_stats_unchanged() raises:
    """A training=False forward must NOT change running stats (inference-safe).
    """
    var model = GoogLeNet(num_classes=10)
    var x = zeros([2, 3, 32, 32], DType.float32)
    var xd = x._data.bitcast[Float32]()
    for i in range(2 * 3 * 32 * 32):
        xd[i] = Float32(i % 3) * 0.3

    var mean_before = _running_mean_sig(model.initial_bn_running_mean)
    _ = model.forward(x, training=False)
    assert_true(
        _running_mean_sig(model.initial_bn_running_mean) == mean_before,
        "inference forward must not mutate BN running stats",
    )


def main() raises:
    print("test_bn_running_stats_persist_after_training...", end="")
    test_bn_running_stats_persist_after_training()
    print(" PASS")
    print("test_bn_stats_persist_in_inception_modules...", end="")
    test_bn_stats_persist_in_inception_modules()
    print(" PASS")
    print("test_inference_forward_leaves_stats_unchanged...", end="")
    test_inference_forward_leaves_stats_unchanged()
    print(" PASS")
    print("ALL GoogLeNet BN-persistence TESTS PASSED")
