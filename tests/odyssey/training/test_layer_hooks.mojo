"""Tests for the per-layer LayerHook mechanism (issue #5418).

Covers:
- HookRegistry registration (by layer name and all-layers wildcard)
- run_pre_forward / run_post_forward dispatch and name matching
- Multi-hook composition in registration order
- STOP short-circuit semantics
- ActivationStatsHook output statistics
"""

from std.testing import assert_true, assert_equal, assert_false
from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros, ones
from odyssey.training.base import CallbackSignal, CONTINUE, STOP
from odyssey.training.layer_hooks import (
    LayerHook,
    HookRegistry,
    ActivationStatsHook,
)


struct CountingHook(Copyable, LayerHook, Movable):
    """Test hook: counts pre/post invocations and can be set to STOP."""

    var pre_count: Int
    var post_count: Int
    var stop_on_pre: Bool

    def __init__(out self, stop_on_pre: Bool = False):
        self.pre_count = 0
        self.post_count = 0
        self.stop_on_pre = stop_on_pre

    def on_pre_forward(
        mut self, layer_name: String, mut input: AnyTensor
    ) raises -> CallbackSignal:
        self.pre_count += 1
        if self.stop_on_pre:
            return STOP
        return CONTINUE

    def on_post_forward(
        mut self, layer_name: String, input: AnyTensor, mut output: AnyTensor
    ) raises -> CallbackSignal:
        self.post_count += 1
        return CONTINUE


def test_register_and_run_pre_forward() raises:
    """A hook registered for a layer fires on that layer's pre-forward."""
    var reg = HookRegistry[CountingHook]()
    reg.register("conv1", CountingHook())

    var x = zeros([2, 3], DType.float32)
    var signal = reg.run_pre_forward("conv1", x)

    assert_equal(signal.value, CONTINUE.value, "should return CONTINUE")
    assert_equal(reg._hooks[0].pre_count, 1, "pre_count incremented")


def test_name_mismatch_does_not_fire() raises:
    """A hook registered for one layer does not fire for a different layer."""
    var reg = HookRegistry[CountingHook]()
    reg.register("conv1", CountingHook())

    var x = zeros([2, 3], DType.float32)
    _ = reg.run_pre_forward("conv2", x)

    assert_equal(reg._hooks[0].pre_count, 0, "must not fire on conv2")


def test_all_layers_wildcard_fires_everywhere() raises:
    """Wildcard registration makes a hook fire on every layer name."""
    var reg = HookRegistry[CountingHook]()
    reg.register_all_layers(CountingHook())

    var x = zeros([2, 3], DType.float32)
    _ = reg.run_pre_forward("conv1", x)
    _ = reg.run_pre_forward("fc1", x)

    assert_equal(reg._hooks[0].pre_count, 2, "wildcard fires on both layers")


def test_post_forward_runs() raises:
    """Post-forward dispatch invokes the matching hook's on_post_forward."""
    var reg = HookRegistry[CountingHook]()
    reg.register("conv1", CountingHook())

    var x = zeros([2, 3], DType.float32)
    var y = zeros([2, 3], DType.float32)
    var signal = reg.run_post_forward("conv1", x, y)

    assert_equal(signal.value, CONTINUE.value, "post returns CONTINUE")
    assert_equal(reg._hooks[0].post_count, 1, "post_count incremented")


def test_multiple_hooks_compose_in_order() raises:
    """Multiple hooks on the same layer all run in registration order."""
    var reg = HookRegistry[CountingHook]()
    reg.register("conv1", CountingHook())
    reg.register("conv1", CountingHook())

    var x = zeros([2, 3], DType.float32)
    _ = reg.run_pre_forward("conv1", x)

    assert_equal(reg._hooks[0].pre_count, 1, "first hook ran")
    assert_equal(reg._hooks[1].pre_count, 1, "second hook ran")


def test_stop_short_circuits_remaining_hooks() raises:
    """A hook returning STOP halts the remaining hooks for that layer."""
    var reg = HookRegistry[CountingHook]()
    reg.register("conv1", CountingHook(stop_on_pre=True))
    reg.register("conv1", CountingHook())

    var x = zeros([2, 3], DType.float32)
    var signal = reg.run_pre_forward("conv1", x)

    assert_equal(signal.value, STOP.value, "registry propagates STOP")
    assert_equal(reg._hooks[0].pre_count, 1, "first (STOP) hook ran")
    assert_equal(reg._hooks[1].pre_count, 0, "second hook short-circuited")


def test_empty_registry_returns_continue() raises:
    """A registry with no hooks returns CONTINUE."""
    var reg = HookRegistry[CountingHook]()
    var x = zeros([2, 3], DType.float32)
    assert_equal(
        reg.run_pre_forward("conv1", x).value,
        CONTINUE.value,
        "empty registry continues",
    )


def test_activation_stats_hook_records_mean() raises:
    """ActivationStatsHook records the mean of a layer's output."""
    var reg = HookRegistry[ActivationStatsHook]()
    reg.register("conv1", ActivationStatsHook())

    var x = zeros([1, 4], DType.float32)
    var out = ones([1, 4], DType.float32)
    _ = reg.run_post_forward("conv1", x, out)

    assert_equal(reg._hooks[0].call_count, 1, "stats hook ran once")
    assert_true(
        reg._hooks[0].last_mean > 0.99 and reg._hooks[0].last_mean < 1.01,
        "mean of all-ones output is 1.0",
    )
    assert_true(
        reg._hooks[0].last_max_abs > 0.99 and reg._hooks[0].last_max_abs < 1.01,
        "max-abs of all-ones output is 1.0",
    )


def test_activation_stats_hook_zero_output() raises:
    """ActivationStatsHook reports zero stats for an all-zero output."""
    var reg = HookRegistry[ActivationStatsHook]()
    reg.register_all_layers(ActivationStatsHook())

    var x = zeros([1, 4], DType.float32)
    var out = zeros([2, 8], DType.float32)
    _ = reg.run_post_forward("any_layer", x, out)

    assert_equal(reg._hooks[0].last_mean, 0.0, "zero output -> zero mean")
    assert_equal(reg._hooks[0].last_max_abs, 0.0, "zero output -> zero max-abs")


def main() raises:
    """Run all test_layer_hooks tests."""
    print("Running test_layer_hooks tests...")

    test_register_and_run_pre_forward()
    print("✓ test_register_and_run_pre_forward")

    test_name_mismatch_does_not_fire()
    print("✓ test_name_mismatch_does_not_fire")

    test_all_layers_wildcard_fires_everywhere()
    print("✓ test_all_layers_wildcard_fires_everywhere")

    test_post_forward_runs()
    print("✓ test_post_forward_runs")

    test_multiple_hooks_compose_in_order()
    print("✓ test_multiple_hooks_compose_in_order")

    test_stop_short_circuits_remaining_hooks()
    print("✓ test_stop_short_circuits_remaining_hooks")

    test_empty_registry_returns_continue()
    print("✓ test_empty_registry_returns_continue")

    test_activation_stats_hook_records_mean()
    print("✓ test_activation_stats_hook_records_mean")

    test_activation_stats_hook_zero_output()
    print("✓ test_activation_stats_hook_zero_output")

    print("\nAll test_layer_hooks tests passed!")
