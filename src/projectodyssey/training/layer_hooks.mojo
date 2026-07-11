"""Per-layer pre/post-forward hooks with read-write tensor access.

`LayerHook` is a per-layer instrumentation point, distinct from the
whole-model `Callback` trait in `base.mojo`. Where `Callback` is read-only
and fired by the trainer around epochs/batches, a `LayerHook` runs
immediately before and after an individual layer's `forward()` and may
mutate the layer's input and output tensors.

The two traits are kept separate so the read-only invariant on `Callback`
is preserved.

Composition: a `HookRegistry[H]` holds hook instances keyed by layer name.
On invocation it runs every hook registered for a layer in registration
order; the first hook returning a non-CONTINUE signal short-circuits the
rest and propagates that signal.

Example:
    ```mojo
    from odyssey.training.layer_hooks import (
        HookRegistry, ActivationStatsHook
    )

    var reg = HookRegistry[ActivationStatsHook]()
    reg.register("conv1", ActivationStatsHook())

    # Inside a model's forward(), around each layer:
    _ = reg.run_pre_forward("conv1", input)
    var out = conv1.forward(input)
    _ = reg.run_post_forward("conv1", input, out)
    ```
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.training.base import CallbackSignal, CONTINUE


trait LayerHook(Copyable, Movable):
    """Per-layer pre/post-forward hook with read-write tensor access.

    Implementations run around an individual layer's `forward()` call.
    Unlike `Callback`, a `LayerHook` receives the layer's input and output
    tensors by mutable reference and may modify them (e.g. forward-time
    output clipping, activation rescaling).

    Both methods return a `CallbackSignal`; returning `STOP` halts the
    remaining hooks for that layer and propagates out of the registry.
    """

    def on_pre_forward(
        mut self, layer_name: String, mut input: AnyTensor
    ) raises -> CallbackSignal:
        """Run before a layer's `forward()`.

        Args:
            layer_name: Name of the layer about to run.
            input: The layer's input tensor (mutable — may be rewritten).

        Returns:
            CONTINUE to proceed, STOP to halt remaining hooks.
        """
        ...

    def on_post_forward(
        mut self, layer_name: String, input: AnyTensor, mut output: AnyTensor
    ) raises -> CallbackSignal:
        """Run after a layer's `forward()`.

        Args:
            layer_name: Name of the layer that just ran.
            input: The layer's input tensor (read-only here).
            output: The layer's output tensor (mutable — may be rewritten).

        Returns:
            CONTINUE to proceed, STOP to halt remaining hooks.
        """
        ...


struct HookRegistry[H: LayerHook](Copyable, Movable):
    """Holds `LayerHook` instances keyed by layer name.

    A model opts into per-layer hooks by owning a `HookRegistry` and calling
    `run_pre_forward` / `run_post_forward` around each layer in its
    `forward()` sequence.

    The registry is parametric on a single hook type `H`. Mojo does not
    support heterogeneous trait-object lists, so all hooks in one registry
    share a type; register multiple instances to compose behavior.

    Parameters:
        H: The concrete `LayerHook` implementation stored by this registry.
    """

    var _hooks: List[Self.H]
    """Registered hook instances, parallel to `_names`."""
    var _names: List[String]
    """Layer-name match for each hook. The empty string matches every layer."""

    def __init__(out self):
        """Create an empty registry."""
        self._hooks = List[Self.H]()
        self._names = List[String]()

    def register(mut self, layer_name: String, var hook: Self.H):
        """Register a hook for a specific layer.

        Args:
            layer_name: The layer name this hook fires on.
            hook: The hook instance (ownership transferred).
        """
        self._names.append(layer_name)
        self._hooks.append(hook^)

    def register_all_layers(mut self, var hook: Self.H):
        """Register a hook that fires on every layer.

        Args:
            hook: The hook instance (ownership transferred).
        """
        self._names.append(String(""))
        self._hooks.append(hook^)

    def _matches(self, registered_name: String, layer_name: String) -> Bool:
        """Whether a registered entry applies to `layer_name`.

        An empty registered name is the all-layers wildcard.
        """
        return registered_name == "" or registered_name == layer_name

    def run_pre_forward(
        mut self, layer_name: String, mut input: AnyTensor
    ) raises -> CallbackSignal:
        """Invoke every matching hook's `on_pre_forward` in registration order.

        Args:
            layer_name: The layer about to run.
            input: The layer's input tensor (mutable).

        Returns:
            The first non-CONTINUE signal, or CONTINUE if all hooks continue.
        """
        for i in range(len(self._hooks)):
            if self._matches(self._names[i], layer_name):
                var signal = self._hooks[i].on_pre_forward(layer_name, input)
                if signal.value != CONTINUE.value:
                    return signal
        return CONTINUE

    def run_post_forward(
        mut self, layer_name: String, input: AnyTensor, mut output: AnyTensor
    ) raises -> CallbackSignal:
        """Invoke every matching hook's `on_post_forward` in registration order.

        Args:
            layer_name: The layer that just ran.
            input: The layer's input tensor (read-only).
            output: The layer's output tensor (mutable).

        Returns:
            The first non-CONTINUE signal, or CONTINUE if all hooks continue.
        """
        for i in range(len(self._hooks)):
            if self._matches(self._names[i], layer_name):
                var signal = self._hooks[i].on_post_forward(
                    layer_name, input, output
                )
                if signal.value != CONTINUE.value:
                    return signal
        return CONTINUE


struct ActivationStatsHook(Copyable, LayerHook, Movable):
    """Example `LayerHook` that records output activation statistics.

    After each layer's `forward()`, records the mean and maximum absolute
    value of the output tensor. Demonstrates the per-layer instrumentation
    use case from issue #5418 — activation statistics that need the layer's
    output tensor, which `LoggingCallback` cannot reach.

    This hook does not mutate tensors; it only observes.
    """

    var last_mean: Float64
    """Mean of the most recent layer output (0.0 before the first call)."""
    var last_max_abs: Float64
    """Maximum absolute value of the most recent layer output."""
    var call_count: Int
    """Number of `on_post_forward` invocations so far."""

    def __init__(out self):
        """Create a hook with zeroed statistics."""
        self.last_mean = 0.0
        self.last_max_abs = 0.0
        self.call_count = 0

    def on_pre_forward(
        mut self, layer_name: String, mut input: AnyTensor
    ) raises -> CallbackSignal:
        """No-op: this hook only observes outputs."""
        return CONTINUE

    def on_post_forward(
        mut self, layer_name: String, input: AnyTensor, mut output: AnyTensor
    ) raises -> CallbackSignal:
        """Record mean and max-abs of the layer output.

        Args:
            layer_name: The layer that just ran.
            input: The layer's input tensor (unused).
            output: The layer's output tensor (observed, not modified).

        Returns:
            CONTINUE — observation never halts the forward pass.
        """
        var n = output.numel()
        if n == 0:
            self.last_mean = 0.0
            self.last_max_abs = 0.0
        else:
            var total = 0.0
            var max_abs = 0.0
            for i in range(n):
                var v = output._get_float64(i)
                total += v
                var abs_v = v if v >= 0.0 else -v
                if abs_v > max_abs:
                    max_abs = abs_v
            self.last_mean = total / Float64(n)
            self.last_max_abs = max_abs
        self.call_count += 1
        return CONTINUE
