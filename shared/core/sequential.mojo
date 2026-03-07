"""Sequential module for composing neural network layers.

This module provides parametric Sequential containers that chain the forward
passes of multiple Module-conforming structs. The output of layer i becomes
the input of layer i+1.

Design Note — The Mojo Trait-Object Constraint:
    Mojo v0.26.1 does not support dynamic dispatch via List[Module] because
    trait objects are not heap-dispatchable without unsafe pointer indirection.
    The parametric approach (Sequential2[T0: Module, T1: Module]) is the
    Mojo-idiomatic solution: composition is resolved at compile time, avoids
    ImplicitlyCopyable requirements, and aligns with the project's functional
    and type-safe design philosophy.

    If Mojo later adds variadic generic parameters (*Ts: Module), these two
    structs can be unified into a single Sequential[*Ts: Module].

Usage:
    ```mojo
    from shared.core.sequential import Sequential2, Sequential3
    from shared.core.layers import Linear, ReLULayer

    var model = Sequential2[Linear, ReLULayer](
        Linear(10, 5),
        ReLULayer(),
    )
    var input = zeros([4, 10], DType.float32)
    var output = model.forward(input)  # Shape: [4, 5], after ReLU
    ```

See Also:
    - shared.core.module: Module trait definition
    - shared.core.layers: Available layer implementations
"""

from shared.core.extensor import ExTensor
from shared.core.module import Module


struct Sequential2[T0: Module, T1: Module](Movable):
    """Two-layer sequential module container.

    Chains two Module-conforming layers: output of layer 0 becomes input
    to layer 1. Parameters from both layers are collected together, and
    train/eval mode is propagated to both layers.

    Parameters:
        T0: Type of the first layer, must implement Module.
        T1: Type of the second layer, must implement Module.

    Attributes:
        layer0: First layer in the sequence.
        layer1: Second layer in the sequence.

    Example:
        ```mojo
        var seq = Sequential2[Linear, ReLULayer](Linear(10, 5), ReLULayer())
        var out = seq.forward(input)
        ```
    """

    var layer0: T0
    var layer1: T1

    fn __init__(out self, owned layer0: T0, owned layer1: T1):
        """Initialize with two layers.

        Args:
            layer0: First layer (takes the original input).
            layer1: Second layer (takes output of layer0).
        """
        self.layer0 = layer0^
        self.layer1 = layer1^

    fn __moveinit__(out self, owned other: Self):
        """Move constructor.

        Args:
            other: Source Sequential2 to move from.
        """
        self.layer0 = other.layer0^
        self.layer1 = other.layer1^

    fn forward(mut self, input: ExTensor) raises -> ExTensor:
        """Compute chained forward pass through both layers.

        Args:
            input: Input tensor for layer0.

        Returns:
            Output tensor from layer1.

        Raises:
            Error: If any layer's forward pass fails.
        """
        var out0 = self.layer0.forward(input)
        return self.layer1.forward(out0)

    fn parameters(self) raises -> List[ExTensor]:
        """Collect trainable parameters from both layers.

        Returns:
            List of ExTensor containing parameters from layer0 then layer1.

        Raises:
            Error: If parameter collection fails.
        """
        var params: List[ExTensor] = []
        var p0 = self.layer0.parameters()
        var p1 = self.layer1.parameters()
        for i in range(len(p0)):
            params.append(p0[i])
        for i in range(len(p1)):
            params.append(p1[i])
        return params^

    fn train(mut self):
        """Switch both layers to training mode."""
        self.layer0.train()
        self.layer1.train()

    fn eval(mut self):
        """Switch both layers to evaluation mode."""
        self.layer0.eval()
        self.layer1.eval()


struct Sequential3[T0: Module, T1: Module, T2: Module](Movable):
    """Three-layer sequential module container.

    Chains three Module-conforming layers in order: 0 -> 1 -> 2.
    Parameters from all three layers are collected together, and
    train/eval mode is propagated to all three layers.

    Parameters:
        T0: Type of the first layer, must implement Module.
        T1: Type of the second layer, must implement Module.
        T2: Type of the third layer, must implement Module.

    Attributes:
        layer0: First layer in the sequence.
        layer1: Second layer in the sequence.
        layer2: Third layer in the sequence.

    Example:
        ```mojo
        var seq = Sequential3[Linear, ReLULayer, Linear](
            Linear(10, 5), ReLULayer(), Linear(5, 2)
        )
        var out = seq.forward(input)
        ```
    """

    var layer0: T0
    var layer1: T1
    var layer2: T2

    fn __init__(out self, owned layer0: T0, owned layer1: T1, owned layer2: T2):
        """Initialize with three layers.

        Args:
            layer0: First layer (takes the original input).
            layer1: Second layer (takes output of layer0).
            layer2: Third layer (takes output of layer1).
        """
        self.layer0 = layer0^
        self.layer1 = layer1^
        self.layer2 = layer2^

    fn __moveinit__(out self, owned other: Self):
        """Move constructor.

        Args:
            other: Source Sequential3 to move from.
        """
        self.layer0 = other.layer0^
        self.layer1 = other.layer1^
        self.layer2 = other.layer2^

    fn forward(mut self, input: ExTensor) raises -> ExTensor:
        """Compute chained forward pass through all three layers.

        Args:
            input: Input tensor for layer0.

        Returns:
            Output tensor from layer2.

        Raises:
            Error: If any layer's forward pass fails.
        """
        var out0 = self.layer0.forward(input)
        var out1 = self.layer1.forward(out0)
        return self.layer2.forward(out1)

    fn parameters(self) raises -> List[ExTensor]:
        """Collect trainable parameters from all three layers.

        Returns:
            List of ExTensor containing parameters from layer0, layer1, layer2.

        Raises:
            Error: If parameter collection fails.
        """
        var params: List[ExTensor] = []
        var p0 = self.layer0.parameters()
        var p1 = self.layer1.parameters()
        var p2 = self.layer2.parameters()
        for i in range(len(p0)):
            params.append(p0[i])
        for i in range(len(p1)):
            params.append(p1[i])
        for i in range(len(p2)):
            params.append(p2[i])
        return params^

    fn train(mut self):
        """Switch all three layers to training mode."""
        self.layer0.train()
        self.layer1.train()
        self.layer2.train()

    fn eval(mut self):
        """Switch all three layers to evaluation mode."""
        self.layer0.eval()
        self.layer1.eval()
        self.layer2.eval()
