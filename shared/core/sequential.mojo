"""Sequential module for composing neural network layers.

This module provides parametric Sequential containers that chain the forward
passes of multiple Module-conforming structs. The output of layer i becomes
the input of layer i+1.

Design Note — The Mojo Trait-Object Constraint:
    Mojo v0.26.1 does not support dynamic dispatch via List[Module] because
    trait objects are not heap-dispatchable without unsafe pointer indirection.
    The parametric approach (Sequential2[T0: Module & Movable, T1: Module & Movable]) is the
    Mojo-idiomatic solution: composition is resolved at compile time, avoids
    ImplicitlyCopyable requirements, and aligns with the project's functional
    and type-safe design philosophy.

    If Mojo later adds variadic generic parameters (*Ts: Module), these two
    structs can be unified into a single Sequential[*Ts: Module].

Usage:
    ```mojo
    from .sequential import Sequential2, Sequential3, Sequential4, Sequential5
    from .layers import Linear, ReLULayer

    # Five-layer model (common for deeper MLPs like LeNet-5 classifier)
    var model = Sequential5[Linear, ReLULayer, Linear, ReLULayer, Linear](
        Linear(10, 8),
        ReLULayer(),
        Linear(8, 6),
        ReLULayer(),
        Linear(6, 2),
    )
    var input = zeros([4, 10], DType.float32)
    var output = model.forward(input)  # Shape: [4, 2]
    ```

See Also:
    - shared.core.module: Module trait definition
    - shared.core.layers: Available layer implementations
"""

from shared.tensor.any_tensor import AnyTensor
from .module import Module


struct Sequential2[T0: Module & Movable, T1: Module & Movable](
    ImplicitlyDestructible, Movable
):
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

    var layer0: Self.T0
    var layer1: Self.T1

    def __init__(out self, var layer0: Self.T0, var layer1: Self.T1):
        """Initialize with two layers.

        Args:
            layer0: First layer (takes the original input).
            layer1: Second layer (takes output of layer0).
        """
        self.layer0 = layer0^
        self.layer1 = layer1^

    def forward(mut self, input: AnyTensor) raises -> AnyTensor:
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

    def parameters(self) raises -> List[AnyTensor]:
        """Collect trainable parameters from both layers.

        Returns:
            List of AnyTensor containing parameters from layer0 then layer1.

        Raises:
            Error: If parameter collection fails.
        """
        var params: List[AnyTensor] = []
        var p0 = self.layer0.parameters()
        var p1 = self.layer1.parameters()
        for i in range(len(p0)):
            params.append(p0[i])
        for i in range(len(p1)):
            params.append(p1[i])
        return params^

    def train(mut self):
        """Switch both layers to training mode."""
        self.layer0.train()
        self.layer1.train()

    def eval(mut self):
        """Switch both layers to evaluation mode."""
        self.layer0.eval()
        self.layer1.eval()


struct Sequential3[
    T0: Module & Movable, T1: Module & Movable, T2: Module & Movable
](ImplicitlyDestructible, Movable):
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

    var layer0: Self.T0
    var layer1: Self.T1
    var layer2: Self.T2

    def __init__(
        out self,
        var layer0: Self.T0,
        var layer1: Self.T1,
        var layer2: Self.T2,
    ):
        """Initialize with three layers.

        Args:
            layer0: First layer (takes the original input).
            layer1: Second layer (takes output of layer0).
            layer2: Third layer (takes output of layer1).
        """
        self.layer0 = layer0^
        self.layer1 = layer1^
        self.layer2 = layer2^

    def forward(mut self, input: AnyTensor) raises -> AnyTensor:
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

    def parameters(self) raises -> List[AnyTensor]:
        """Collect trainable parameters from all three layers.

        Returns:
            List of AnyTensor containing parameters from layer0, layer1, layer2.

        Raises:
            Error: If parameter collection fails.
        """
        var params: List[AnyTensor] = []
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

    def train(mut self):
        """Switch all three layers to training mode."""
        self.layer0.train()
        self.layer1.train()
        self.layer2.train()

    def eval(mut self):
        """Switch all three layers to evaluation mode."""
        self.layer0.eval()
        self.layer1.eval()
        self.layer2.eval()


struct Sequential4[
    T0: Module & Movable,
    T1: Module & Movable,
    T2: Module & Movable,
    T3: Module & Movable,
](ImplicitlyDestructible, Movable):
    """Four-layer sequential module container.

    Chains four Module-conforming layers in order: 0 -> 1 -> 2 -> 3.
    Parameters from all four layers are collected together, and
    train/eval mode is propagated to all four layers.

    Parameters:
        T0: Type of the first layer, must implement Module.
        T1: Type of the second layer, must implement Module.
        T2: Type of the third layer, must implement Module.
        T3: Type of the fourth layer, must implement Module.

    Attributes:
        layer0: First layer in the sequence.
        layer1: Second layer in the sequence.
        layer2: Third layer in the sequence.
        layer3: Fourth layer in the sequence.

    Example:
        ```mojo
        var seq = Sequential4[Linear, ReLULayer, Linear, ReLULayer](
            Linear(10, 8), ReLULayer(), Linear(8, 6), ReLULayer()
        )
        var out = seq.forward(input)
        ```
    """

    var layer0: Self.T0
    var layer1: Self.T1
    var layer2: Self.T2
    var layer3: Self.T3

    def __init__(
        out self,
        var layer0: Self.T0,
        var layer1: Self.T1,
        var layer2: Self.T2,
        var layer3: Self.T3,
    ):
        """Initialize with four layers.

        Args:
            layer0: First layer (takes the original input).
            layer1: Second layer (takes output of layer0).
            layer2: Third layer (takes output of layer1).
            layer3: Fourth layer (takes output of layer2).
        """
        self.layer0 = layer0^
        self.layer1 = layer1^
        self.layer2 = layer2^
        self.layer3 = layer3^

    def forward(mut self, input: AnyTensor) raises -> AnyTensor:
        """Compute chained forward pass through all four layers.

        Args:
            input: Input tensor for layer0.

        Returns:
            Output tensor from layer3.

        Raises:
            Error: If any layer's forward pass fails.
        """
        var out0 = self.layer0.forward(input)
        var out1 = self.layer1.forward(out0)
        var out2 = self.layer2.forward(out1)
        return self.layer3.forward(out2)

    def parameters(self) raises -> List[AnyTensor]:
        """Collect trainable parameters from all four layers.

        Returns:
            List of AnyTensor containing parameters from all layers.

        Raises:
            Error: If parameter collection fails.
        """
        var params: List[AnyTensor] = []
        var p0 = self.layer0.parameters()
        var p1 = self.layer1.parameters()
        var p2 = self.layer2.parameters()
        var p3 = self.layer3.parameters()
        for i in range(len(p0)):
            params.append(p0[i])
        for i in range(len(p1)):
            params.append(p1[i])
        for i in range(len(p2)):
            params.append(p2[i])
        for i in range(len(p3)):
            params.append(p3[i])
        return params^

    def train(mut self):
        """Switch all four layers to training mode."""
        self.layer0.train()
        self.layer1.train()
        self.layer2.train()
        self.layer3.train()

    def eval(mut self):
        """Switch all four layers to evaluation mode."""
        self.layer0.eval()
        self.layer1.eval()
        self.layer2.eval()
        self.layer3.eval()


struct Sequential5[
    T0: Module & Movable,
    T1: Module & Movable,
    T2: Module & Movable,
    T3: Module & Movable,
    T4: Module & Movable,
](ImplicitlyDestructible, Movable):
    """Five-layer sequential module container.

    Chains five Module-conforming layers in order: 0 -> 1 -> 2 -> 3 -> 4.
    Parameters from all five layers are collected together, and
    train/eval mode is propagated to all five layers.

    Parameters:
        T0: Type of the first layer, must implement Module.
        T1: Type of the second layer, must implement Module.
        T2: Type of the third layer, must implement Module.
        T3: Type of the fourth layer, must implement Module.
        T4: Type of the fifth layer, must implement Module.

    Attributes:
        layer0: First layer in the sequence.
        layer1: Second layer in the sequence.
        layer2: Third layer in the sequence.
        layer3: Fourth layer in the sequence.
        layer4: Fifth layer in the sequence.

    Example:
        ```mojo
        var seq = Sequential5[Linear, ReLULayer, Linear, ReLULayer, Linear](
            Linear(10, 8), ReLULayer(), Linear(8, 6), ReLULayer(), Linear(6, 2)
        )
        var out = seq.forward(input)
        ```
    """

    var layer0: Self.T0
    var layer1: Self.T1
    var layer2: Self.T2
    var layer3: Self.T3
    var layer4: Self.T4

    def __init__(
        out self,
        var layer0: Self.T0,
        var layer1: Self.T1,
        var layer2: Self.T2,
        var layer3: Self.T3,
        var layer4: Self.T4,
    ):
        """Initialize with five layers.

        Args:
            layer0: First layer (takes the original input).
            layer1: Second layer (takes output of layer0).
            layer2: Third layer (takes output of layer1).
            layer3: Fourth layer (takes output of layer2).
            layer4: Fifth layer (takes output of layer3).
        """
        self.layer0 = layer0^
        self.layer1 = layer1^
        self.layer2 = layer2^
        self.layer3 = layer3^
        self.layer4 = layer4^

    def forward(mut self, input: AnyTensor) raises -> AnyTensor:
        """Compute chained forward pass through all five layers.

        Args:
            input: Input tensor for layer0.

        Returns:
            Output tensor from layer4.

        Raises:
            Error: If any layer's forward pass fails.
        """
        var out0 = self.layer0.forward(input)
        var out1 = self.layer1.forward(out0)
        var out2 = self.layer2.forward(out1)
        var out3 = self.layer3.forward(out2)
        return self.layer4.forward(out3)

    def parameters(self) raises -> List[AnyTensor]:
        """Collect trainable parameters from all five layers.

        Returns:
            List of AnyTensor containing parameters from all layers.

        Raises:
            Error: If parameter collection fails.
        """
        var params: List[AnyTensor] = []
        var p0 = self.layer0.parameters()
        var p1 = self.layer1.parameters()
        var p2 = self.layer2.parameters()
        var p3 = self.layer3.parameters()
        var p4 = self.layer4.parameters()
        for i in range(len(p0)):
            params.append(p0[i])
        for i in range(len(p1)):
            params.append(p1[i])
        for i in range(len(p2)):
            params.append(p2[i])
        for i in range(len(p3)):
            params.append(p3[i])
        for i in range(len(p4)):
            params.append(p4[i])
        return params^

    def train(mut self):
        """Switch all five layers to training mode."""
        self.layer0.train()
        self.layer1.train()
        self.layer2.train()
        self.layer3.train()
        self.layer4.train()

    def eval(mut self):
        """Switch all five layers to evaluation mode."""
        self.layer0.eval()
        self.layer1.eval()
        self.layer2.eval()
        self.layer3.eval()
        self.layer4.eval()
