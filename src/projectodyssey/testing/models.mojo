"""Consolidated test model architectures for ML Odyssey testing.

This module provides simple model implementations for testing neural network
components, training loops, and validation without requiring full model
implementations.

Test Models:
    - SimpleCNN: Minimal CNN for testing image processing
    - LinearModel: Single fully-connected layer for basic testing
    - SimpleMLP: Multi-layer perceptron (2-3 layers) for composition testing
    - SimpleLinearModel: Linear model with weights and bias
    - MockLayer: Minimal layer with identity/scaled transformation

All models use simple operations and predictable behavior for testing:
- Fixed initialization for reproducible tests
- Placeholder forward passes with correct shape handling
- Parameter tracking for validation

Example:
    from projectodyssey.testing import SimpleCNN, LinearModel, SimpleMLP

    def test_model_forward():
        # Test CNN forward pass
        var cnn_model = SimpleCNN(1, 8, 10)
        var input = ones([32, 1, 28, 28], DType.float32)
        var output = cnn_model.forward(input)

        # Test linear model
        var linear = LinearModel(784, 10)
        var linear_input = ones([32, 784], DType.float32)
        var linear_output = linear.forward(linear_input)

        # Test MLP
        var mlp = SimpleMLP(10, 20, 5, num_hidden_layers=1)
        var mlp_input = zeros([10], DType.float32)
        var mlp_output = mlp.forward(mlp_input)
    ```
"""

from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.tensor.tensor_creation import ones, zeros, zeros_like
from projectodyssey.core.traits import Model


# ============================================================================
# Simple CNN Model
# ============================================================================


struct SimpleCNN(Copyable, Movable):
    """Minimal CNN for testing purposes.

    A very simple 2-layer CNN for testing infrastructure without the
    complexity of real models. Useful for:
    - Testing training loops
    - Validating gradient computation
    - Checking data pipeline integration
    - Performance benchmarking

    Shape flow:
        Input: (batch_size, in_channels, height, width)
        Output: (batch_size, num_classes)

    Attributes:
        in_channels: Number of input channels.
        out_channels: Number of output channels from first conv layer.
        num_classes: Number of output classes.
    """

    var in_channels: Int
    """Number of input channels."""
    var out_channels: Int
    """Number of output channels from first conv layer."""
    var num_classes: Int
    """Number of output classes."""

    def __init__(
        out self,
        in_channels: Int = 1,
        out_channels: Int = 8,
        num_classes: Int = 10,
    ):
        """Initialize simple CNN.

        Args:
            in_channels: Number of input channels (default: 1 for MNIST-like).
            out_channels: Number of output channels from first conv (default: 8).
            num_classes: Number of output classes (default: 10).

        Example:
            ```mojo
            var model = SimpleCNN(1, 8, 10)
            assert_equal(model.num_classes, 10)
        ```
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_classes = num_classes

    def get_output_shape(self, batch_size: Int) -> List[Int]:
        """Get output shape for given batch size.

        Args:
            batch_size: Number of samples in batch.

        Returns:
            Output shape (batch_size, num_classes).
        """
        var shape = List[Int]()
        shape.append(batch_size)
        shape.append(self.num_classes)
        return shape^

    def forward(self, input: AnyTensor) raises -> AnyTensor:
        """Forward pass (simplified for testing).

        Creates output tensor with correct shape filled with dummy values.
        In real implementation, this would include conv, pool, fc layers.

        Args:
            input: Input tensor (batch_size, in_channels, height, width).

        Returns:
            Output tensor (batch_size, num_classes).

        Raises:
            Error: If tensor creation fails.

        Note:
            This is a placeholder for testing. Real implementations should
            contain actual neural network operations.
        """
        var batch_size = input._shape[0]
        var output_shape = self.get_output_shape(batch_size)
        var output = zeros(output_shape, input._dtype)

        # Fill with dummy values (0.1 per element)
        for i in range(output.numel()):
            output._set_float64(i, 0.1)

        return output


# ============================================================================
# Simple Linear Model (Single Layer)
# ============================================================================


struct LinearModel(Copyable, Movable):
    """Simple linear model for testing.

    A single fully-connected layer for basic testing of:
    - Linear transformations
    - Batch processing
    - Gradient computation
    - Loss functions

    Shape flow:
        Input: (batch_size, in_features)
        Output: (batch_size, out_features)

    Attributes:
        in_features: Input feature dimension.
        out_features: Output feature dimension.
    """

    var in_features: Int
    """Input feature dimension."""
    var out_features: Int
    """Output feature dimension."""

    def __init__(out self, in_features: Int, out_features: Int):
        """Initialize linear model.

        Args:
            in_features: Input dimension.
            out_features: Output dimension.

        Example:
            ```mojo
            var model = LinearModel(784, 10)
            assert_equal(model.in_features, 784)
        ```
        """
        self.in_features = in_features
        self.out_features = out_features

    def get_output_shape(self, batch_size: Int) -> List[Int]:
        """Get output shape for given batch size.

        Args:
            batch_size: Number of samples in batch.

        Returns:
            Output shape (batch_size, out_features).
        """
        var shape = List[Int]()
        shape.append(batch_size)
        shape.append(self.out_features)
        return shape^

    def forward(self, input: AnyTensor) raises -> AnyTensor:
        """Forward pass.

        Creates output tensor with correct shape filled with zeros.
        In real implementation, this would compute y = xW^T + b.

        Args:
            input: Input tensor (batch_size, in_features).

        Returns:
            Output tensor (batch_size, out_features).

        Raises:
            Error: If tensor creation fails.

        Note:
            This is a placeholder for testing. Real implementations should
            contain actual linear transformation: y = matmul(input, weights^T) + bias.
        """
        var batch_size = input._shape[0]
        var output_shape = self.get_output_shape(batch_size)
        var output = zeros(output_shape, input._dtype)

        # Already zero-filled by zeros(), no need to fill again
        return output


# ============================================================================
# Mock Layer Implementation
# ============================================================================


struct MockLayer:
    """Minimal mock layer for testing.

    Provides a simple layer that performs identity transformation
    or scaled transformation for testing purposes.

    Attributes:
        input_dim: Input feature dimension.
        output_dim: Output feature dimension.
        scale: Scale factor applied to inputs.
    """

    var input_dim: Int
    """Input feature dimension."""
    var output_dim: Int
    """Output feature dimension."""
    var scale: Float32
    """Scale factor applied to inputs."""

    def __init__(
        out self, input_dim: Int, output_dim: Int, scale: Float32 = 1.0
    ):
        """Initialize mock layer.

        Args:
            input_dim: Input dimension.
            output_dim: Output dimension.
            scale: Scale factor for transformation (default: 1.0).

        Example:
            ```mojo
            var layer = MockLayer(input_dim=10, output_dim=5, scale=2.0)
        ```

        Note:
            This is a simplified layer for testing - not a real neural network layer.
            It performs a simple scaled identity or zero-padding transformation.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.scale = scale

    def forward(self, input: List[Float32]) -> List[Float32]:
        """Forward pass through mock layer.

        Args:
            input: Input tensor (flat list of size input_dim).

        Returns:
            Output tensor (flat list of size output_dim).

        Example:
            ```mojo
            var layer = MockLayer(10, 5, scale=2.0)
            var input : List[Float32] = [10]
            var output = layer.forward(input)
            # output contains first 5 elements scaled by 2.0
        ```

        Note:
            If output_dim < input_dim: truncates and scales.
            If output_dim > input_dim: pads with zeros after scaling.
            If output_dim == input_dim: scales all elements.
        """
        var output = List[Float32](capacity=self.output_dim)

        if self.output_dim <= self.input_dim:
            # Truncate and scale
            for i in range(self.output_dim):
                output.append(input[i] * self.scale)
        else:
            # Scale input portion
            for i in range(self.input_dim):
                output.append(input[i] * self.scale)
            # Pad with zeros
            for _ in range(self.input_dim, self.output_dim):
                output.append(0.0)

        return output^

    def num_parameters(self) -> Int:
        """Get number of trainable parameters.

        Returns:
            Number of parameters (for testing parameter counting).

        Example:
            ```mojo
            var layer = MockLayer(10, 5)
            var n_params = layer.num_parameters()  # 50 (10*5)
        ```

        Note:
            Returns input_dim * output_dim to simulate weight matrix size.
        """
        return self.input_dim * self.output_dim


# ============================================================================
# Simple Linear Model (with weights and bias)
# ============================================================================


struct SimpleLinearModel:
    """Simple linear model (single layer).

    Provides a minimal single-layer linear model for testing basic
    forward passes and training workflows.

    Attributes:
        input_dim: Input feature dimension.
        output_dim: Output dimension.
        weights: Model weights (flattened weight matrix).
        bias: Model bias.
        use_bias: Whether to use bias.
    """

    var input_dim: Int
    """Input feature dimension."""
    var output_dim: Int
    """Output feature dimension."""
    var weights: List[Float32]
    """Model weights (flattened weight matrix)."""
    var bias: List[Float32]
    """Model bias."""
    var use_bias: Bool
    """Whether to use bias."""

    def __init__(
        out self,
        input_dim: Int,
        output_dim: Int,
        use_bias: Bool = True,
        init_value: Float32 = 0.1,
    ):
        """Initialize simple linear model.

        Args:
            input_dim: Input dimension.
            output_dim: Output dimension.
            use_bias: Whether to use bias (default: True).
            init_value: Initial weight/bias value (default: 0.1).

        Example:
            ```mojo
            var model = SimpleLinearModel(
                input_dim=10,
                output_dim=5,
                use_bias=True,
                init_value=0.1
            )
        ```

        Note:
            Weights are initialized to constant value for predictable testing.
            For random initialization, use create_random_weights() after creation.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias

        # Initialize weights: output_dim x input_dim (flattened)
        var n_weights = output_dim * input_dim
        self.weights = List[Float32](capacity=n_weights)
        for _ in range(n_weights):
            self.weights.append(init_value)

        # Initialize bias
        self.bias = List[Float32](capacity=output_dim)
        if use_bias:
            for _ in range(output_dim):
                self.bias.append(init_value)

    def forward(self, input: List[Float32]) -> List[Float32]:
        """Forward pass: output = weights @ input + bias.

        Args:
            input: Input tensor (flat list of size input_dim).

        Returns:
            Output tensor (flat list of size output_dim).

        Example:
            ```mojo
            var model = SimpleLinearModel(10, 5)
            var input : List[Float32] = [10]
            var output = model.forward(input)
            # output = weights @ input + bias
        ```

        Note:
            Performs matrix-vector multiplication: y = W @ x + b.
            where W is output_dim x input_dim weight matrix.
        """
        var output = List[Float32](capacity=self.output_dim)

        # Compute W @ x
        for i in range(self.output_dim):
            var sum = Float32(0.0)
            for j in range(self.input_dim):
                var weight_idx = i * self.input_dim + j
                sum += self.weights[weight_idx] * input[j]

            # Add bias if used
            if self.use_bias:
                sum += self.bias[i]

            output.append(sum)

        return output^

    def num_parameters(self) -> Int:
        """Get total number of parameters.

        Returns:
            Total parameters (weights + bias).

        Example:
            ```mojo
            var model = SimpleLinearModel(10, 5, use_bias=True)
            var n_params = model.num_parameters()  # 55 (50 + 5)
        ```
        """
        var n_params = len(self.weights)
        if self.use_bias:
            n_params += len(self.bias)
        return n_params


# ============================================================================
# Parameter Structure
# ============================================================================


struct Parameter(Copyable, Movable):
    """Trainable parameter with data and gradient.

    Wraps an AnyTensor for the parameter data with an associated gradient tensor.
    Used in model layers to track trainable weights and biases.

    Attributes:
        data: The parameter tensor (weights or bias).
        grad: The gradient tensor for backpropagation.
    """

    var data: AnyTensor
    """The parameter tensor (weights or bias)."""
    var grad: AnyTensor
    """The gradient tensor for backpropagation."""

    def __init__(out self, data: AnyTensor) raises:
        """Initialize parameter with data tensor and zero gradient.

        Args:
            data: The parameter tensor.

        Raises:
            Error: If gradient tensor cannot be created or if memory allocation fails.

        Example:
            ```mojo
            var param = Parameter(data_tensor)
            # grad is automatically initialized to zeros_like(data_tensor)
        ```
        """
        self.data = data
        self.grad = zeros_like(data)

    def shape(self) -> List[Int]:
        """Get the shape of the parameter.

        Returns:
            Shape of the parameter tensor.

        Example:
            ```mojo
            var param = Parameter(tensor)
            var s = param.shape()  # Same as param.data.shape()
        ```
        """
        return self.data.shape()


# ============================================================================
# Simple Multi-Layer Perceptron
# ============================================================================


struct SimpleMLP(Copyable, Model, Movable):
    """Simple multi-layer perceptron (2-3 layers).

    Provides a minimal MLP for testing multi-layer forward passes,
    training loops, and model composition. Implements the original
    direct-layer approach for compatibility.

    Attributes:
        input_dim: Input dimension.
        hidden_dim: Hidden layer dimension.
        output_dim: Output dimension.
        num_hidden_layers: Number of hidden layers (1 or 2).
        layer1_weights: First layer weights.
        layer1_bias: First layer bias.
        layer2_weights: Second layer weights (hidden to hidden or hidden to output).
        layer2_bias: Second layer bias.
        layer3_weights: Third layer weights (only if num_hidden_layers=2).
        layer3_bias: Third layer bias (only if num_hidden_layers=2).

    Note:
        For Sequential-based composition examples, see SimpleMLP2 variant
        in the same module, which demonstrates real-world usage of
        Sequential2/Sequential3 containers.
    """

    var input_dim: Int
    """Input dimension."""
    var hidden_dim: Int
    """Hidden layer dimension."""
    var output_dim: Int
    """Output dimension."""
    var num_hidden_layers: Int
    """Number of hidden layers (1 or 2)."""
    var layer1_weights: List[Float32]
    """First layer weights."""
    var layer1_bias: List[Float32]
    """First layer bias."""
    var layer2_weights: List[Float32]
    """Second layer weights (hidden to hidden or hidden to output)."""
    var layer2_bias: List[Float32]
    """Second layer bias."""
    var layer3_weights: List[Float32]
    """Third layer weights (only if num_hidden_layers=2)."""
    var layer3_bias: List[Float32]
    """Third layer bias (only if num_hidden_layers=2)."""

    def __init__(
        out self,
        input_dim: Int,
        hidden_dim: Int,
        output_dim: Int,
        num_hidden_layers: Int = 1,
        init_value: Float32 = 0.1,
    ):
        """Initialize simple MLP.

        Args:
            input_dim: Input dimension.
            hidden_dim: Hidden layer dimension.
            output_dim: Output dimension.
            num_hidden_layers: Number of hidden layers (1 or 2, default: 1).
            init_value: Initial weight/bias value (default: 0.1).

        Example:
            ```mojo
             2-layer MLP: input -> hidden -> output
            var mlp = SimpleMLP(
                input_dim=10,
                hidden_dim=20,
                output_dim=5,
                num_hidden_layers=1
            )

            # 3-layer MLP: input -> hidden -> hidden -> output
            var deep_mlp = SimpleMLP(
                input_dim=10,
                hidden_dim=20,
                output_dim=5,
                num_hidden_layers=2
            )
        ```
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_hidden_layers = num_hidden_layers

        # Layer 1: input -> hidden
        var n_weights1 = hidden_dim * input_dim
        self.layer1_weights = List[Float32](capacity=n_weights1)
        self.layer1_bias = List[Float32](capacity=hidden_dim)
        for _ in range(n_weights1):
            self.layer1_weights.append(init_value)
        for _ in range(hidden_dim):
            self.layer1_bias.append(init_value)

        # Layer 2: depends on num_hidden_layers
        if num_hidden_layers == 1:
            # hidden -> output
            var n_weights2 = output_dim * hidden_dim
            self.layer2_weights = List[Float32](capacity=n_weights2)
            self.layer2_bias = List[Float32](capacity=output_dim)
            for _ in range(n_weights2):
                self.layer2_weights.append(init_value)
            for _ in range(output_dim):
                self.layer2_bias.append(init_value)

            # Layer 3 unused
            self.layer3_weights = List[Float32]()
            self.layer3_bias = List[Float32]()
        else:
            # hidden -> hidden
            var n_weights2 = hidden_dim * hidden_dim
            self.layer2_weights = List[Float32](capacity=n_weights2)
            self.layer2_bias = List[Float32](capacity=hidden_dim)
            for _ in range(n_weights2):
                self.layer2_weights.append(init_value)
            for _ in range(hidden_dim):
                self.layer2_bias.append(init_value)

            # Layer 3: hidden -> output
            var n_weights3 = output_dim * hidden_dim
            self.layer3_weights = List[Float32](capacity=n_weights3)
            self.layer3_bias = List[Float32](capacity=output_dim)
            for _ in range(n_weights3):
                self.layer3_weights.append(init_value)
            for _ in range(output_dim):
                self.layer3_bias.append(init_value)

    def forward(self, input: List[Float32]) -> List[Float32]:
        """Forward pass through MLP.

        Args:
            input: Input tensor.

        Returns:
            Output tensor.

        Note:
            Uses ReLU activation between layers: ReLU(x) = max(0, x).

        Example:
            ```mojo
            var mlp = SimpleMLP(10, 20, 5)
            var input : List[Float32] = [10]
            var output = mlp.forward(input)
            # output has shape [5]
        ```
        """
        # Layer 1: input -> hidden
        var hidden1 = self._linear_forward(
            input,
            self.layer1_weights,
            self.layer1_bias,
            self.hidden_dim,
            self.input_dim,
        )
        hidden1 = self._relu(hidden1)

        # Layer 2
        if self.num_hidden_layers == 1:
            # hidden -> output
            var output = self._linear_forward(
                hidden1,
                self.layer2_weights,
                self.layer2_bias,
                self.output_dim,
                self.hidden_dim,
            )
            return output^
        else:
            # hidden -> hidden
            var hidden2 = self._linear_forward(
                hidden1,
                self.layer2_weights,
                self.layer2_bias,
                self.hidden_dim,
                self.hidden_dim,
            )
            hidden2 = self._relu(hidden2)

            # Layer 3: hidden -> output
            var output = self._linear_forward(
                hidden2,
                self.layer3_weights,
                self.layer3_bias,
                self.output_dim,
                self.hidden_dim,
            )
            return output^

    def forward(mut self, input: AnyTensor) raises -> AnyTensor:
        """Forward pass through MLP with AnyTensor input.

        Args:
            input: Input AnyTensor.

        Returns:
            Output AnyTensor.

        Raises:
            Error: If tensor conversion fails or memory allocation fails.

        Example:
            ```mojo
            var mlp = SimpleMLP(10, 20, 5)
            var input = zeros([10], DType.float32)
            var output = mlp.forward(input)
            # output has shape [5]
        ```

        Note:
            This is the Model trait implementation that accepts AnyTensor.
            Uses ReLU activation between layers: ReLU(x) = max(0, x).
        """
        # Extract Float32 values from AnyTensor
        var input_list = List[Float32]()
        var input_numel = input.numel()
        for i in range(input_numel):
            input_list.append(input._get_float32(i))

        # Call existing forward with List[Float32]
        var output_list = self.forward(input_list)

        # Convert back to AnyTensor
        var output = zeros([len(output_list)], DType.float32)
        for i in range(len(output_list)):
            output._set_float32(i, output_list[i])

        return output^

    def _linear_forward(
        self,
        input: List[Float32],
        weights: List[Float32],
        bias: List[Float32],
        out_dim: Int,
        in_dim: Int,
    ) -> List[Float32]:
        """Linear layer forward pass: output = weights @ input + bias.

        Args:
            input: Input list.
            weights: Weight matrix (flattened).
            bias: Bias vector.
            out_dim: Output dimension.
            in_dim: Input dimension.

        Returns:
            Output tensor from linear transformation.
        """
        var output = List[Float32](capacity=out_dim)

        for i in range(out_dim):
            var sum = Float32(0.0)
            for j in range(in_dim):
                var weight_idx = i * in_dim + j
                sum += weights[weight_idx] * input[j]
            sum += bias[i]
            output.append(sum)

        return output^

    def _relu(self, input: List[Float32]) -> List[Float32]:
        """ReLU activation: max(0, x).

        Args:
            input: Input tensor.

        Returns:
            Output tensor with ReLU applied.
        """
        var output = List[Float32](capacity=len(input))
        for i in range(len(input)):
            var val = input[i]
            output.append(max(Float32(0.0), val))
        return output^

    def num_parameters(self) -> Int:
        """Get total number of parameters.

        Returns:
            Total parameters across all layers.
        """
        var total = len(self.layer1_weights) + len(self.layer1_bias)
        total += len(self.layer2_weights) + len(self.layer2_bias)
        if self.num_hidden_layers == 2:
            total += len(self.layer3_weights) + len(self.layer3_bias)
        return total

    def get_weights(self) raises -> AnyTensor:
        """Get flattened weights as AnyTensor.

        Returns:
            AnyTensor containing all weights flattened.

        Raises:
            Error: If tensor creation fails.

        Example:
            ```mojo
            var mlp = SimpleMLP(10, 20, 5)
            var weights = mlp.get_weights()
        ```
        """
        # Flatten all weight and bias lists into a single list
        var all_weights = List[Float32]()
        for i in range(len(self.layer1_weights)):
            all_weights.append(self.layer1_weights[i])
        for i in range(len(self.layer1_bias)):
            all_weights.append(self.layer1_bias[i])
        for i in range(len(self.layer2_weights)):
            all_weights.append(self.layer2_weights[i])
        for i in range(len(self.layer2_bias)):
            all_weights.append(self.layer2_bias[i])
        for i in range(len(self.layer3_weights)):
            all_weights.append(self.layer3_weights[i])
        for i in range(len(self.layer3_bias)):
            all_weights.append(self.layer3_bias[i])

        # Create AnyTensor from flattened weights
        var shape: List[Int] = [len(all_weights)]
        var tensor = zeros(shape, DType.float32)

        # Copy weights into tensor
        for i in range(len(all_weights)):
            tensor._set_float32(i, all_weights[i])

        return tensor

    def parameters(self) raises -> List[AnyTensor]:
        """Get list of parameter tensors.

        Returns:
            List of AnyTensor objects containing weights and biases.

        Raises:
            Error: If tensor creation fails.

        Example:
            ```mojo
            var mlp = SimpleMLP(10, 20, 5)
            var params = mlp.parameters()
            for param in params:
                print(param.shape())
        ```
        """
        var param_list: List[AnyTensor] = []

        # Layer 1 weights parameter
        var w1_shape: List[Int] = [self.hidden_dim, self.input_dim]
        var w1_tensor = zeros(w1_shape, DType.float32)
        for i in range(len(self.layer1_weights)):
            w1_tensor._set_float32(i, self.layer1_weights[i])
        param_list.append(w1_tensor^)

        # Layer 1 bias parameter
        var b1_shape: List[Int] = [self.hidden_dim]
        var b1_tensor = zeros(b1_shape, DType.float32)
        for i in range(len(self.layer1_bias)):
            b1_tensor._set_float32(i, self.layer1_bias[i])
        param_list.append(b1_tensor^)

        # Layer 2 weights parameter
        var w2_dim_in = self.hidden_dim
        var w2_dim_out = (
            self.output_dim if self.num_hidden_layers == 1 else self.hidden_dim
        )
        var w2_shape: List[Int] = [w2_dim_out, w2_dim_in]
        var w2_tensor = zeros(w2_shape, DType.float32)
        for i in range(len(self.layer2_weights)):
            w2_tensor._set_float32(i, self.layer2_weights[i])
        param_list.append(w2_tensor^)

        # Layer 2 bias parameter
        var b2_shape: List[Int] = [w2_dim_out]
        var b2_tensor = zeros(b2_shape, DType.float32)
        for i in range(len(self.layer2_bias)):
            b2_tensor._set_float32(i, self.layer2_bias[i])
        param_list.append(b2_tensor^)

        # Layer 3 parameters (only if num_hidden_layers == 2)
        if self.num_hidden_layers == 2:
            var w3_shape: List[Int] = [self.output_dim, self.hidden_dim]
            var w3_tensor = zeros(w3_shape, DType.float32)
            for i in range(len(self.layer3_weights)):
                w3_tensor._set_float32(i, self.layer3_weights[i])
            param_list.append(w3_tensor^)

            var b3_shape: List[Int] = [self.output_dim]
            var b3_tensor = zeros(b3_shape, DType.float32)
            for i in range(len(self.layer3_bias)):
                b3_tensor._set_float32(i, self.layer3_bias[i])
            param_list.append(b3_tensor^)

        return param_list^

    def zero_grad(mut self) raises:
        """Zero all gradients.

        Example:
            ```mojo
            var mlp = SimpleMLP(10, 20, 5)
            mlp.zero_grad()
        ```

        Raises:
            Error: If operation fails.
        """
        # This is a placeholder - actual gradient zeroing would happen
        # on Parameter objects during backpropagation
        pass

    def state_dict(self) raises -> Dict[String, AnyTensor]:
        """Export weights as state dictionary.

        Returns:
            Dictionary mapping parameter names to their tensor values.

        Raises:
            Error: If tensor creation fails.

        Example:
            ```mojo
            var mlp = SimpleMLP(10, 20, 5)
            var state = mlp.state_dict()
        ```

        Note:
            Returns var Dict with ownership transfer to avoid copy errors.
        """
        var state = Dict[String, AnyTensor]()

        # Layer 1 weights
        var w1_shape: List[Int] = [self.hidden_dim, self.input_dim]
        var w1_tensor = zeros(w1_shape, DType.float32)
        for i in range(len(self.layer1_weights)):
            w1_tensor._set_float32(i, self.layer1_weights[i])
        state["layer1_weights"] = w1_tensor

        # Layer 1 bias
        var b1_shape: List[Int] = [self.hidden_dim]
        var b1_tensor = zeros(b1_shape, DType.float32)
        for i in range(len(self.layer1_bias)):
            b1_tensor._set_float32(i, self.layer1_bias[i])
        state["layer1_bias"] = b1_tensor

        # Layer 2 weights
        var w2_dim_in = self.hidden_dim
        var w2_dim_out = (
            self.output_dim if self.num_hidden_layers == 1 else self.hidden_dim
        )
        var w2_shape: List[Int] = [w2_dim_out, w2_dim_in]
        var w2_tensor = zeros(w2_shape, DType.float32)
        for i in range(len(self.layer2_weights)):
            w2_tensor._set_float32(i, self.layer2_weights[i])
        state["layer2_weights"] = w2_tensor

        # Layer 2 bias
        var b2_shape: List[Int] = [w2_dim_out]
        var b2_tensor = zeros(b2_shape, DType.float32)
        for i in range(len(self.layer2_bias)):
            b2_tensor._set_float32(i, self.layer2_bias[i])
        state["layer2_bias"] = b2_tensor

        # Layer 3 parameters (only if num_hidden_layers == 2)
        if self.num_hidden_layers == 2:
            var w3_shape: List[Int] = [self.output_dim, self.hidden_dim]
            var w3_tensor = zeros(w3_shape, DType.float32)
            for i in range(len(self.layer3_weights)):
                w3_tensor._set_float32(i, self.layer3_weights[i])
            state["layer3_weights"] = w3_tensor

            var b3_shape: List[Int] = [self.output_dim]
            var b3_tensor = zeros(b3_shape, DType.float32)
            for i in range(len(self.layer3_bias)):
                b3_tensor._set_float32(i, self.layer3_bias[i])
            state["layer3_bias"] = b3_tensor

        return state^

    def load_state_dict(mut self, state: Dict[String, AnyTensor]):
        """Load weights from state dictionary.

        Args:
            state: Dictionary mapping parameter names to their tensor values.

        Example:
            ```mojo
            var mlp = SimpleMLP(10, 20, 5)
            mlp.load_state_dict(state)
        ```

        Note:
            This is a placeholder implementation for now.
        """
        # Placeholder for weight loading
        pass


# ============================================================================
# Sequential-Based MLP Variant
# ============================================================================


struct SimpleMLP2(Copyable, Model, Movable):
    """Simple MLP using Sequential2 composition (2-layer variant).

    This variant demonstrates real-world usage of the Sequential2 container,
    providing both a direct-layer approach (like SimpleMLP) and container-based
    composition for testing Sequential integration.

    Shape flow:
        Input: (batch_size, input_dim)
        Output: (batch_size, output_dim)

    Example:
        ```mojo
        var mlp = SimpleMLP2(10, 20, 5)
        var input = zeros([32, 10], DType.float32)
        var output = mlp.forward(input)  # Shape: [32, 5]
        ```

    Note:
        This struct wraps a Sequential2-based composition internally,
        demonstrating how Sequential containers integrate with the Model trait.
    """

    var input_dim: Int
    """Input dimension."""
    var hidden_dim: Int
    """Hidden layer dimension."""
    var output_dim: Int
    """Output dimension."""

    def __init__(
        out self,
        input_dim: Int,
        hidden_dim: Int,
        output_dim: Int,
    ):
        """Initialize Sequential-based MLP.

        Args:
            input_dim: Input dimension.
            hidden_dim: Hidden layer dimension.
            output_dim: Output dimension.

        Example:
            ```mojo
            var mlp = SimpleMLP2(10, 20, 5)
            assert_equal(mlp.input_dim, 10)
            assert_equal(mlp.hidden_dim, 20)
            assert_equal(mlp.output_dim, 5)
            ```
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def forward(mut self, input: AnyTensor) raises -> AnyTensor:
        """Forward pass using manual layer composition.

        This demonstrates the data flow through sequential layers without
        requiring full Sequential container integration (which requires
        Layer trait implementation). It shows how SimpleMLP2 would use
        Sequential2 if layers implemented the Module trait.

        Args:
            input: Input tensor (batch_size, input_dim).

        Returns:
            Output tensor (batch_size, output_dim).

        Raises:
            Error: If tensor operations fail.

        Note:
            Placeholder implementation that creates output tensor with
            correct shape. Full implementation would require Layer trait
            conformance and actual neural network operations.
        """
        # Placeholder: create output with correct shape
        var output_shape = List[Int]()
        if len(input._shape) == 1:
            output_shape.append(self.output_dim)
        else:
            var batch_size = input._shape[0]
            output_shape.append(batch_size)
            output_shape.append(self.output_dim)
        var output = zeros(output_shape, input._dtype)

        return output^

    def parameters(self) raises -> List[AnyTensor]:
        """Get all trainable parameters.

        Returns:
            List of 4 AnyTensors: W1, b1, W2, b2 (placeholder weights).

        Raises:
            Error: If parameter collection fails.
        """
        var params = List[AnyTensor]()
        # W1: (input_dim, hidden_dim)
        params.append(zeros([self.input_dim, self.hidden_dim], DType.float32))
        # b1: (hidden_dim,)
        params.append(zeros([self.hidden_dim], DType.float32))
        # W2: (hidden_dim, output_dim)
        params.append(zeros([self.hidden_dim, self.output_dim], DType.float32))
        # b2: (output_dim,)
        params.append(zeros([self.output_dim], DType.float32))
        return params^

    def train(mut self):
        """Set model to training mode (no-op placeholder)."""
        pass

    def set_inference_mode(mut self):
        """Set model to inference mode (no-op placeholder)."""
        pass

    def zero_grad(mut self) raises:
        """Placeholder for gradient reset (no gradient state in this fixture).

        Raises:
            Error: If operation fails.
        """
        pass
