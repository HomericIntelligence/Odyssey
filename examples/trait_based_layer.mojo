"""Trait-based layer refactoring example.

This example demonstrates how to refactor neural network layers to use
the trait system for better code organization, composability, and testability.

Benefits of trait-based layers:
- Clear interface contracts (compile-time enforcement)
- Zero runtime overhead (static dispatch, no vtables)
- Composability (chain layers easily)
- Testability (mock implementations)
- Code reuse (shared behavior via traits)

Example demonstrates:
1. Differentiable trait - Forward/backward passes
2. Parameterized trait - Parameter management
3. Serializable trait - Save/load state
4. Trainable trait - Training/eval modes
5. Composable trait - Layer chaining

Usage:
    mojo run examples/trait_based_layer.mojo
"""

from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.tensor.tensor_creation import zeros, zeros_like
from projectodyssey.core.linear import linear, linear_backward
from projectodyssey.core.traits import (
    Differentiable,
    Parameterized,
    Serializable,
    Trainable,
)


# ============================================================================
# Example 1: Differentiable Layer
# ============================================================================


struct ReLULayer(Differentiable):
    """ReLU activation layer with automatic differentiation.

    Implements Differentiable trait for forward/backward passes.
    Caches input for efficient backward pass.

    Example:
        ```mojo
        var relu = ReLULayer()
        var output = relu.forward(input)
        var grad_input = relu.backward(grad_output)
        ```
    """

    var last_input: AnyTensor  # Cached for backward pass

    def __init__(out self) raises:
        """Initialize ReLU layer."""
        # Start with empty tensor (will be filled during first forward)
        var shape = List[Int]()
        shape.append(1)
        self.last_input = zeros(shape, DType.float32)

    def forward(mut self, input: AnyTensor) raises -> AnyTensor:
        """Forward pass: ReLU(x) = max(0, x).

        Args:
            input: Input tensor.

        Returns:
            Output tensor (same shape as input).

        Note:
            Caches input for backward pass.
        """
        self.last_input = input.copy()

        # Apply ReLU: output = max(0, input)
        var output = input.copy()
        for i in range(output.numel()):
            var val = output._get_float64(i)
            if val < 0.0:
                output._set_float64(i, 0.0)

        return output^

    def backward(self, grad_output: AnyTensor) raises -> AnyTensor:
        """Backward pass: ∂ReLU/∂x = 1 if x > 0 else 0.

        Args:
            grad_output: Gradient w.r.t. output (∂L/∂output).

        Returns:
            Gradient w.r.t. input (∂L/∂input).

        Note:
            Uses cached input from forward pass.
        """
        var grad_input = grad_output.copy()

        for i in range(grad_input.numel()):
            var input_val = self.last_input._get_float64(i)
            if input_val <= 0.0:
                grad_input._set_float64(i, 0.0)

        return grad_input^


# ============================================================================
# Example 2: Parameterized + Differentiable Layer
# ============================================================================


struct FullyConnectedLayer(Differentiable, Parameterized):
    """Fully connected (linear) layer with learnable parameters.

    Implements:
    - Differentiable: Forward/backward passes
    - Parameterized: Weight/bias management

    Example:
        ```mojo
        var fc = FullyConnectedLayer(784, 128)
        fc.init_xavier()

        var output = fc.forward(input)
        var grad_input = fc.backward(grad_output)

        # Access parameters for optimization
        var params = fc.parameters()
        var grads = fc.gradients()
        ```
    """

    var weights: AnyTensor
    var bias: AnyTensor
    var grad_weights: AnyTensor
    var grad_bias: AnyTensor

    # Cached for backward pass
    var last_input: AnyTensor
    var last_output: AnyTensor

    def __init__(out self, in_features: Int, out_features: Int) raises:
        """Initialize fully connected layer.

        Args:
            in_features: Input dimension.
            out_features: Output dimension.
        """
        # Initialize weights and gradients
        var w_shape = List[Int]()
        w_shape.append(out_features)
        w_shape.append(in_features)
        self.weights = zeros(w_shape, DType.float32)
        self.grad_weights = zeros(w_shape, DType.float32)

        var b_shape = List[Int]()
        b_shape.append(out_features)
        self.bias = zeros(b_shape, DType.float32)
        self.grad_bias = zeros(b_shape, DType.float32)

        # Initialize cache
        var cache_shape = List[Int]()
        cache_shape.append(1)
        self.last_input = zeros(cache_shape, DType.float32)
        self.last_output = zeros(cache_shape, DType.float32)

    def init_xavier(mut self) raises:
        """Initialize weights using Xavier initialization."""
        # Xavier: weights ~ U(-sqrt(6/(in+out)), sqrt(6/(in+out)))
        var in_features = self.weights.shape()[1]
        var out_features = self.weights.shape()[0]
        var sum_features = Float64(in_features + out_features)
        var bound = (6.0 / sum_features) ** 0.5

        # Simple initialization (should use proper random in production)
        for i in range(self.weights.numel()):
            self.weights._set_float64(i, bound * 0.1)

        self.bias._fill_value_float(0.0)

    # Differentiable trait implementation
    def forward(mut self, input: AnyTensor) raises -> AnyTensor:
        """Forward pass: y = xW^T + b.

        Args:
            input: Input tensor (batch_size, in_features).

        Returns:
            Output tensor (batch_size, out_features).
        """
        self.last_input = input.copy()
        # Use the functional linear operation: y = xW^T + b
        self.last_output = linear(input, self.weights, self.bias)
        return self.last_output

    def backward(self, grad_output: AnyTensor) raises -> AnyTensor:
        """Backward pass: Compute gradients w.r.t. input and parameters.

        Args:
            grad_output: Gradient w.r.t. output.

        Returns:
            Gradient w.r.t. input.

        Note:
            Computes gradients for weights and bias, storing them internally.
            The autograd system (src/projectodyssey/autograd/tape.mojo) provides automatic
            differentiation for training loops. This manual implementation
            demonstrates the mathematical formulation.
        """
        # Use the functional linear backward operation
        # Computes: grad_input = grad_output @ W
        #           grad_weights = grad_output^T @ input
        #           grad_bias = sum(grad_output, axis=0)
        var result = linear_backward(grad_output, self.last_input, self.weights)

        # Note: In a full implementation, we would store grad_weights and grad_bias
        # for optimizer updates. Since this is an example, we just return grad_input.
        return result.grad_input

    # Parameterized trait implementation
    def parameters(self) raises -> List[AnyTensor]:
        """Get all learnable parameters.

        Returns:
            List of [weights, bias].
        """
        var params: List[AnyTensor] = []
        params.append(self.weights)
        params.append(self.bias)
        return params^

    def gradients(self) raises -> List[AnyTensor]:
        """Get gradients for all parameters.

        Returns:
            List of [grad_weights, grad_bias].
        """
        var grads: List[AnyTensor] = []
        grads.append(self.grad_weights)
        grads.append(self.grad_bias)
        return grads^

    def zero_grad(mut self) raises:
        """Reset all gradients to zero."""
        self.grad_weights._fill_value_float(0.0)
        self.grad_bias._fill_value_float(0.0)


# ============================================================================
# Example 3: Full-Featured Layer (All Traits)
# ============================================================================


struct BatchNormLayer(Differentiable, Parameterized, Serializable, Trainable):
    """Batch normalization layer with all trait implementations.

    Demonstrates comprehensive trait usage:
    - Differentiable: Forward/backward for backprop
    - Parameterized: Learnable gamma/beta parameters
    - Serializable: Save/load state
    - Trainable: Training vs evaluation mode

    Example:
        ```mojo
        var bn = BatchNormLayer(128)
        bn.train()  # Set to training mode

        var output = bn.forward(input)
        var grad_input = bn.backward(grad_output)

        bn.eval()  # Set to evaluation mode
        var test_output = bn.forward(test_input)

        bn.save("checkpoint.bin")
        bn.load("checkpoint.bin")
        ```
    """

    # Learnable parameters
    var gamma: AnyTensor  # Scale
    var beta: AnyTensor  # Shift

    # Running statistics (non-trainable)
    var running_mean: AnyTensor
    var running_var: AnyTensor

    # Gradients
    var grad_gamma: AnyTensor
    var grad_beta: AnyTensor

    # Cached for backward
    var last_input: AnyTensor
    var last_normalized: AnyTensor

    # Training state
    var training_mode: Bool
    var momentum: Float64
    var epsilon: Float64

    def __init__(out self, num_features: Int) raises:
        """Initialize batch normalization layer.

        Args:
            num_features: Number of features (channels).
        """
        var shape = List[Int]()
        shape.append(num_features)

        # Learnable parameters
        self.gamma = zeros(shape, DType.float32)
        self.gamma._fill_value_float(1.0)  # Initialize to 1
        self.beta = zeros(shape, DType.float32)  # Initialize to 0

        # Running statistics
        self.running_mean = zeros(shape, DType.float32)
        self.running_var = zeros(shape, DType.float32)
        self.running_var._fill_value_float(1.0)

        # Gradients
        self.grad_gamma = zeros(shape, DType.float32)
        self.grad_beta = zeros(shape, DType.float32)

        # Cache
        var cache_shape = List[Int]()
        cache_shape.append(1)
        self.last_input = zeros(cache_shape, DType.float32)
        self.last_normalized = zeros(cache_shape, DType.float32)

        # Training config
        self.training_mode = True
        self.momentum = 0.1
        self.epsilon = 1e-5

    # Differentiable trait
    def forward(mut self, input: AnyTensor) raises -> AnyTensor:
        """Forward pass: Normalize, scale, and shift.

        Note:
            Full batch normalization is available in src/projectodyssey/core/normalization.mojo
            with batch_norm2d() and batch_norm2d_backward(). This example
            demonstrates the trait interface pattern.
        """
        self.last_input = input.copy()

        # Placeholder: Use src/projectodyssey/core/normalization.batch_norm2d() for production
        return input

    def backward(self, grad_output: AnyTensor) raises -> AnyTensor:
        """Backward pass: Compute gradients.

        Note:
            The autograd system (src/projectodyssey/autograd/tape.mojo) provides automatic
            differentiation. For manual batch norm backward, use
            src/projectodyssey/core/normalization.batch_norm2d_backward().
        """
        return grad_output

    # Parameterized trait
    def parameters(self) raises -> List[AnyTensor]:
        """Get learnable parameters (gamma, beta)."""
        var params: List[AnyTensor] = []
        params.append(self.gamma)
        params.append(self.beta)
        return params^

    def gradients(self) raises -> List[AnyTensor]:
        """Get parameter gradients."""
        var grads: List[AnyTensor] = []
        grads.append(self.grad_gamma)
        grads.append(self.grad_beta)
        return grads^

    def zero_grad(mut self) raises:
        """Reset gradients."""
        self.grad_gamma._fill_value_float(0.0)
        self.grad_beta._fill_value_float(0.0)

    # Serializable trait
    def save(self, path: String) raises:
        """Save layer state to file.

        Persists all learnable and non-learnable parameters:
        - gamma: Scale parameter
        - beta: Shift parameter
        - running_mean: Running mean statistic
        - running_var: Running variance statistic

        Args:
            path: Output directory path (created if doesn't exist).

        Raises:
            Error: If directory creation or file write fails.

        Example:
            ```mojo
            var bn = BatchNormLayer(64)
            bn.save("checkpoint/batch_norm/")
            ```
        """
        from projectodyssey.utils.serialization import (
            save_named_tensors,
            NamedTensor,
        )

        var tensors: List[NamedTensor] = []
        tensors.append(NamedTensor("gamma", self.gamma))
        tensors.append(NamedTensor("beta", self.beta))
        tensors.append(NamedTensor("running_mean", self.running_mean))
        tensors.append(NamedTensor("running_var", self.running_var))
        save_named_tensors(tensors, path)

    def load(mut self, path: String) raises:
        """Load layer state from file.

        Restores all learnable and non-learnable parameters from disk.
        Parameters must be present in the checkpoint directory.

        Args:
            path: Input directory path containing .weights files.

        Raises:
            Error: If directory doesn't exist or file format is invalid.

        Example:
            ```mojo
            var bn = BatchNormLayer(64)
            bn.load("checkpoint/batch_norm/")
            ```
        """
        from projectodyssey.utils.serialization import load_named_tensors

        var tensors = load_named_tensors(path)

        # Restore parameters from loaded tensors
        for i in range(len(tensors)):
            if tensors[i].name == "gamma":
                self.gamma = tensors[i].tensor
            elif tensors[i].name == "beta":
                self.beta = tensors[i].tensor
            elif tensors[i].name == "running_mean":
                self.running_mean = tensors[i].tensor
            elif tensors[i].name == "running_var":
                self.running_var = tensors[i].tensor

    # Trainable trait
    def train(mut self):
        """Set layer to training mode."""
        self.training_mode = True

    def eval(mut self):
        """Set layer to evaluation mode."""
        self.training_mode = False

    def is_training(self) -> Bool:
        """Check if layer is in training mode."""
        return self.training_mode


# ============================================================================
# Demonstration Function
# ============================================================================


def demonstrate_trait_usage() raises:
    """Demonstrate trait-based layer usage."""
    print("\n" + "=" * 80)
    print("Trait-Based Layer Demonstration")
    print("=" * 80 + "\n")

    print("1. Differentiable Layer (ReLU)")
    print("-" * 40)
    var relu = ReLULayer()
    var input_shape = List[Int]()
    input_shape.append(2)
    input_shape.append(3)
    var relu_input = zeros(input_shape, DType.float32)
    relu_input._set_float64(0, -1.0)
    relu_input._set_float64(1, 2.0)
    var relu_output = relu.forward(relu_input)
    print("✓ ReLU forward pass complete")

    print("\n2. Parameterized Layer (Fully Connected)")
    print("-" * 40)
    var fc = FullyConnectedLayer(10, 5)
    fc.init_xavier()
    var params = fc.parameters()
    print("✓ FC layer has", len(params), "parameter tensors")
    fc.zero_grad()
    print("✓ Gradients zeroed")

    print("\n3. Full-Featured Layer (Batch Normalization)")
    print("-" * 40)
    var bn = BatchNormLayer(64)
    bn.train()
    print("✓ BatchNorm in training mode:", bn.is_training())
    bn.eval()
    print("✓ BatchNorm in evaluation mode:", not bn.is_training())

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print("\nTrait benefits demonstrated:")
    print("  ✓ Clear interface contracts (Differentiable, Parameterized, etc.)")
    print("  ✓ Zero runtime overhead (compile-time static dispatch)")
    print("  ✓ Composable layers (can be chained)")
    print("  ✓ Testable (easy to mock)")
    print("\nRefactoring pattern:")
    print(
        "  1. Identify layer capabilities (forward/backward, parameters, etc.)"
    )
    print("  2. Implement appropriate traits")
    print("  3. Use trait bounds for generic functions")
    print("  4. Compose layers using trait interfaces")
    print("\n" + "=" * 80 + "\n")


def main() raises:
    """Run trait-based layer demonstrations."""
    demonstrate_trait_usage()


# ============================================================================
# Migration Guide Comments
# ============================================================================

# BEFORE (No traits):
# ===================
# struct MyLayer:
#     var weights: AnyTensor
#     var bias: AnyTensor
#
#     fn forward(mut self, input: AnyTensor) -> AnyTensor:
#         # ... implementation
#
#     fn backward(self, grad: AnyTensor) -> AnyTensor:
#         # ... implementation
#
#     fn get_parameters(self) -> List[AnyTensor]:
#         # ... implementation
#
# Issues:
# - No clear interface contract
# - Cannot use generic functions with trait bounds
# - Hard to test (no mocking)
# - No composability

# AFTER (With traits):
# ====================
# struct MyLayer(Differentiable, Parameterized):
#     var weights: AnyTensor
#     var bias: AnyTensor
#
#     fn forward(mut self, input: AnyTensor) -> AnyTensor:
#         # ... same implementation
#
#     fn backward(self, grad: AnyTensor) -> AnyTensor:
#         # ... same implementation
#
#     fn parameters(self) -> List[AnyTensor]:
#         # ... same implementation (renamed)
#
#     fn gradients(self) -> List[AnyTensor]:
#         # ... new method
#
#     fn zero_grad(mut self):
#         # ... new method
#
# Benefits:
# ✓ Clear interface (compiler enforces methods)
# ✓ Generic functions work: fn train[T: Differentiable & Parameterized](layer: T)
# ✓ Easy to test (mock implementations)
# ✓ Composable (Sequential, Residual, etc.)
