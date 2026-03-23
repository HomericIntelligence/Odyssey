"""Core traits for ML components.

Provides zero-cost trait-based abstractions for neural network components,
enabling polymorphism without runtime overhead. All traits compile to
static dispatch (no vtables, no dynamic dispatch).

Benefits:
- Type-safe polymorphism at compile time
- Zero runtime overhead (no virtual function calls)
- Clear interface contracts
- Composable abstractions
- Testability (mock implementations)

Trait Categories:
1. Differentiable - Components with forward/backward passes
2. Parameterized - Components with learnable parameters
3. Serializable - Components that can be saved/loaded
4. Composable - Components that can be chained

Example:
    struct MyLayer(Differentiable, Parameterized):
        fn forward(self, input: AnyTensor) -> AnyTensor:
            # ... implementation.

        fn backward(self, grad_output: AnyTensor) -> AnyTensor:
            # ... implementation.

        fn parameters(self) -> List[AnyTensor]:
            return [self.weights, self.bias]
    ```
"""

from shared.tensor.any_tensor import AnyTensor


trait Differentiable:
    """Components that support automatic differentiation.

    Implement this trait for all neural network layers and operations
    that participate in backpropagation.

    Required Methods:
        forward: Compute output from input (forward pass)
        backward: Compute input gradient from output gradient (backward pass)

    Contract:
        - forward and backward must be mathematical inverses
        - backward must preserve batch dimension
        - backward must handle None/zero gradients gracefully

    Example:
        ```mojo
        struct ReLULayer(Differentiable):
            var last_input: AnyTensor  # Cache for backward pass

            fn forward(mut self, input: AnyTensor) -> AnyTensor:
                self.last_input = input.copy()
                return relu(input)

            fn backward(self, grad_output: AnyTensor) -> AnyTensor:
                return relu_backward(grad_output, self.last_input)
        ```
    """

    fn forward(mut self, input: AnyTensor) raises -> AnyTensor:
        """Compute forward pass.

        Args:
            input: Input tensor (batch_size, ...).

        Returns:
            Output tensor (batch_size, ...).

        Raises:
            Error: If input shape is invalid.

        Note:
            May cache values needed for backward pass.
        """
        ...

    fn backward(self, grad_output: AnyTensor) raises -> AnyTensor:
        """Compute backward pass (input gradient).

        Args:
            grad_output: Gradient w.r.t. output (∂L/∂output).

        Returns:
            Gradient w.r.t. input (∂L/∂input).

        Raises:
            Error: If backward called before forward.

        Note:
            Uses values cached during forward pass.
        """
        ...


trait Parameterized:
    """Components with learnable parameters.

    Implement this trait for layers that have weights, biases, or other
    trainable parameters that should be updated during optimization.

    Required Methods:
        parameters: Return list of all parameter tensors
        gradients: Return list of all gradient tensors
        zero_grad: Reset all gradients to zero

    Contract:
        - parameters() and gradients() must return same-length lists
        - Parameters and gradients must correspond (same order)
        - zero_grad() must clear all gradient accumulation

    Example:
        ```mojo
        struct LinearLayer(Parameterized):
            var weights: AnyTensor
            var bias: AnyTensor
            var grad_weights: AnyTensor
            var grad_bias: AnyTensor

            fn parameters(self) -> List[AnyTensor]:
                return [self.weights, self.bias]

            fn gradients(self) -> List[AnyTensor]:
                return [self.grad_weights, self.grad_bias]

            fn zero_grad(mut self):
                self.grad_weights.fill(0.0)
                self.grad_bias.fill(0.0)
        ```
    """

    fn parameters(self) raises -> List[AnyTensor]:
        """Get all learnable parameters.

        Returns:
            List of parameter tensors.

        Raises:
            Error: If operation fails.

        Note:
            Order must match gradients() return order.
            Do not include non-trainable parameters (e.g., batch norm running stats).
        """
        ...

    fn gradients(self) raises -> List[AnyTensor]:
        """Get gradients for all parameters.

        Returns:
            List of gradient tensors.

        Raises:
            Error: If operation fails.

        Note:
            Must correspond 1:1 with parameters().
            Gradients are accumulated across mini-batches.
        """
        ...

    fn zero_grad(mut self) raises:
        """Reset all gradients to zero.

        Called at the beginning of each mini-batch to clear
        accumulated gradients from previous iteration.

        Raises:
            Error: If operation fails.

        Example:
            ```mojo
            model.zero_grad()  # Clear gradients
            loss = forward_pass(model, input, target)
            backward_pass(loss)  # Accumulate gradients
            optimizer.step(model.parameters(), model.gradients())
            ```
        """
        ...


trait Serializable:
    """Components that can be saved and loaded.

    Implement this trait for models and layers that need to persist
    state to disk (checkpointing, model saving).

    Required Methods:
        save: Write state to file
        load: Read state from file

    Contract:
        - save() must write all necessary state
        - load() must restore exact state
        - Round-trip (save->load) must be identity
        - File format should be documented

    Example:
        ```mojo
        struct ConvLayer(Serializable):
            var weights: AnyTensor
            var bias: AnyTensor

            fn save(self, path: String) raises:
                # Save weights and bias to file
                write_tensor(path + "/weights.bin", self.weights)
                write_tensor(path + "/bias.bin", self.bias)

            fn load(mut self, path: String) raises:
                # Load weights and bias from file
                self.weights = read_tensor(path + "/weights.bin")
                self.bias = read_tensor(path + "/bias.bin")
        ```
    """

    fn save(self, path: String) raises:
        """Save component state to file.

        Args:
            path: File path or directory.

        Raises:
            Error: If write fails or path is invalid.

        Note:
            Should save all state needed to restore component.
            Include metadata (shapes, dtypes, version).
        """
        ...

    fn load(mut self, path: String) raises:
        """Load component state from file.

        Args:
            path: File path or directory.

        Raises:
            Error: If file doesn't exist, is corrupted, or has version mismatch.

        Note:
            Should validate loaded state (shapes, dtypes).
            Handle version migration if needed.
        """
        ...


trait Composable(Differentiable):
    """Components that can be composed into pipelines.

    Implement this trait for layers and operations that can be
    chained together (e.g., Sequential, Residual connections).

    Required Methods:
        compose: Chain this component with another

    Contract:
        - Composition must preserve differentiability
        - Output shape of self must match input shape of other
        - Associative: (A ∘ B) ∘ C = A ∘ (B ∘ C)

    Example:
        ```mojo
        struct Sequential(Composable):
            var layers: List[Composable]

            fn compose[T: Composable](self, other: T) -> ComposedOp[Self, T]:
                return ComposedOp[Self, T](self, other)

        # Usage:
        var model = Linear(784, 128).compose(ReLU()).compose(Linear(128, 10))
        ```
    """

    fn compose[T: Differentiable & Copyable & Movable](self, other: T) raises:
        """Compose this component with another.

        NOTE: The compose method signature is intentionally incomplete because Mojo's
        current type system limitations prevent returning ComposedOp[Self, T]. The trait
        `Self` type does not automatically satisfy the required trait constraints
        (Differentiable & Copyable & Movable) needed by ComposedOp.

        However, ComposedOp itself is fully implemented and can be used directly:
        ```
            var composed = ComposedOp[LayerType1, LayerType2](layer1, layer2)
            var output = composed.forward(input)
            var grad = composed.backward(grad_output)
        ```

        Args:
            other: The component to compose with this one.
                The output shape of self must match the input shape of other.

        Raises:
            Error: This method is a stub. Use ComposedOp[F, S](first, second) directly.

        See Also:
            - ComposedOp: Fully implemented struct for operation composition
            - https://docs.modular.com/mojo/manual/traits/.
        """
        raise Error("Use ComposedOp[F, S](first, second) directly for now")


struct ComposedOp[
    F: Differentiable & Copyable & Movable,
    S: Differentiable & Copyable & Movable,
](Composable, Differentiable):
    """Composition of two differentiable operations.

    Chains two differentiable operations together (first.forward → second.forward)
    and applies the chain rule during backward pass (second.backward → first.backward).

    The composed operation behaves as a single differentiable layer, with the
    intermediate tensor cached for the backward pass.

    Type Parameters:
        F: First operation (must be Differentiable, Copyable, and Movable)
        S: Second operation (must be Differentiable, Copyable, and Movable)

    Contract:
        - Output shape of F must match input shape of S
        - Forward pass: output = S(F(input))
        - Backward pass applies chain rule: grad_input = F.backward(S.backward(grad_output))

    Example:
    ```
        struct ReLU(Differentiable, Copyable, Movable):
            # ... implementation

        struct Linear(Differentiable, Copyable, Movable):
            # ... implementation

        # Create composed operation
        var relu = ReLU()
        var linear = Linear(128, 64)
        var composed = ComposedOp[ReLU, Linear](relu, linear)

        # Use in forward/backward pass
        var output = composed.forward(input)
        var grad = composed.backward(loss_grad)
    ```

    Attributes:
        first: The first operation in the composition.
        second: The second operation in the composition.
        _intermediate: Cached intermediate tensor for backward pass.

    See Also:
        Composable.compose() - Factory method for creating ComposedOp
        Differentiable - Base trait for forward/backward operations
    """

    var first: Self.F
    """The first operation in the composition."""
    var second: Self.S
    """The second operation in the composition."""
    var _intermediate: AnyTensor
    """Cached intermediate tensor (output of first operation, input to second)."""

    fn __init__(out self, first: Self.F, second: Self.S) raises:
        """Initialize composed operation with first and second components.

        Args:
            first: The first differentiable operation.
            second: The second differentiable operation.

        Raises:
            Error: If tensor initialization fails.
        """
        self.first = first.copy()
        self.second = second.copy()
        # Initialize with empty tensor - will be set during forward pass
        self._intermediate = AnyTensor(List[Int](), DType.float32)

    fn forward(mut self, input: AnyTensor) raises -> AnyTensor:
        """Compute forward pass by chaining operations.

        Applies first operation, then second operation:
            output = second.forward(first.forward(input))

        The intermediate result is cached for use in backward pass.

        Args:
            input: Input tensor to the first operation.

        Returns:
            Output tensor from the second operation.

        Raises:
            Error: If either forward pass fails.

        Note:
            This method caches the intermediate tensor (output of first operation)
            for use during backward pass. The tensor is copied to preserve its
            values across potential modifications.
        """
        # Forward through first operation
        var intermediate = self.first.forward(input)

        # Cache intermediate result for backward pass
        self._intermediate = intermediate.copy()

        # Forward through second operation
        var output = self.second.forward(intermediate)

        return output

    fn backward(self, grad_output: AnyTensor) raises -> AnyTensor:
        """Compute backward pass by applying chain rule.

        Applies operations in reverse order with chain rule:
            grad_input = first.backward(second.backward(grad_output))

        Uses the cached intermediate tensor to compute gradients.

        Args:
            grad_output: Gradient w.r.t. the output (∂L/∂output).

        Returns:
            Gradient w.r.t. the input (∂L/∂input).

        Raises:
            Error: If either backward pass fails.

        Note:
            Requires that forward() was called beforehand to cache
            the intermediate tensor. If forward() was not called,
            this will use an empty tensor and likely fail.
        """
        # Backward through second operation
        var grad_intermediate = self.second.backward(grad_output)

        # Backward through first operation
        var grad_input = self.first.backward(grad_intermediate)

        return grad_input


trait Trainable:
    """Components that support training mode.

    Implement this trait for components that behave differently during
    training vs. inference (e.g., Dropout, BatchNorm).

    Required Methods:
        train: Set to training mode
        eval: Set to evaluation mode
        is_training: Check current mode

    Example:
        ```mojo
        struct Dropout(Trainable):
            var training: Bool
            var p: Float64

            fn train(mut self):
                self.training = True

            fn eval(mut self):
                self.training = False

            fn is_training(self) -> Bool:
                return self.training

            fn forward(self, input: AnyTensor) -> AnyTensor:
                if self.training:
                    # Apply dropout
                else:
                    # No dropout during inference
        ```
    """

    fn train(mut self):
        """Set component to training mode.

        Enables training-specific behavior (dropout, batch norm updates, etc.).
        """
        ...

    fn eval(mut self):
        """Set component to evaluation mode.

        Disables training-specific behavior for inference.
        """
        ...

    fn is_training(self) -> Bool:
        """Check if component is in training mode.

        Returns:
            True if training, False if evaluating.
        """
        ...


# ============================================================================
# Training Loop Traits (see #2392, #2393, #2397 for implementation)
# ============================================================================


trait Model:
    """Neural network model interface for generic TrainingLoop.

    Defines the contract for models that can be trained using TrainingLoop.
    All neural network models should implement this trait.

    Required Methods:
        forward: Execute forward pass
        parameters: Return trainable parameters
        zero_grad: Reset parameter gradients

    Example:
        ```mojo
        struct SimpleMLP(Model):
            fn forward(mut self, input: AnyTensor) raises -> AnyTensor:
                # ... layer computations ...
                return output^

            fn parameters(self) raises -> List[AnyTensor]:
                return [self.layer1_weights, self.layer1_bias, ...]^

            fn zero_grad(mut self) raises:
                # Reset all gradient accumulators
        ```
    """

    fn forward(mut self, input: AnyTensor) raises -> AnyTensor:
        """Execute forward pass through the model.

        Args:
            input: Input tensor (batch_size, input_dim).

        Returns:
            Output tensor (batch_size, output_dim).

        Raises:
            Error: If input shape is invalid.
        """
        ...

    fn parameters(self) raises -> List[AnyTensor]:
        """Return list of all trainable parameters.

        Returns:
            List of parameter tensors.

        Raises:
            Error: If operation fails.

        Note:
            Used by optimizers to update weights.
        """
        ...

    fn zero_grad(mut self) raises:
        """Reset all parameter gradients to zero.

        Raises:
            Error: If operation fails.

        Note:
            Should be called before each backward pass.
        """
        ...


trait Loss:
    """Loss function interface for generic TrainingLoop.

    Defines the contract for loss functions that measure prediction error.

    Required Methods:
        compute: Calculate loss between predictions and targets

    Example:
        ```mojo
        struct MSELoss(Loss):
            fn compute(self, pred: AnyTensor, target: AnyTensor) raises -> AnyTensor:
                var diff = subtract(pred, target)
                return mean(multiply(diff, diff))
        ```
    """

    fn compute(self, pred: AnyTensor, target: AnyTensor) raises -> AnyTensor:
        """Compute loss between predictions and targets.

        Args:
            pred: Model predictions (batch_size, ...).
            target: Ground truth targets (batch_size, ...).

        Returns:
            Scalar loss value.

        Raises:
            Error: If shapes are incompatible.
        """
        ...


trait Optimizer:
    """Optimizer interface for generic TrainingLoop.

    Defines the contract for optimization algorithms that update parameters.

    Required Methods:
        step: Update parameters based on gradients
        zero_grad: Reset optimizer state

    Example:
        ```mojo
        struct SGD(Optimizer):
            var learning_rate: Float32

            fn step(mut self, params: List[AnyTensor]) raises:
                for param in params:
                    param -= self.learning_rate * param.grad

            fn zero_grad(mut self) raises:
                # Clear any optimizer-specific state
        ```
    """

    fn step(mut self, params: List[AnyTensor]) raises:
        """Update parameters using computed gradients.

        Args:
            params: List of parameter tensors to update.

        Raises:
            Error: If operation fails.

        Note:
            Assumes gradients are already computed.
        """
        ...

    fn zero_grad(mut self) raises:
        """Reset optimizer state.

        Raises:
            Error: If operation fails.

        Note:
            May be called before parameter zero_grad().
        """
        ...

    fn get_learning_rate(self) -> Float64:
        """Get the current learning rate.

        Returns:
            Current learning rate as Float64.

        Note:
            Used by TrainingLoop to access learning rate for autograd optimizer.
        """
        ...
