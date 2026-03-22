"""Module trait for neural network layers and models.

This module provides a standard interface for composing neural network layers
and models. The Module trait enables:

- Consistent interface for all layers (Linear, Conv2D, etc.).
- Easy parameter collection for optimizers.
- Nested module support (models containing layers).
- Training vs eval mode switching.

The trait follows a simple interface pattern where implementations can vary
but must support forward passes, parameter access, and mode switching.

Key Design Principles:
- Minimal interface: forward(), parameters(), train(), eval().
- No required default implementations - subclasses implement what they need.
- Training flag for layers that need it (dropout, batch norm).
- Parameter collection for optimizer integration.

Example:
    ```mojo
    from .module import Module
    from .layers import Linear
    from .extensor import ExTensor, zeros

    # Linear layer implements Module trait
    var layer = Linear(10, 5)
    var input = zeros([4, 10], DType.float32)

    # Call forward pass
    var output = layer.forward(input)

    # Get trainable parameters
    var params = layer.parameters()

    # Switch to eval mode if supported
    layer.eval()
    ```

Sequential Usage Example:
    ```mojo
    from .module import Module
    from .layers import Linear, ReLU
    from .extensor import ExTensor, zeros

    # Compose layers using Sequential2
    from .sequential import Sequential2
    from .layers import ReLULayer
    var model = Sequential2[Linear, ReLULayer](
        Linear(784, 256),
        ReLULayer(),
    )

    var input = zeros([32, 784], DType.float32)
    var output = model.forward(input)  # shape: [32, 256] after ReLU

    # Get parameters and switch modes
    var params = model.parameters()
    model.train()
    ```

See Also:
    - shared.core.layers.linear: Linear layer implementation example
"""

from .extensor import AnyTensor


trait Module:
    """Standard interface for neural network modules and layers.

    A Module is any component in a neural network that can be composed
    with other modules. This includes:
    - Individual layers (Linear, Conv2D, etc.)
    - Activation functions with state
    - Normalization layers (BatchNorm, LayerNorm, etc.)
    - Complete models (sequences of layers)

    All modules must implement the forward pass and parameter collection
    Modules that need training/eval mode switching implement train() and eval()

    Type Parameters:
        - No generic parameters - all modules work with AnyTensor
    """

    fn forward(mut self, input: AnyTensor) raises -> AnyTensor:
        """Compute forward pass of the module.

        Args:
            input: Input tensor to the module. Shape depends on module type.

        Returns:
            Output tensor from the module. Shape depends on module type.

        Raises:
            Error: If tensor operations fail or shapes are incompatible.

        Note:
            This method is required for all Module implementations.
            It defines the core computation of the module.
        """
        ...

    fn parameters(self) raises -> List[AnyTensor]:
        """Get list of trainable parameters.

        Returns a list of all learnable parameters (weights, biases, etc.)
        that this module manages. The list may be empty for modules with
        no trainable parameters (e.g., activation functions, pooling).

        Returns:
            List of AnyTensor containing all trainable parameters.
            Order should be consistent across calls.

        Raises:
            Error: If parameter collection fails.

        Note:
            - Returned parameters are typically references/copies.
            - Order should be deterministic for reproducibility.
            - Nested modules should recursively include sub-module parameters.
            - For optimization: frameworks typically flatten this for gradient updates.
        """
        ...

    fn train(mut self):
        """Switch module to training mode.

        Sets the module to training mode, enabling features like:
        - Dropout regularization.
        - Batch normalization with running statistics.
        - Other training-specific behaviors.

        Note:
            - Default (no-op) if module doesn't need mode switching.
            - Can be overridden by layers that need it.
            - Should be called before training loop.

        Example:
            ```mojo
            var model = MyModel()
            model.train()  # Enable dropout, etc.
            # ... training loop ...
            ```
        """
        ...

    fn eval(mut self):
        """Switch module to evaluation mode.

        Sets the module to evaluation (inference) mode, disabling features like:
        - Dropout regularization.
        - Batch normalization updates.
        - Other training-specific behaviors.

        Note:
            - Default (no-op) if module doesn't need mode switching.
            - Can be overridden by layers that need it.
            - Should be called before inference.

        Example:
            ```mojo
            var model = MyModel()
            model.eval()  # Disable dropout, etc.
            var output = model.forward(input)
            ```
        """
        ...
