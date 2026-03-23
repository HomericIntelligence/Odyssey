"""Gradient Tape - Operation recording for automatic differentiation.

Implements a tape-based autograd system that records operations during forward
pass and replays them in reverse during backward propagation.

The gradient tape maintains a directed acyclic graph (DAG) of operations where:
- Nodes represent tensors (Variables)
- Edges represent operations with their backward functions
- Topological ordering enables correct gradient flow via chain rule

This design follows PyTorch's eager execution model where operations are
recorded as they happen, rather than building a static computation graph.

Key Components:
- TapeNode: Represents a single operation in the computation graph
- GradientTape: Global tape that records all operations
- backward(): Traverses tape in reverse, applying chain rule

Architecture:
    Forward Pass:
        x = Variable(...)  # No recording
        y = x + 2          # Records: TapeNode(op="add", inputs=[x], output=y)
        z = y * 3          # Records: TapeNode(op="mul", inputs=[y], output=z)

    Backward Pass:
        z.backward()       # Traverse tape in reverse:
                          # 1. grad_z = ones_like(z)  # dz/dz = 1
                          # 2. grad_y = mul_backward(grad_z, y, 3)  # Chain rule
                          # 3. grad_x = add_backward(grad_y, x, 2)  # Chain rule
                          # 4. x.grad = grad_x

Examples:
    # Enable gradient tape recording
    var tape = GradientTape()
    tape.enable()

    # Operations are recorded automatically
    var x = Variable(data, requires_grad=True)
    var y = x * 2
    var z = y + 3
    var loss = z.sum()

    # Compute gradients
    tape.backward(loss)

    # Access gradients
    print(x.grad)  # Chain rule: dLoss/dx
"""

from shared.tensor.any_tensor import AnyTensor, ones_like, zeros_like
from shared.core import (
    add_backward,
    subtract_backward,
    multiply_backward,
    divide_backward,
    sum_backward,
    mean_backward,
    matmul_backward,
    relu_backward,
    sigmoid_backward,
    tanh_backward,
    softmax_backward,
)
from shared.autograd.tape_types import (
    SavedTensors,
    TapeNode,
    VariableRegistry,
)
from shared.autograd.backward_ops import (
    backward_add,
    backward_subtract,
    backward_multiply,
    backward_divide,
    backward_sum,
    backward_mean,
    backward_matmul,
    backward_relu,
    backward_sigmoid,
    backward_tanh,
)


# Operation types supported by the gradient tape
comptime OP_ADD = "add"
comptime OP_SUBTRACT = "subtract"
comptime OP_MULTIPLY = "multiply"
comptime OP_DIVIDE = "divide"
comptime OP_MATMUL = "matmul"
comptime OP_POWER = "power"
comptime OP_SUM = "sum"
comptime OP_MEAN = "mean"
comptime OP_RELU = "relu"
comptime OP_SIGMOID = "sigmoid"
comptime OP_TANH = "tanh"
comptime OP_SOFTMAX = "softmax"
comptime OP_NEG = "neg"
comptime OP_EXP = "exp"
comptime OP_LOG = "log"
comptime OP_SQRT = "sqrt"


struct GradientTape:
    """Global gradient tape for recording operations.

        The tape maintains a chronological list of all operations performed on
        Variables with requires_grad=True. During backward(), the tape is traversed
        in reverse to compute gradients via the chain rule.

        Attributes:
            nodes: Chronological list of recorded operations.
            enabled: Whether the tape is currently recording.
            registry: Variable registry for gradient storage.

        Design Note:
            This uses a global tape approach (like TensorFlow's GradientTape) rather
            than per-tensor graph tracking (like PyTorch). The global tape is simpler
            to implement but requires explicit enable/disable calls.

    Examples:
            var tape = GradientTape()
            tape.enable()

            var x = Variable(data, requires_grad=True)
            var y = x * 2  # Recorded
            var z = y + 3  # Recorded

            tape.backward(z)  # Compute all gradients
            tape.disable()
    """

    var nodes: List[TapeNode]
    """Chronological list of recorded operations."""
    var enabled: Bool
    """Flag indicating whether the tape is currently recording."""
    var registry: VariableRegistry
    """Registry mapping variable IDs to gradient tensors."""

    fn __init__(out self):
        """Initialize an empty gradient tape."""
        self.nodes: List[TapeNode] = []
        self.enabled = False
        self.registry = VariableRegistry()

    fn enable(mut self):
        """Enable recording of operations.

        After calling enable(), all operations on Variables with requires_grad=True
        will be recorded in the tape.

        Examples:
            var tape = GradientTape()
            tape.enable()
            var y = x + 1  # Recorded
        """
        self.enabled = True

    fn disable(mut self):
        """Disable recording of operations.

        After calling disable(), operations will not be recorded. This is useful
        for inference or when you want to break the computation graph.

        Examples:
            tape.disable()
            var y = x + 1  # Not recorded (no gradient tracking)
        """
        self.enabled = False

    fn clear(mut self):
        """Clear all recorded operations.

        Resets the tape to an empty state. Should be called after backward()
        to free memory and prepare for the next forward pass.

        Examples:
            tape.backward(loss)
            tape.clear()  # Free memory.
        """
        self.nodes: List[TapeNode] = []
        self.registry.clear()

    fn register_variable(mut self, requires_grad: Bool) raises -> Int:
        """Register a new variable in the tape's registry.

        Args:
            requires_grad: Whether this variable requires gradients.

        Returns:
            Unique ID for the variable.

        Raises:
            Error: If operation fails.
        """
        return self.registry.register(requires_grad)

    fn record(
        mut self,
        op_type: String,
        input_ids: List[Int],
        output_id: Int,
        var saved: SavedTensors,
    ):
        """Record an operation in the tape.

        This method is called internally by Variable operations to register
        themselves in the computation graph.

        Args:
            op_type: String identifier for the operation.
            input_ids: IDs of input Variables.
            output_id: ID of output Variable.
            saved: Saved tensors for backward pass.

        Note:
            Only records if tape is enabled.

        Examples:
            # Internal use by Variable operations
            if tape.enabled:
                tape.record("add", input_ids, output_id, saved)
        """
        if not self.enabled:
            return

        var node = TapeNode(op_type, input_ids, output_id, saved^)
        self.nodes.append(node^)

    fn _dispatch_backward_op(
        mut self, op_type: String, node_idx: Int, grad_output: AnyTensor
    ) raises:
        """Dispatch backward pass computation for the given operation type.

        Args:
            op_type: Operation type constant (e.g., OP_ADD, OP_MATMUL).
            node_idx: Index of the node in the tape.
            grad_output: Gradient flowing back from downstream operations.

        Raises:
            Error if operation type is not supported.
        """
        # Binary arithmetic operations
        if op_type == OP_ADD:
            backward_add(self.nodes, self.registry, node_idx, grad_output)
        elif op_type == OP_SUBTRACT:
            backward_subtract(self.nodes, self.registry, node_idx, grad_output)
        elif op_type == OP_MULTIPLY:
            backward_multiply(self.nodes, self.registry, node_idx, grad_output)
        elif op_type == OP_DIVIDE:
            backward_divide(self.nodes, self.registry, node_idx, grad_output)
        # Reduction operations
        elif op_type == OP_SUM:
            backward_sum(self.nodes, self.registry, node_idx, grad_output)
        elif op_type == OP_MEAN:
            backward_mean(self.nodes, self.registry, node_idx, grad_output)
        # Matrix operations
        elif op_type == OP_MATMUL:
            backward_matmul(self.nodes, self.registry, node_idx, grad_output)
        # Activation functions
        elif op_type == OP_RELU:
            backward_relu(self.nodes, self.registry, node_idx, grad_output)
        elif op_type == OP_SIGMOID:
            backward_sigmoid(self.nodes, self.registry, node_idx, grad_output)
        elif op_type == OP_TANH:
            backward_tanh(self.nodes, self.registry, node_idx, grad_output)
        else:
            raise Error(
                "Unsupported operation type for backward pass: " + op_type
            )

    fn backward(mut self, output_id: Int, output_grad: AnyTensor) raises:
        """Compute gradients by traversing tape in reverse.

        Applies the chain rule in reverse topological order:
        1. Initialize gradient of final output with provided grad.
        2. For each node in reverse:
           a. Get upstream gradient (dL/d_output).
           b. Call backward function to get local gradients (d_output/d_inputs).
           c. Apply chain rule: dL/d_input = dL/d_output * d_output/d_input.
           d. Accumulate gradients in input Variables.

        Args:
            output_id: ID of the output variable to backprop from.
            output_grad: Gradient of the loss with respect to output.

        Raises:
            Error: If operation fails.

        Examples:
        ```
            var loss = compute_loss(x)
            var ones = ones_like(loss.data)
            tape.backward(loss.id, ones)  # Populates x.grad
        ```
        """
        # Set the gradient of the output
        self.registry.set_grad(output_id, output_grad)

        # Traverse nodes in reverse order to apply chain rule
        var num_nodes = len(self.nodes)
        for idx in range(num_nodes):
            # Process in reverse: node at (num_nodes - 1 - idx)
            var node_idx = num_nodes - 1 - idx

            # Get the output gradient for this node
            var node_output_id = self.nodes[node_idx].output_id
            if not self.registry.has_gradient(node_output_id):
                continue

            var grad_output = self.registry.get_grad(node_output_id)
            var op_type = self.nodes[node_idx].op_type

            # Dispatch to appropriate backward function
            self._dispatch_backward_op(op_type, node_idx, grad_output)

    fn get_grad(self, var_id: Int) raises -> AnyTensor:
        """Get the computed gradient for a variable.

        Args:
            var_id: The variable ID.

        Returns:
            The gradient tensor.

        Raises:
            Error: Gradient not found for variable.
        """
        return self.registry.get_grad(var_id)


struct NoGradContext(Copyable, Movable):
    """Context manager for disabling gradient computation.

    Provides control over gradient tracking through explicit enter/exit methods.
    This enables disabling gradient tracking for inference or when building
    computation graphs that shouldn't be differentiated.

    The context stores the previous enabled state internally and restores it
    on exit, supporting nested no-grad contexts correctly.

    Example:
    ```
        var ctx = NoGradContext()
        ctx.enter(tape)
        var output = model(input)  # No gradients tracked
        ctx.exit(tape)  # Gradients re-enabled
    ```

    Design Note:
        Uses explicit enter(tape)/exit(tape) methods instead of __enter__/__exit__
        because Mojo's context manager protocol doesn't support external mutable state.
        This pattern matches the explicit-tape-passing pattern used throughout the
        autograd module, and aligns with how Variables are passed explicitly to
        operations.

    See Also:
        disable_gradient_tracking() - Convenience function wrapping enter()
        restore_gradient_tracking() - Convenience function wrapping exit()
    """

    var _previous_enabled: Bool
    """Store the previous enabled state to restore on exit."""

    fn __init__(out self):
        """Initialize no-grad context with default state."""
        self._previous_enabled = False

    fn enter(mut self, mut tape: GradientTape):
        """Enter no-grad context by disabling gradient tracking.

        Args:
            tape: The gradient tape to disable.

        Note:
            Stores the current enabled state so exit() can restore it.
        """
        self._previous_enabled = tape.enabled
        tape.disable()

    fn exit(mut self, mut tape: GradientTape):
        """Exit no-grad context by restoring previous tracking state.

        Args:
            tape: The gradient tape to restore.

        Note:
            Restores the state from before enter() was called.
        """
        if self._previous_enabled:
            tape.enable()
        else:
            tape.disable()


fn disable_gradient_tracking(mut tape: GradientTape) -> Bool:
    """Disable gradient tracking and return the previous state.

    This is a convenience function that wraps NoGradContext.enter().
    It's useful when you want to disable gradient tracking without
    managing a context object explicitly.

    Args:
        tape: The gradient tape to disable.

    Returns:
        Bool: The previous enabled state (True if was enabled, False otherwise).
              Use this value with restore_gradient_tracking() to restore state.

    Example:
    ```
        var was_enabled = disable_gradient_tracking(tape)
        var output = model(input)  # No gradients tracked
        restore_gradient_tracking(tape, was_enabled)  # Restore previous state
    ```

    See Also:
        restore_gradient_tracking() - Restore the previous state.
        NoGradContext - For context manager style usage.
    """
    var previous_enabled = tape.enabled
    tape.disable()
    return previous_enabled


fn restore_gradient_tracking(mut tape: GradientTape, was_enabled: Bool):
    """Restore gradient tracking to a previous state.

    This is a convenience function that pairs with disable_gradient_tracking().
    It restores the gradient tape to the state it was in before disable_gradient_tracking()
    was called.

    Args:
        tape: The gradient tape to restore.
        was_enabled: The previous enabled state from disable_gradient_tracking().

    Example:
    ```
        var was_enabled = disable_gradient_tracking(tape)
        var output = model(input)
        restore_gradient_tracking(tape, was_enabled)
    ```

    See Also:
        disable_gradient_tracking() - Get the previous state
        NoGradContext - For context manager style usage
    """
    if was_enabled:
        tape.enable()
    else:
        tape.disable()
