"""Type definitions for the gradient tape system.

This module contains the core type definitions used by both tape.mojo and
backward_ops.mojo to avoid circular import issues.

Design Note:
    By separating type definitions into their own module, we can break the
    circular dependency between tape.mojo and backward_ops.mojo:
    - tape_types.mojo: Defines TapeNode, SavedTensors, VariableRegistry
    - backward_ops.mojo: Imports types from tape_types, implements backward ops
    - tape.mojo: Imports types from tape_types, imports functions from backward_ops
"""

from projectodyssey.tensor.any_tensor import AnyTensor, zeros_like


struct SavedTensors(Copyable, Movable):
    """Container for tensors saved during forward pass for backward computation.

    Different operations need different tensors saved:
    - Binary ops (add, mul): Need both inputs and output.
    - Unary ops (relu, exp): Need input and output.
    - Reductions (sum, mean): Need input tensor for gradient computation.
    """

    var tensors: List[AnyTensor]
    """Saved tensors for backward computation."""
    var shapes: List[List[Int]]
    """Saved shapes for tensor reconstruction."""
    var scalars: List[Float64]
    """Saved scalar values for backward computation."""

    def __init__(out self):
        """Initialize empty saved tensors."""
        self.tensors = List[AnyTensor]()
        self.shapes = List[List[Int]]()
        self.scalars = List[Float64]()

    def add_tensor(mut self, tensor: AnyTensor) raises:
        """Save a tensor for backward pass.

        Modifies self in-place by appending tensor to internal storage.

        Raises:
            Error: If operation fails.
        """
        # Create a copy of the tensor using tensor's __setitem__
        var copy = zeros_like(tensor)
        var size = tensor.numel()
        for i in range(size):
            copy.set(i, Float64(tensor._data.bitcast[Float32]()[i]))
        self.tensors.append(copy^)

    def add_shape(mut self, shape: List[Int]):
        """Save a shape for backward pass."""
        var shape_copy = List[Int]()
        for i in range(len(shape)):
            shape_copy.append(shape[i])
        self.shapes.append(shape_copy^)

    def add_scalar(mut self, value: Float64):
        """Save a scalar for backward pass."""
        self.scalars.append(value)


struct TapeNode(Copyable, Movable):
    """Represents a single operation in the computation graph.

    Each node records:
    - The operation type (e.g., "add", "multiply", "matmul").
    - Input variable IDs that were used.
    - Output variable ID that was produced.
    - Saved tensors needed for the backward pass.

    During backward propagation, nodes are traversed in reverse topological
    order, and each node's backward function is called to compute gradients.

    Attributes:
        op_type: String identifier for the operation (e.g., "add", "matmul").
        input_ids: IDs of input Variables (for tracking dependencies).
        output_id: ID of output Variable.
        saved_tensors: Tensors saved for backward pass.
    """

    var op_type: String
    """String identifier for the operation type."""
    var input_ids: List[Int]
    """IDs of input variables for dependency tracking."""
    var output_id: Int
    """ID of the output variable produced by this operation."""
    var saved: SavedTensors
    """Tensors saved during forward pass for backward computation."""

    def __init__(
        out self, op_type: String, input_ids: List[Int], output_id: Int
    ):
        """Initialize a tape node.

        Args:
            op_type: String identifier for the operation.
            input_ids: IDs of input Variables.
            output_id: ID of output Variable.
        """
        self.op_type = op_type
        self.input_ids = input_ids.copy()
        self.output_id = output_id
        self.saved = SavedTensors()

    def __init__(
        out self,
        op_type: String,
        input_ids: List[Int],
        output_id: Int,
        var saved: SavedTensors,
    ):
        """Initialize a tape node with saved tensors.

        Args:
            op_type: String identifier for the operation.
            input_ids: IDs of input Variables.
            output_id: ID of output Variable.
            saved: Saved tensors for backward pass.
        """
        self.op_type = op_type
        self.input_ids = input_ids.copy()
        self.output_id = output_id
        self.saved = saved^


struct VariableRegistry:
    """Registry mapping variable IDs to their gradient tensors.

    This allows the backward pass to look up and accumulate gradients
    for variables by their ID.
    """

    var grads: List[AnyTensor]
    """Gradient tensors indexed by variable ID."""
    var has_grad: List[Bool]
    """Flag indicating whether gradient has been computed for each variable."""
    var requires_grad: List[Bool]
    """Flag indicating whether each variable requires gradients."""
    var next_id: Int
    """Counter for assigning unique IDs to variables."""

    def __init__(out self):
        """Initialize empty registry."""
        self.grads = []
        self.has_grad = []
        self.requires_grad = []
        self.next_id = 0

    def register(mut self, requires_grad: Bool) raises -> Int:
        """Register a new variable and return its ID.

        Args:
            requires_grad: Whether this variable requires gradients.

        Returns:
            The unique ID assigned to this variable.

        Raises:
            Error: If operation fails.
        """
        var id = self.next_id
        self.next_id += 1

        # Extend lists to accommodate new ID
        # Create a placeholder tensor (will be replaced when gradient is computed)
        var placeholder_shape = List[Int]()
        placeholder_shape.append(1)
        var placeholder = AnyTensor(placeholder_shape, DType.float32)
        self.grads.append(placeholder^)
        self.has_grad.append(False)
        self.requires_grad.append(requires_grad)

        return id

    def set_grad(mut self, id: Int, grad: AnyTensor) raises:
        """Set or accumulate gradient for a variable.

        Args:
            id: Variable ID.
            grad: Gradient tensor to set/accumulate.

        Raises:
            Error: If operation fails.
        """
        if id >= len(self.grads):
            return

        var size = grad.numel()
        if self.has_grad[id]:
            # Accumulate gradients - use tensor __setitem__
            var existing = self.grads[id]
            for i in range(size):
                var existing_val = existing._data.bitcast[Float32]()[i]
                var grad_val = grad._data.bitcast[Float32]()[i]
                existing.set(i, Float64(existing_val + grad_val))
            self.grads[id] = existing^
        else:
            # First gradient - copy it using tensor __setitem__
            var grad_copy = zeros_like(grad)
            for i in range(size):
                grad_copy.set(i, Float64(grad._data.bitcast[Float32]()[i]))
            self.grads[id] = grad_copy^
            self.has_grad[id] = True

    def get_grad(self, id: Int) raises -> AnyTensor:
        """Get gradient for a variable.

        Args:
            id: Variable ID.

        Returns:
            The gradient tensor (or placeholder if not computed).

        Raises:
            Error: If operation fails.
        """
        if id < len(self.grads):
            return self.grads[id]
        # Return empty placeholder
        var placeholder_shape = List[Int]()
        placeholder_shape.append(1)
        return AnyTensor(placeholder_shape, DType.float32)

    def has_gradient(self, id: Int) -> Bool:
        """Check if a variable has a computed gradient.

        Args:
            id: Variable ID.

        Returns:
            True if gradient has been computed.
        """
        if id < len(self.has_grad):
            return self.has_grad[id]
        return False

    def clear(mut self):
        """Clear all gradients but keep variable registrations."""
        for i in range(len(self.has_grad)):
            self.has_grad[i] = False
