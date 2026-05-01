"""Example: Mojo Patterns - Trait-Based Design

This example demonstrates creating reusable interfaces with traits.

Usage:
    pixi run mojo run examples/mojo_patterns/trait_example.mojo

See documentation: docs/core/mojo-patterns.md
"""

from std.memory import UnsafePointer, alloc

# Note: This is a conceptual example demonstrating trait patterns.
# It uses a simplified Tensor stub for illustration purposes.


struct Tensor(Copyable, ImplicitlyCopyable):
    """Simplified tensor stub for demonstration purposes.

    Note: This is a minimal stub for demonstrating traits.
    In real implementations, gradients are managed separately.
    This stub uses a pointer to break circular reference.
    """

    var size: Int
    var _grad_ptr: UnsafePointer[
        Int, origin=MutAnyOrigin
    ]  # Points to grad size

    def __init__(out self, size: Int):
        """Create tensor with given size."""
        self.size = size
        # Allocate space for gradient size
        self._grad_ptr = alloc[Int](1)
        self._grad_ptr[] = size

    def __init__(out self, *, copyexisting: Self):
        """Copy constructor."""
        self.size = existing.size
        self._grad_ptr = alloc[Int](1)
        self._grad_ptr[] = existing._grad_ptr[]

    def __del__(deinit self):
        """Destructor to free gradient pointer."""
        if self._grad_ptr:
            self._grad_ptr.free()

    def get_grad(self) -> Tensor:
        """Get gradient as a Tensor (stub - returns zero tensor of same size).
        """
        return Tensor(self._grad_ptr[])

    def set_grad(mut self, value: Tensor):
        """Set gradient (stub - just updates size)."""
        self._grad_ptr[] = value.size

    def __mul__(self, scalar: Float64) -> Tensor:
        """Multiply tensor by scalar (stub)."""
        return Tensor(self.size)

    def __rmul__(self, scalar: Float64) -> Tensor:
        """Right multiply: scalar * tensor (stub)."""
        return Tensor(self.size)

    @staticmethod
    def randn(rows: Int, cols: Int) -> Tensor:
        """Create tensor with random values (stub)."""
        return Tensor(rows * cols)

    @staticmethod
    def zeros(size: Int, dtype: DType) -> Tensor:
        """Create tensor filled with zeros (stub)."""
        return Tensor(size)

    @staticmethod
    def zeros_like(other: Tensor) -> Tensor:
        """Create tensor of zeros with same shape (stub)."""
        return Tensor(other.size)

    def __matmul__(self, other: Tensor) -> Tensor:
        """Matrix multiplication stub."""
        return Tensor(self.size)

    def __add__(self, other: Tensor) -> Tensor:
        """Addition stub."""
        return Tensor(self.size)

    def __sub__(self, other: Float64) -> Tensor:
        """Subtraction stub."""
        return Tensor(self.size)

    def __isub__(mut self, other: Tensor):
        """In-place subtraction stub."""
        pass

    def transpose(self) -> Tensor:
        """Transpose stub (accessed as .T property in example)."""
        return Tensor(self.size)

    def shape(self) -> List[Int]:
        """Return shape stub."""
        var result = List[Int]()
        result.append(self.size)
        return result^


trait Module:
    """Base trait for neural network modules."""

    def forward(mut self, input: Tensor) -> Tensor:
        """Forward pass."""
        ...

    def parameters(mut self) -> List[Tensor]:
        """Get trainable parameters."""
        ...


trait Optimizer:
    """Base trait for optimizers."""

    def step(self, mut parameters: List[Tensor]):
        """Update parameters."""
        ...

    def zero_grad(self, mut parameters: List[Tensor]):
        """Zero gradients."""
        for i in range(len(parameters)):
            parameters[i].set_grad(Tensor.zeros_like(parameters[i]))


struct Linear(Module):
    """Linear layer implementing Module trait."""

    var weight: Tensor
    var bias: Tensor

    def __init__(out self, input_size: Int, output_size: Int):
        self.weight = Tensor.randn(output_size, input_size)
        self.bias = Tensor.zeros(output_size, DType.float32)

    def forward(mut self, input: Tensor) -> Tensor:
        return input @ self.weight.transpose() + self.bias

    def parameters(mut self) -> List[Tensor]:
        return [self.weight, self.bias]


struct Adam(Optimizer):
    """Adam optimizer implementing Optimizer trait."""

    var lr: Float64
    var beta1: Float64
    var beta2: Float64

    def __init__(
        out self,
        lr: Float64 = 0.001,
        beta1: Float64 = 0.9,
        beta2: Float64 = 0.999,
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2

    def step(self, mut parameters: List[Tensor]):
        # Adam update logic
        for i in range(len(parameters)):
            # Update with momentum and adaptive learning rate
            var grad = parameters[i].get_grad()
            parameters[i] -= self.lr * grad


def main() raises:
    """Demonstrate trait-based design."""
    var layer = Linear(784, 128)
    var optimizer = Adam(lr=0.001)

    var input = Tensor.randn(32, 784)
    var output = layer.forward(input)

    print("Layer output shape:", output.shape())

    # Demonstrate optimizer trait methods
    var params = layer.parameters()
    optimizer.zero_grad(params)
    optimizer.step(params)

    print("Trait-based design example complete!")
