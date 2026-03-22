"""Example: Mojo Patterns - Ownership and Borrowing

This example demonstrates Mojo's ownership system for memory safety.

Usage:
    pixi run mojo run examples/mojo-patterns/ownership_example.mojo

See documentation: docs/core/mojo-patterns.md
"""

from shared.core import AnyTensor, ones, sum, mean, multiply, item, full


# Borrowed: read-only access (no ownership transfer)
fn compute_loss(predictions: AnyTensor, targets: AnyTensor) raises -> Float64:
    """Compute loss without taking ownership."""
    var diff = predictions - targets
    var squared = multiply(diff, diff)
    var loss_tensor = mean(squared)
    return item(loss_tensor)


# Owned: take ownership (move semantics)
fn consume_tensor(var tensor: AnyTensor) raises -> Float64:
    """Take ownership and consume tensor."""
    var sum_tensor = sum(tensor)
    var result = item(sum_tensor)
    # tensor is destroyed here
    return result


# Inout: mutable reference (modify in place)
fn update_weights(
    mut weights: AnyTensor, gradients: AnyTensor, lr: Float64
) raises:
    """Update weights in place."""
    var lr_tensor = full(gradients.shape(), lr, gradients.dtype())
    var update = multiply(lr_tensor, gradients)
    weights -= update  # Modifies original


fn main() raises:
    """Demonstrate ownership patterns."""

    # Example 1: Borrowed parameters (read-only)
    var pred = ones([10, 10], DType.float64)
    var target = ones([10, 10], DType.float64)
    var loss = compute_loss(pred, target)  # No ownership transfer
    print("Loss (borrowed):", loss)

    # Example 2: Owned parameter (transfer ownership)
    var temp_tensor = ones([5, 5], DType.float64)
    var sum_value = consume_tensor(temp_tensor)  # temp_tensor is consumed
    print("Sum (owned):", sum_value)
    # Cannot use temp_tensor here - it was consumed!

    # Example 3: Inout parameter (mutable reference)
    var weights = ones([10, 10], DType.float64)
    var grads = ones([10, 10], DType.float64)
    print("Weights before update:", weights[0])
    update_weights(weights, grads, 0.01)  # Modifies weights in place
    print("Weights after update:", weights[0])

    print("\nOwnership example complete!")
