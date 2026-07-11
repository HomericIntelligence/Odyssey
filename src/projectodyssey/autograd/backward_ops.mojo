"""Backward operation implementations for automatic differentiation.

This module provides backward pass implementations for all operation types
recorded in the GradientTape. Each function computes gradients using the
chain rule and stores them in the tape's registry.

Operation Categories:
- Binary arithmetic: add, subtract, multiply, divide
- Matrix operations: matmul
- Reduction operations: sum, mean
- Activation functions: relu, sigmoid, tanh

Design Note:
    Backward operations are implemented as standalone functions that receive
    tape components (nodes, registry) rather than the full GradientTape.
    This avoids circular imports and allows clear separation of concerns.

Architecture:
    Each backward function follows the same pattern:
    1. Extract saved tensors from the node at the given index
    2. Call the core backward function from odyssey.core
    3. Store computed gradients in the registry for each input variable

Example:
    # Called during tape.backward() for an addition operation
    backward_add(nodes, registry, node_idx, grad_output)
    # This:
    # 1. Gets saved tensors a and b from nodes[node_idx]
    # 2. Calls add_backward(grad_output, a, b) to get gradients
    # 3. Stores gradients in registry for the input variables
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.core.arithmetic import (
    add_backward,
    subtract_backward,
    multiply_backward,
    divide_backward,
)
from odyssey.core.reduction import sum_backward, mean_backward
from odyssey.core.matrix import matmul_backward
from odyssey.core.activation import (
    relu_backward,
    sigmoid_backward,
    tanh_backward,
)
from odyssey.core.linear import linear_backward
from odyssey.core.conv import conv2d_backward
from odyssey.core.pooling import maxpool2d_backward
from odyssey.core.loss import cross_entropy_backward

# Import types from tape_types (avoids circular import with tape.mojo)
from odyssey.autograd.tape_types import TapeNode, VariableRegistry


# ============================================================================
# Binary Arithmetic Operations
# ============================================================================


def backward_add(
    nodes: List[TapeNode],
    mut registry: VariableRegistry,
    idx: Int,
    grad_output: AnyTensor,
) raises:
    """Backward pass for element-wise addition.

    Computes gradients for: c = a + b
    Given: grad_output = dL/dc
    Returns: dL/da = grad_output, dL/db = grad_output

    Args:
        nodes: List of tape nodes containing saved tensors.
        registry: Variable registry to store computed gradients.
        idx: Index of the addition node in the tape.
        grad_output: Gradient flowing back from downstream operations.
    """
    if len(nodes[idx].saved.tensors) < 2:
        return
    var a = nodes[idx].saved.tensors[0].copy()
    var b = nodes[idx].saved.tensors[1].copy()
    var result = add_backward(grad_output, a, b)
    if len(nodes[idx].input_ids) >= 1:
        registry.set_grad(nodes[idx].input_ids[0], result.grad_a)
    if len(nodes[idx].input_ids) >= 2:
        registry.set_grad(nodes[idx].input_ids[1], result.grad_b)


def backward_subtract(
    nodes: List[TapeNode],
    mut registry: VariableRegistry,
    idx: Int,
    grad_output: AnyTensor,
) raises:
    """Backward pass for element-wise subtraction.

    Computes gradients for: c = a - b
    Given: grad_output = dL/dc
    Returns: dL/da = grad_output, dL/db = -grad_output

    Args:
        nodes: List of tape nodes containing saved tensors.
        registry: Variable registry to store computed gradients.
        idx: Index of the subtraction node in the tape.
        grad_output: Gradient flowing back from downstream operations.
    """
    if len(nodes[idx].saved.tensors) < 2:
        return
    var a = nodes[idx].saved.tensors[0].copy()
    var b = nodes[idx].saved.tensors[1].copy()
    var result = subtract_backward(grad_output, a, b)
    if len(nodes[idx].input_ids) >= 1:
        registry.set_grad(nodes[idx].input_ids[0], result.grad_a)
    if len(nodes[idx].input_ids) >= 2:
        registry.set_grad(nodes[idx].input_ids[1], result.grad_b)


def backward_multiply(
    nodes: List[TapeNode],
    mut registry: VariableRegistry,
    idx: Int,
    grad_output: AnyTensor,
) raises:
    """Backward pass for element-wise multiplication.

    Computes gradients for: c = a * b
    Given: grad_output = dL/dc
    Returns: dL/da = grad_output * b, dL/db = grad_output * a

    Args:
        nodes: List of tape nodes containing saved tensors.
        registry: Variable registry to store computed gradients.
        idx: Index of the multiplication node in the tape.
        grad_output: Gradient flowing back from downstream operations.
    """
    if len(nodes[idx].saved.tensors) < 2:
        return
    var a = nodes[idx].saved.tensors[0].copy()
    var b = nodes[idx].saved.tensors[1].copy()
    var result = multiply_backward(grad_output, a, b)
    if len(nodes[idx].input_ids) >= 1:
        registry.set_grad(nodes[idx].input_ids[0], result.grad_a)
    if len(nodes[idx].input_ids) >= 2:
        registry.set_grad(nodes[idx].input_ids[1], result.grad_b)


def backward_divide(
    nodes: List[TapeNode],
    mut registry: VariableRegistry,
    idx: Int,
    grad_output: AnyTensor,
) raises:
    """Backward pass for element-wise division.

    Computes gradients for: c = a / b
    Given: grad_output = dL/dc
    Returns: dL/da = grad_output / b, dL/db = -grad_output * a / (b^2)

    Args:
        nodes: List of tape nodes containing saved tensors.
        registry: Variable registry to store computed gradients.
        idx: Index of the division node in the tape.
        grad_output: Gradient flowing back from downstream operations.
    """
    if len(nodes[idx].saved.tensors) < 2:
        return
    var a = nodes[idx].saved.tensors[0].copy()
    var b = nodes[idx].saved.tensors[1].copy()
    var result = divide_backward(grad_output, a, b)
    if len(nodes[idx].input_ids) >= 1:
        registry.set_grad(nodes[idx].input_ids[0], result.grad_a)
    if len(nodes[idx].input_ids) >= 2:
        registry.set_grad(nodes[idx].input_ids[1], result.grad_b)


# ============================================================================
# Reduction Operations
# ============================================================================


def backward_sum(
    nodes: List[TapeNode],
    mut registry: VariableRegistry,
    idx: Int,
    grad_output: AnyTensor,
) raises:
    """Backward pass for sum reduction.

    Computes gradient for: y = sum(x, axis)
    Given: grad_output = dL/dy
    Returns: dL/dx = grad_output (broadcasted to x.shape)

    Args:
        nodes: List of tape nodes containing saved tensors.
        registry: Variable registry to store computed gradients.
        idx: Index of the sum node in the tape.
        grad_output: Gradient flowing back from downstream operations.
    """
    if len(nodes[idx].saved.tensors) < 1:
        return
    var x = nodes[idx].saved.tensors[0]
    var axis = -1
    if len(nodes[idx].saved.scalars) >= 1:
        axis = Int(nodes[idx].saved.scalars[0])
    var grad_input = sum_backward(grad_output, x, axis)
    if len(nodes[idx].input_ids) >= 1:
        registry.set_grad(nodes[idx].input_ids[0], grad_input)


def backward_mean(
    nodes: List[TapeNode],
    mut registry: VariableRegistry,
    idx: Int,
    grad_output: AnyTensor,
) raises:
    """Backward pass for mean reduction.

    Computes gradient for: y = mean(x, axis)
    Given: grad_output = dL/dy
    Returns: dL/dx = grad_output / N (broadcasted to x.shape, scaled by 1/N)

    Args:
        nodes: List of tape nodes containing saved tensors.
        registry: Variable registry to store computed gradients.
        idx: Index of the mean node in the tape.
        grad_output: Gradient flowing back from downstream operations.
    """
    if len(nodes[idx].saved.tensors) < 1:
        return
    var x = nodes[idx].saved.tensors[0]
    var axis = -1
    if len(nodes[idx].saved.scalars) >= 1:
        axis = Int(nodes[idx].saved.scalars[0])
    var grad_input = mean_backward(grad_output, x, axis)
    if len(nodes[idx].input_ids) >= 1:
        registry.set_grad(nodes[idx].input_ids[0], grad_input)


# ============================================================================
# Matrix Operations
# ============================================================================


def backward_matmul(
    nodes: List[TapeNode],
    mut registry: VariableRegistry,
    idx: Int,
    grad_output: AnyTensor,
) raises:
    """Backward pass for matrix multiplication.

    Computes gradients for: C = A @ B
    Given: grad_output = dL/dC
    Returns: dL/dA = grad_output @ B^T, dL/dB = A^T @ grad_output

    Args:
        nodes: List of tape nodes containing saved tensors.
        registry: Variable registry to store computed gradients.
        idx: Index of the matmul node in the tape.
        grad_output: Gradient flowing back from downstream operations.
    """
    if len(nodes[idx].saved.tensors) < 2:
        return
    var a = nodes[idx].saved.tensors[0].copy()
    var b = nodes[idx].saved.tensors[1].copy()
    var result = matmul_backward(grad_output, a, b)
    if len(nodes[idx].input_ids) >= 1:
        registry.set_grad(nodes[idx].input_ids[0], result.grad_a)
    if len(nodes[idx].input_ids) >= 2:
        registry.set_grad(nodes[idx].input_ids[1], result.grad_b)


# ============================================================================
# Activation Functions
# ============================================================================


def backward_relu(
    nodes: List[TapeNode],
    mut registry: VariableRegistry,
    idx: Int,
    grad_output: AnyTensor,
) raises:
    """Backward pass for ReLU activation.

    Computes gradient for: y = ReLU(x) = max(0, x)
    Given: grad_output = dL/dy
    Returns: dL/dx = grad_output * (x > 0)

    Args:
        nodes: List of tape nodes containing saved tensors.
        registry: Variable registry to store computed gradients.
        idx: Index of the ReLU node in the tape.
        grad_output: Gradient flowing back from downstream operations.
    """
    if len(nodes[idx].saved.tensors) < 1:
        return
    var x = nodes[idx].saved.tensors[0]
    var grad_input = relu_backward(grad_output, x)
    if len(nodes[idx].input_ids) >= 1:
        registry.set_grad(nodes[idx].input_ids[0], grad_input)


def backward_sigmoid(
    nodes: List[TapeNode],
    mut registry: VariableRegistry,
    idx: Int,
    grad_output: AnyTensor,
) raises:
    """Backward pass for sigmoid activation.

    Computes gradient for: y = sigmoid(x) = 1 / (1 + exp(-x))
    Given: grad_output = dL/dy, output = y
    Returns: dL/dx = grad_output * y * (1 - y)

    Note: The saved tensor is the OUTPUT of sigmoid, not the input.
          This is more numerically stable and efficient.

    Args:
        nodes: List of tape nodes containing saved tensors.
        registry: Variable registry to store computed gradients.
        idx: Index of the sigmoid node in the tape.
        grad_output: Gradient flowing back from downstream operations.
    """
    if len(nodes[idx].saved.tensors) < 1:
        return
    var output = nodes[idx].saved.tensors[0]
    var grad_input = sigmoid_backward(grad_output, output)
    if len(nodes[idx].input_ids) >= 1:
        registry.set_grad(nodes[idx].input_ids[0], grad_input)


def backward_flatten(
    nodes: List[TapeNode],
    mut registry: VariableRegistry,
    idx: Int,
    grad_output: AnyTensor,
) raises:
    """Backward pass for flatten.

    Computes gradient for: y = flatten(x)
    Given: grad_output = dL/dy (rank-2)
    Returns: dL/dx = reshape(grad_output, input_shape)

    Saved layout:
        shapes[0] = input.shape() (original rank-N shape).

    Args:
        nodes: List of tape nodes containing saved shapes.
        registry: Variable registry to store computed gradients.
        idx: Index of the flatten node in the tape.
        grad_output: Gradient flowing back from downstream operations.
    """
    if len(nodes[idx].saved.shapes) < 1:
        return
    var input_shape = nodes[idx].saved.shapes[0].copy()
    var grad_input = grad_output.reshape(input_shape)
    if len(nodes[idx].input_ids) >= 1:
        registry.set_grad(nodes[idx].input_ids[0], grad_input)


def backward_tanh(
    nodes: List[TapeNode],
    mut registry: VariableRegistry,
    idx: Int,
    grad_output: AnyTensor,
) raises:
    """Backward pass for tanh activation.

    Computes gradient for: y = tanh(x)
    Given: grad_output = dL/dy, output = y
    Returns: dL/dx = grad_output * (1 - y^2)

    Note: The saved tensor is the OUTPUT of tanh, not the input.
          This is more numerically stable and efficient.

    Args:
        nodes: List of tape nodes containing saved tensors.
        registry: Variable registry to store computed gradients.
        idx: Index of the tanh node in the tape.
        grad_output: Gradient flowing back from downstream operations.
    """
    if len(nodes[idx].saved.tensors) < 1:
        return
    var output = nodes[idx].saved.tensors[0]
    var grad_input = tanh_backward(grad_output, output)
    if len(nodes[idx].input_ids) >= 1:
        registry.set_grad(nodes[idx].input_ids[0], grad_input)


# ============================================================================
# Phase 2 substrate ops (convnet primitives)
# ============================================================================


def backward_linear(
    nodes: List[TapeNode],
    mut registry: VariableRegistry,
    idx: Int,
    grad_output: AnyTensor,
) raises:
    """Backward pass for linear (fully connected) layer.

    Saved layout:
        tensors[0] = input x  (batch, in_features)
        tensors[1] = weights  (out_features, in_features)

    Routes:
        input_ids[0] -> grad_input
        input_ids[1] -> grad_weights
        input_ids[2] -> grad_bias (if 3 input_ids)
    """
    if len(nodes[idx].saved.tensors) < 2:
        return
    var x = nodes[idx].saved.tensors[0].copy()
    var weights = nodes[idx].saved.tensors[1].copy()
    var grads = linear_backward(grad_output, x, weights)
    if len(nodes[idx].input_ids) >= 1:
        registry.set_grad(nodes[idx].input_ids[0], grads.grad_input)
    if len(nodes[idx].input_ids) >= 2:
        registry.set_grad(nodes[idx].input_ids[1], grads.grad_weights)
    if len(nodes[idx].input_ids) >= 3:
        registry.set_grad(nodes[idx].input_ids[2], grads.grad_bias)


def backward_conv2d(
    nodes: List[TapeNode],
    mut registry: VariableRegistry,
    idx: Int,
    grad_output: AnyTensor,
) raises:
    """Backward pass for 2D convolution.

    Saved layout:
        tensors[0] = input x   (batch, in_C, H, W)
        tensors[1] = kernel    (out_C, in_C, kH, kW)
        scalars[0] = stride  (Float64-cast Int)
        scalars[1] = padding (Float64-cast Int)

    Routes (same as linear):
        input_ids[0] -> grad_input
        input_ids[1] -> grad_weights
        input_ids[2] -> grad_bias (if 3 input_ids)
    """
    if len(nodes[idx].saved.tensors) < 2:
        return
    if len(nodes[idx].saved.scalars) < 2:
        return
    var x = nodes[idx].saved.tensors[0].copy()
    var kernel = nodes[idx].saved.tensors[1].copy()
    var stride = Int(nodes[idx].saved.scalars[0])
    var padding = Int(nodes[idx].saved.scalars[1])
    var grads = conv2d_backward(grad_output, x, kernel, stride, padding)
    if len(nodes[idx].input_ids) >= 1:
        registry.set_grad(nodes[idx].input_ids[0], grads.grad_input)
    if len(nodes[idx].input_ids) >= 2:
        registry.set_grad(nodes[idx].input_ids[1], grads.grad_weights)
    if len(nodes[idx].input_ids) >= 3:
        registry.set_grad(nodes[idx].input_ids[2], grads.grad_bias)


def backward_maxpool2d(
    nodes: List[TapeNode],
    mut registry: VariableRegistry,
    idx: Int,
    grad_output: AnyTensor,
) raises:
    """Backward pass for 2D max pooling.

    Saved layout:
        tensors[0] = input x   (needed to recompute argmax positions)
        scalars[0] = kernel_size (Float64-cast Int)
        scalars[1] = stride      (Float64-cast Int)
        scalars[2] = padding     (Float64-cast Int)
    """
    if len(nodes[idx].saved.tensors) < 1:
        return
    if len(nodes[idx].saved.scalars) < 3:
        return
    var x = nodes[idx].saved.tensors[0].copy()
    var kernel_size = Int(nodes[idx].saved.scalars[0])
    var stride = Int(nodes[idx].saved.scalars[1])
    var padding = Int(nodes[idx].saved.scalars[2])
    var grad_input = maxpool2d_backward(
        grad_output, x, kernel_size, stride, padding
    )
    if len(nodes[idx].input_ids) >= 1:
        registry.set_grad(nodes[idx].input_ids[0], grad_input)


def backward_cross_entropy(
    nodes: List[TapeNode],
    mut registry: VariableRegistry,
    idx: Int,
    grad_output: AnyTensor,
) raises:
    """Backward pass for cross-entropy loss.

    Saved layout:
        tensors[0] = logits  (batch, num_classes)
        tensors[1] = targets (batch, num_classes) — non-trainable, not routed.

    Only logits receive a gradient (input_ids[0]). Targets are non-trainable.
    """
    if len(nodes[idx].saved.tensors) < 2:
        return
    var logits = nodes[idx].saved.tensors[0].copy()
    var targets = nodes[idx].saved.tensors[1].copy()
    var grad_logits = cross_entropy_backward(grad_output, logits, targets)
    if len(nodes[idx].input_ids) >= 1:
        registry.set_grad(nodes[idx].input_ids[0], grad_logits)
