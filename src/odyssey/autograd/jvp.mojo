"""Forward-mode Jacobian-vector products for dynamic tensors.

This module represents a forward-mode value as a ``DualTensor``:

    primal + epsilon * tangent

The primitive rules propagate both components through a computation.  A
``JVPFunction`` is a concrete, copyable struct rather than a bare function
value.  That pattern supports deterministic captured model state on Mojo 1.0
without relying on escaping closures or error-prone ``thin`` annotations.

Only float16, float32, and float64 ``AnyTensor`` values are accepted.  A primal
and its tangent must have exactly the same shape and dtype.  BF16 is rejected
until every downstream primitive dispatcher used here supports it.
"""

from odyssey.core.activation import relu, relu_backward
from odyssey.core.arithmetic import add, multiply, subtract
from odyssey.core.matrix import matmul
from odyssey.core.validation import validate_matching_tensors
from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros_like


def _validate_jvp_dtype(tensor: AnyTensor, name: String) raises:
    var dtype = tensor.dtype()
    if (
        dtype != DType.float16
        and dtype != DType.float32
        and dtype != DType.float64
    ):
        raise Error(
            name
            + ": JVP supports only float16, float32, and float64; got "
            + String(dtype)
        )


struct DualTensor(Copyable, Movable):
    """A dynamic tensor value and its forward-mode tangent.

    ``primal`` and ``tangent`` are owned, reference-counted ``AnyTensor``
    handles.  The constructor consumes its local arguments into the fields and
    validates shape and supported dtype before any rule can observe the pair.
    """

    var primal: AnyTensor
    var tangent: AnyTensor

    def __init__(
        out self, var primal: AnyTensor, var tangent: AnyTensor
    ) raises:
        validate_matching_tensors(
            primal, tangent, "DualTensor.primal", "DualTensor.tangent"
        )
        _validate_jvp_dtype(primal, "DualTensor.primal")
        _validate_jvp_dtype(tangent, "DualTensor.tangent")
        self.primal = primal^
        self.tangent = tangent^


trait JVPFunction(Copyable, Movable):
    """Callable model interface consumed by :func:`jvp`.

    Implement this trait on a struct and store deterministic weights or other
    model state in its fields.  Using a trait-bound concrete type keeps state
    available to the compiler and avoids escaping-closure/function-value
    lifetime constraints in Mojo 1.0.
    """

    def __call__(self, input: DualTensor) raises -> DualTensor:
        ...


def dual_constant(var primal: AnyTensor) raises -> DualTensor:
    """Lift a floating tensor into forward mode with an all-zero tangent."""
    var tangent = zeros_like(primal)
    return DualTensor(primal^, tangent^)


def dual_add(lhs: DualTensor, rhs: DualTensor) raises -> DualTensor:
    """Forward rule for elementwise addition."""
    var primal = add(lhs.primal, rhs.primal)
    var tangent = add(lhs.tangent, rhs.tangent)
    return DualTensor(primal^, tangent^)


def dual_subtract(lhs: DualTensor, rhs: DualTensor) raises -> DualTensor:
    """Forward rule for elementwise subtraction."""
    var primal = subtract(lhs.primal, rhs.primal)
    var tangent = subtract(lhs.tangent, rhs.tangent)
    return DualTensor(primal^, tangent^)


def dual_multiply(lhs: DualTensor, rhs: DualTensor) raises -> DualTensor:
    """Forward rule for an elementwise product.

    ``(lhs * rhs)_dot = lhs_dot * rhs + lhs * rhs_dot``.
    """
    var primal = multiply(lhs.primal, rhs.primal)
    var lhs_term = multiply(lhs.tangent, rhs.primal)
    var rhs_term = multiply(lhs.primal, rhs.tangent)
    var tangent = add(lhs_term, rhs_term)
    return DualTensor(primal^, tangent^)


def dual_matmul(lhs: DualTensor, rhs: DualTensor) raises -> DualTensor:
    """Forward rule for matrix multiplication.

    ``(A @ B)_dot = A_dot @ B + A @ B_dot``.
    """
    var primal = matmul(lhs.primal, rhs.primal)
    var lhs_term = matmul(lhs.tangent, rhs.primal)
    var rhs_term = matmul(lhs.primal, rhs.tangent)
    var tangent = add(lhs_term, rhs_term)
    return DualTensor(primal^, tangent^)


def dual_relu(input: DualTensor) raises -> DualTensor:
    """Forward rule for ReLU.

    The tangent is masked by ``input.primal > 0``.  At the non-differentiable
    kink (zero), Odyssey follows its existing ReLU backward convention and
    chooses a tangent of zero.
    """
    var primal = relu(input.primal)
    var tangent = relu_backward(input.tangent, input.primal)
    return DualTensor(primal^, tangent^)


def dual_linear(
    input: DualTensor, weight: DualTensor, bias: DualTensor
) raises -> DualTensor:
    """Compose matmul and addition for ``input @ weight + bias``.

    Weight and bias may themselves carry tangents.  Use :func:`dual_constant`
    for ordinary fixed model parameters.
    """
    var projected = dual_matmul(input, weight)
    var output = dual_add(projected, bias)
    return output^


def jvp[
    F: JVPFunction
](forward: F, primal: AnyTensor, tangent: AnyTensor) raises -> DualTensor:
    """Evaluate ``forward`` and its Jacobian-vector product at ``primal``.

    Args:
        forward: Concrete callable struct implementing ``JVPFunction``.
        primal: Floating input tensor ``x``.
        tangent: Floating seed vector ``v`` with the same shape/dtype as ``x``.

    Returns:
        ``DualTensor`` containing ``forward(x)`` and ``J(forward)(x) @ v``.
    """
    # AnyTensor copies are reference-counted owned handles.  Constructing the
    # seed this way keeps the caller's handles valid while the local DualTensor
    # has explicit ownership of its two fields.
    var seed = DualTensor(primal.copy(), tangent.copy())
    var result = forward(seed)
    return result^


def jvp_only[
    F: JVPFunction
](forward: F, primal: AnyTensor, tangent: AnyTensor) raises -> AnyTensor:
    """Return only ``J(forward)(primal) @ tangent``.

    The returned tangent is numerically identical to ``jvp(...).tangent`` and
    has a smaller retained return payload because the primal is not returned.
    This variant still computes the same primal intermediates required by the
    forward-mode rules; it makes no claim about peak memory or allocation count.
    """
    var result = jvp(forward, primal, tangent)
    return result.tangent.copy()
