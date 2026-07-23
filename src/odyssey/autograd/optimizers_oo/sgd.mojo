"""SGD optimizer wrapped as an OO `Optimizer` (Variable/GradientTape API).

Math is delegated to `sgd_step` in `odyssey.training.optimizers.sgd`.
State (velocity buffers) is keyed by parameter ID in a Dict so the same
struct can drive a Variable list of arbitrary size.

This is the canonical OO wrapper for SGD — the previous inlined
implementation in `src/odyssey/autograd/optimizers.mojo` has been
moved here and now thin-delegates to the canonical functional step.
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros_like
from odyssey.autograd.variable import Variable
from odyssey.autograd.tape import GradientTape
from odyssey.autograd.optimizer_base import (
    Optimizer,
    zero_grad_impl,
    validate_learning_rate,
)
from odyssey.training.optimizers.sgd import sgd_step


@fieldwise_init
struct SGD(Copyable, Movable, Optimizer):
    """Stochastic Gradient Descent optimizer (OO delegator).

    Update rule (delegated to `sgd_step`):
        Without momentum: param = param - lr * grad
        With momentum:    v = m * v + grad;  param = param - lr * v
        With weight decay (L2):
            grad = grad + wd * param   (folded before update)

    Constructor signature intentionally allows an optional `weight_decay`
    kwarg (default 0.0) so `SGD(learning_rate=0.01)` keeps the legacy
    surface and `SGD(learning_rate=0.01, momentum=0.9, weight_decay=1e-4)`
    adds L2 regularization via the canonical `sgd_step`.
    """

    var learning_rate: Float64
    var momentum: Float64
    var weight_decay: Float64
    var velocities: Dict[Int, AnyTensor]

    def __init__(
        out self,
        learning_rate: Float64,
        momentum: Float64 = 0.0,
        weight_decay: Float64 = 0.0,
    ):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = Dict[Int, AnyTensor]()

    def step(
        mut self, mut parameters: List[Variable], mut tape: GradientTape
    ) raises:
        """Update parameters via SGD (delegates to sgd_step)."""
        for i in range(len(parameters)):
            if not parameters[i].requires_grad:
                continue
            var param_id = parameters[i].id
            if not tape.registry.has_gradient(param_id):
                continue
            var grad = tape.registry.get_grad(param_id)
            var param_data = parameters[i].data
            if param_id not in self.velocities:
                self.velocities[param_id] = zeros_like(param_data)
            var velocity = self.velocities[param_id]
            var result = sgd_step(
                param_data,
                grad,
                velocity,
                self.learning_rate,
                self.momentum,
                self.weight_decay,
            )
            parameters[i].data = result[0]
            self.velocities[param_id] = result[1]

    def zero_grad(self, mut tape: GradientTape):
        zero_grad_impl(tape)

    def get_lr(self) -> Float64:
        return self.learning_rate

    def set_lr(mut self, lr: Float64) raises:
        validate_learning_rate(lr)
        self.learning_rate = lr
