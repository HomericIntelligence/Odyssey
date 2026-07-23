"""LARS optimizer wrapped as an OO `Optimizer` (Variable/GradientTape API).

Delegates math to `lars_step` in `odyssey.training.optimizers.lars`.
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
from odyssey.training.optimizers.lars import lars_step


@fieldwise_init
struct LARS(Copyable, Movable, Optimizer):
    """LARS (Layer-wise Adaptive Rate Scaling) OO wrapper."""

    var learning_rate: Float64
    var momentum: Float64
    var weight_decay: Float64
    var trust_coefficient: Float64
    var epsilon: Float64
    var velocities: Dict[Int, AnyTensor]

    def __init__(
        out self,
        learning_rate: Float64,
        momentum: Float64 = 0.9,
        weight_decay: Float64 = 0.0001,
        trust_coefficient: Float64 = 0.001,
        epsilon: Float64 = 1e-8,
    ):
        """Initialize LARS optimizer."""
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.trust_coefficient = trust_coefficient
        self.epsilon = epsilon
        self.velocities = Dict[Int, AnyTensor]()

    def step(
        mut self, mut parameters: List[Variable], mut tape: GradientTape
    ) raises:
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
            var result = lars_step(
                param_data,
                grad,
                velocity,
                self.learning_rate,
                self.momentum,
                self.weight_decay,
                self.trust_coefficient,
                self.epsilon,
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
