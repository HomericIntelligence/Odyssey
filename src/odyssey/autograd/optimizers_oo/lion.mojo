"""Lion optimizer wrapped as an OO `Optimizer` (Variable/GradientTape API).

Pure-functional math is delegated to `lion_step` in
`odyssey.training.optimizers.lion`. State (signed momentum) is held per
parameter ID in a Dict.

Reference: Chen et al. 2023, "Symbolic Discovery of Optimization Algorithms".
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
from odyssey.training.optimizers.lion import lion_step


@fieldwise_init
struct Lion(Copyable, Movable, Optimizer):
    """Lion (EvoLved Sign Momentum) OO wrapper.

    State: a single signed-momentum buffer per parameter ID.
    """

    var learning_rate: Float64
    var beta1: Float64
    var beta2: Float64
    var weight_decay: Float64
    var momenta: Dict[Int, AnyTensor]

    def __init__(
        out self,
        learning_rate: Float64,
        beta1: Float64 = 0.9,
        beta2: Float64 = 0.99,
        weight_decay: Float64 = 0.0,
    ):
        """Initialize Lion optimizer."""
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.momenta = Dict[Int, AnyTensor]()

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
            if param_id not in self.momenta:
                self.momenta[param_id] = zeros_like(param_data)
            var momentum = self.momenta[param_id]
            var result = lion_step(
                param_data,
                grad,
                momentum,
                self.learning_rate,
                self.beta1,
                self.beta2,
                self.weight_decay,
            )
            parameters[i].data = result[0]
            self.momenta[param_id] = result[1]

    def zero_grad(self, mut tape: GradientTape):
        zero_grad_impl(tape)

    def get_lr(self) -> Float64:
        return self.learning_rate

    def set_lr(mut self, lr: Float64) raises:
        validate_learning_rate(lr)
        self.learning_rate = lr
