"""AdaGrad optimizer wrapped as an OO `Optimizer` (Variable/GradientTape API).

Math is delegated to `adagrad_step` in
`odyssey.training.optimizers.adagrad`. State (accumulator of squared
gradients) is keyed by parameter ID in a Dict.

This is the canonical OO wrapper for AdaGrad — the previous inlined
implementation in `src/odyssey/autograd/optimizers.mojo` has been moved
here and now thin-delegates to the canonical functional step.
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
from odyssey.training.optimizers.adagrad import adagrad_step


@fieldwise_init
struct AdaGrad(Copyable, Movable, Optimizer):
    """AdaGrad (Adaptive Gradient) optimizer (OO delegator).

    Update rule (delegated to `adagrad_step`):
        accum_t = accum_{t-1} + grad_t²       (NOT folded with weight decay)
        update  = lr * grad / (√accum_t + ε) + (weight_decay * θ) if wd>0
        θ_t    = θ_{t-1} - update

    Weight decay is applied *outside* the adaptive scaling — this matches
    the original autograd AdaGrad semantics so the refactor introduces
    no behavioral drift.
    """

    var learning_rate: Float64
    var epsilon: Float64
    var weight_decay: Float64
    var G_buffers: Dict[Int, AnyTensor]

    def __init__(
        out self,
        learning_rate: Float64,
        epsilon: Float64 = 1e-10,
        weight_decay: Float64 = 0.0,
    ):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.G_buffers = Dict[Int, AnyTensor]()

    def step(
        mut self, mut parameters: List[Variable], mut tape: GradientTape
    ) raises:
        """Update parameters via AdaGrad (delegates to adagrad_step)."""
        for i in range(len(parameters)):
            if not parameters[i].requires_grad:
                continue
            var param_id = parameters[i].id
            if not tape.registry.has_gradient(param_id):
                continue
            var grad = tape.registry.get_grad(param_id)
            var param_data = parameters[i].data
            if param_id not in self.G_buffers:
                self.G_buffers[param_id] = zeros_like(param_data)
            var accum = self.G_buffers[param_id]
            var result = adagrad_step(
                param_data,
                grad,
                accum,
                self.learning_rate,
                self.epsilon,
                self.weight_decay,
            )
            parameters[i].data = result[0]
            self.G_buffers[param_id] = result[1]

    def zero_grad(self, mut tape: GradientTape):
        zero_grad_impl(tape)

    def get_lr(self) -> Float64:
        return self.learning_rate

    def set_lr(mut self, lr: Float64) raises:
        validate_learning_rate(lr)
        self.learning_rate = lr

    def reset_accumulators(mut self):
        """Reset accumulated squared gradient buffers."""
        self.G_buffers.clear()
