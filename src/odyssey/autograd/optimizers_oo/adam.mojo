"""Adam optimizer wrapped as an OO `Optimizer` (Variable/GradientTape API).

Math is delegated to `adam_step` in `odyssey.training.optimizers.adam`.
State (first + second moment buffers) is keyed by parameter ID in a Dict.

This is the canonical OO wrapper for Adam — the previous inlined
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
from odyssey.training.optimizers.adam import adam_step


@fieldwise_init
struct Adam(Copyable, Movable, Optimizer):
    """Adam (Adaptive Moment Estimation) optimizer (OO delegator).

    Update rule (delegated to `adam_step`):
        m = β₁ * m + (1 - β₁) * g
        v = β₂ * v + (1 - β₂) * g²
        m̂ = m / (1 - β₁^t);  v̂ = v / (1 - β₂^t)
        θ = θ - lr * m̂ / (√v̂ + ε)

    Weight decay is *coupled* (L2 folded into the gradient by `adam_step`
    before accumulation).

    `self.t` increments once per `step()` call and is passed to every
    per-parameter `adam_step` invocation so bias correction is consistent
    across the parameter list.
    """

    var learning_rate: Float64
    var beta1: Float64
    var beta2: Float64
    var epsilon: Float64
    var weight_decay: Float64
    var t: Int
    var m_buffers: Dict[Int, AnyTensor]
    var v_buffers: Dict[Int, AnyTensor]

    def __init__(
        out self,
        learning_rate: Float64 = 0.001,
        beta1: Float64 = 0.9,
        beta2: Float64 = 0.999,
        epsilon: Float64 = 1e-8,
        weight_decay: Float64 = 0.0,
    ):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.t = 0
        self.m_buffers = Dict[Int, AnyTensor]()
        self.v_buffers = Dict[Int, AnyTensor]()

    def step(
        mut self, mut parameters: List[Variable], mut tape: GradientTape
    ) raises:
        """Update parameters via Adam (delegates to adam_step).

        `len(self.m_buffers)` and `len(self.v_buffers)` give the count of
        parameters with initialized moment buffers.
        """
        self.t += 1
        for i in range(len(parameters)):
            if not parameters[i].requires_grad:
                continue
            var param_id = parameters[i].id
            if not tape.registry.has_gradient(param_id):
                continue
            var grad = tape.registry.get_grad(param_id)
            var param_data = parameters[i].data
            if param_id not in self.m_buffers:
                self.m_buffers[param_id] = zeros_like(param_data)
                self.v_buffers[param_id] = zeros_like(param_data)
            var m = self.m_buffers[param_id]
            var v = self.v_buffers[param_id]
            var result = adam_step(
                param_data,
                grad,
                m,
                v,
                self.t,
                self.learning_rate,
                self.beta1,
                self.beta2,
                self.epsilon,
                self.weight_decay,
            )
            parameters[i].data = result[0]
            self.m_buffers[param_id] = result[1]
            self.v_buffers[param_id] = result[2]

    def zero_grad(self, mut tape: GradientTape):
        zero_grad_impl(tape)

    def get_lr(self) -> Float64:
        return self.learning_rate

    def set_lr(mut self, lr: Float64) raises:
        validate_learning_rate(lr)
        self.learning_rate = lr
