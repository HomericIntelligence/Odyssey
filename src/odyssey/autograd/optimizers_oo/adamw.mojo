"""AdamW optimizer wrapped as an OO `Optimizer` (Variable/GradientTape API).

Math is delegated to `adamw_step` in `odyssey.training.optimizers.adamw`.
State (first + second moment buffers) is keyed by parameter ID in a Dict.

Weight decay is *decoupled*: applied directly to parameters after the
Adam update (Loshchilov & Hutter 2019).

This is the canonical OO wrapper for AdamW — the previous inlined
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
from odyssey.training.optimizers.adamw import adamw_step


@fieldwise_init
struct AdamW(Copyable, Movable, Optimizer):
    """AdamW (Adam with decoupled weight decay) optimizer (OO delegator).

    Update rule (delegated to `adamw_step`):
        m = β₁ * m + (1 - β₁) * g
        v = β₂ * v + (1 - β₂) * g²
        m̂ = m / (1 - β₁^t);  v̂ = v / (1 - β₂^t)
        θ = θ - lr * m̂ / (√v̂ + ε)
        θ = θ - weight_decay * θ            # decoupled
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
        weight_decay: Float64 = 0.01,
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
        """Update parameters via AdamW (delegates to adamw_step)."""
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
            var result = adamw_step(
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
