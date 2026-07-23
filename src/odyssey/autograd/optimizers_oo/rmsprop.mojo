"""RMSprop optimizer wrapped as an OO `Optimizer` (Variable/GradientTape API).

Math is delegated to `rmsprop_step` in
`odyssey.training.optimizers.rmsprop`. State (square averages and optional
momentum buffer) is keyed by parameter ID in a Dict.

This is the canonical OO wrapper for RMSprop — the previous inlined
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
from odyssey.training.optimizers.rmsprop import rmsprop_step

# `Optional` is provided by the Mojo 1.0 prelude (see rmsprop.mojo which
# uses `Optional[AnyTensor]` without an explicit import); no import line
# is needed here.


@fieldwise_init
struct RMSprop(Copyable, Movable, Optimizer):
    """RMSprop (Root Mean Square Propagation) optimizer (OO delegator).

    Update rule (delegated to `rmsprop_step`):

        Without momentum:
            v_t = α * v_{t-1} + (1 - α) * g_t²
            θ_t = θ_{t-1} - lr * g_t / (√v_t + ε)

        With momentum:
            buf_t = momentum * buf_{t-1} + g_t / (√v_t + ε)
            θ_t   = θ_{t-1} - lr * buf_t

    `t` is passed as `1` to `rmsprop_step` because RMSprop has no bias
    correction (the canonical validates `t>0` but does not use it
    semantically).
    """

    var learning_rate: Float64
    var alpha: Float64
    var epsilon: Float64
    var weight_decay: Float64
    var momentum: Float64
    var v_buffers: Dict[Int, AnyTensor]
    var m_buffers: Dict[Int, AnyTensor]

    def __init__(
        out self,
        learning_rate: Float64 = 0.01,
        alpha: Float64 = 0.99,
        epsilon: Float64 = 1e-8,
        weight_decay: Float64 = 0.0,
        momentum: Float64 = 0.0,
    ):
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.v_buffers = Dict[Int, AnyTensor]()
        self.m_buffers = Dict[Int, AnyTensor]()

    def step(
        mut self, mut parameters: List[Variable], mut tape: GradientTape
    ) raises:
        """Update parameters via RMSprop (delegates to rmsprop_step)."""
        for i in range(len(parameters)):
            if not parameters[i].requires_grad:
                continue
            var param_id = parameters[i].id
            if not tape.registry.has_gradient(param_id):
                continue
            var grad = tape.registry.get_grad(param_id)
            var param_data = parameters[i].data
            if param_id not in self.v_buffers:
                self.v_buffers[param_id] = zeros_like(param_data)
                if self.momentum > 0.0:
                    self.m_buffers[param_id] = zeros_like(param_data)
            var v = self.v_buffers[param_id]
            var buf: Optional[AnyTensor] = None
            if self.momentum > 0.0 and param_id in self.m_buffers:
                buf = self.m_buffers[param_id]
            var result = rmsprop_step(
                param_data,
                grad,
                v,
                1,
                self.learning_rate,
                self.alpha,
                self.epsilon,
                self.weight_decay,
                self.momentum,
                buf,
            )
            parameters[i].data = result[0]
            self.v_buffers[param_id] = result[1]
            if self.momentum > 0.0:
                self.m_buffers[param_id] = result[2]

    def zero_grad(self, mut tape: GradientTape):
        zero_grad_impl(tape)

    def get_lr(self) -> Float64:
        return self.learning_rate

    def set_lr(mut self, lr: Float64) raises:
        validate_learning_rate(lr)
        self.learning_rate = lr
