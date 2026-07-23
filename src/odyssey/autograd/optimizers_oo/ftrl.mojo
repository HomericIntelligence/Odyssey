"""FTRL-Proximal optimizer wrapped as an OO `Optimizer` API.

Online-learning optimizer with L1-sparsity (McMahan et al. 2013).

Delegates math to `ftrl_step` in `odyssey.training.optimizers.ftrl`. State:
two buffers per parameter ID — `z` (linearized gradient/weight sum) and `n`
(sum of squared gradients).
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
from odyssey.training.optimizers.ftrl import ftrl_step


@fieldwise_init
struct FTRLProximal(Copyable, Movable, Optimizer):
    """FTRL-Proximal OO wrapper with L1/L2 regularization."""

    var learning_rate: Float64
    var alpha: Float64
    var beta: Float64
    var lambda1: Float64
    var lambda2: Float64
    var z_buffers: Dict[Int, AnyTensor]
    var n_buffers: Dict[Int, AnyTensor]

    def __init__(
        out self,
        learning_rate: Float64,
        alpha: Float64 = 0.1,
        beta: Float64 = 1.0,
        lambda1: Float64 = 0.0,
        lambda2: Float64 = 0.0,
    ):
        """Initialize FTRL-Proximal optimizer."""
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.beta = beta
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.z_buffers = Dict[Int, AnyTensor]()
        self.n_buffers = Dict[Int, AnyTensor]()

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
            if param_id not in self.z_buffers:
                self.z_buffers[param_id] = zeros_like(param_data)
                self.n_buffers[param_id] = zeros_like(param_data)
            var z = self.z_buffers[param_id]
            var n = self.n_buffers[param_id]
            var result = ftrl_step(
                param_data,
                grad,
                z,
                n,
                self.learning_rate,
                self.alpha,
                self.beta,
                self.lambda1,
                self.lambda2,
            )
            parameters[i].data = result[0]
            self.z_buffers[param_id] = result[1]
            self.n_buffers[param_id] = result[2]

    def zero_grad(self, mut tape: GradientTape):
        zero_grad_impl(tape)

    def get_lr(self) -> Float64:
        return self.learning_rate

    def set_lr(mut self, lr: Float64) raises:
        validate_learning_rate(lr)
        self.learning_rate = lr
