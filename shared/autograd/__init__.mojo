"""Autograd - Automatic Differentiation for ML Odyssey.

This module provides gradient computation capabilities for training neural networks.

Note:
    Mojo v0.26.1+ automatically exports all imported symbols to package consumers.
    No ``__all__`` equivalent is needed. Any symbol imported in this module is
    automatically available to users of the ``shared.autograd`` package.

    This module also re-exports selected backward functions from ``shared.core``
    (pooling and dropout backward passes) so callers can import them from a single
    location. These re-exports work cleanly — there is no chain limitation when
    importing from ``shared.autograd`` directly.

    Import from ``shared.autograd`` (works):

    ```mojo
    from shared.autograd import Variable, GradientTape
    from shared.autograd import maxpool2d_backward, dropout_backward
    ```

    Importing from the ``shared`` top-level package is subject to the chain
    limitation described in #3210 and should be avoided.

Core Components:
- Variable: Tensor wrapper with gradient tracking
- GradientTape: Operation recording and backward pass execution
- NoGradContext: Context for disabling gradient computation
- SGD: Stochastic gradient descent optimizer
- Adam: Adaptive Moment Estimation optimizer
- AdaGrad: Adaptive gradient descent optimizer
- StepLR, ExponentialLR: Learning rate schedulers for adaptive decay
- Variable operations: Tape-integrated ops (add, multiply, matmul, etc.)
- Functional helpers: Practical gradient computation for common patterns

Tape-Based Autograd API (Recommended for full automatic differentiation):
    from shared.autograd import Variable, GradientTape
    from shared.autograd import variable_add, variable_multiply, variable_sum

    # Create gradient tape
    var tape = GradientTape()
    tape.enable()

    # Create variables with gradient tracking
    var x = Variable(data, requires_grad=True, tape)
    var y = Variable(weights, requires_grad=True, tape)

    # Perform operations (automatically recorded)
    var z = variable_multiply(x, y, tape)
    var loss = variable_sum(z, tape)

    # Compute gradients automatically
    loss.backward(tape)

    # Access gradients
    var grad_x = tape.get_grad(x.id)
    var grad_y = tape.get_grad(y.id)

Functional Helpers API (For simple gradient patterns):
    from shared.autograd import mse_loss_and_grad, SGD

    # Compute loss and gradient in one call
    var result = mse_loss_and_grad(predictions, targets)
    var loss = result.loss
    var grad = result.grad

    # Update parameters using optimizer
    var optimizer = SGD(learning_rate=0.01)
    optimizer.step(parameters)

Available Variable Operations:
- variable_add, variable_subtract, variable_multiply, variable_divide
- variable_matmul
- variable_sum, variable_mean
- variable_relu, variable_sigmoid, variable_tanh
- variable_neg

Available Loss+Grad Helpers:
- mse_loss_and_grad: Mean squared error (regression)
- bce_loss_and_grad: Binary cross-entropy (binary classification)
- ce_loss_and_grad: Cross-entropy with softmax (multi-class classification)

Gradient Clipping Utilities:
- clip_grad_value_: Clip each gradient element to [-max_value, max_value]
- clip_grad_norm_: Clip gradient L2 norm per parameter
- clip_grad_global_norm_: Clip based on global L2 norm across all parameters

Design Philosophy:
    The autograd module provides two APIs:
    1. Tape-based autograd: Full automatic differentiation with computation graph
    2. Functional helpers: Simple loss+gradient helpers for common patterns

Status:
    ✅ GradientTape with backward() implementation
    ✅ Variable operations with tape recording
    ✅ NoGradContext with enter/exit and convenience functions
    ✅ Functional gradient helpers (mse, bce, ce)
    ✅ SGD optimizer
    ✅ Adam optimizer
    ✅ AdaGrad optimizer
    ✅ AdamW optimizer (decoupled weight decay)
    ✅ RMSprop optimizer
    ✅ Gradient clipping utilities (value, norm, global norm)

References:
    - Tape implementation: tape.mojo
    - Variable operations: variable.mojo
    - Gradient helpers: functional.mojo
    - Design rationale: DESIGN.md
"""

# Core autograd components
from shared.autograd.variable import (
    Variable,
    variable_add,
    variable_subtract,
    variable_multiply,
    variable_divide,
    variable_matmul,
    variable_sum,
    variable_mean,
    variable_relu,
    variable_sigmoid,
    variable_tanh,
    variable_neg,
)

from shared.autograd.tape_types import (
    TapeNode,
    SavedTensors,
    VariableRegistry,
)

from shared.autograd.tape import (
    GradientTape,
    NoGradContext,
    disable_gradient_tracking,
    restore_gradient_tracking,
    # Operation type aliases
    OP_ADD,
    OP_SUBTRACT,
    OP_MULTIPLY,
    OP_DIVIDE,
    OP_MATMUL,
    OP_SUM,
    OP_MEAN,
    OP_RELU,
    OP_SIGMOID,
    OP_TANH,
    OP_NEG,
    OP_POWER,
    OP_SOFTMAX,
    OP_EXP,
    OP_LOG,
    OP_SQRT,
)

from shared.autograd.optimizers import SGD, Adam, AdaGrad, RMSprop, AdamW

from shared.autograd.schedulers import (
    StepLR,
    ExponentialLR,
)

# Gradient clipping utilities are defined in shared.core.grad_utils to avoid
# circular type resolution: importing shared.core.extensor from shared.autograd
# causes extensor.mojo to be compiled twice with distinct type identities.
from shared.core.grad_utils import (
    clip_grad_value_,
    clip_grad_norm_,
    clip_grad_global_norm_,
)

from shared.autograd.functional import (
    LossAndGrad,
    mse_loss_and_grad,
    bce_loss_and_grad,
    ce_loss_and_grad,
    compute_gradient,
    multiply_scalar,
    add_scalar,
    subtract_scalar,
    divide_scalar,
    apply_gradient,
    apply_gradients,
)

# Note: (Mojo v0.26.1): In Mojo, all imported symbols are automatically available
# to package consumers. No __all__ equivalent is needed.
#
# Note: backward functions for pooling and dropout (maxpool2d_backward,
# avgpool2d_backward, global_avgpool2d_backward, dropout_backward,
# dropout2d_backward) are intentionally NOT re-exported here.
# Importing shared.core.pooling or shared.core.dropout from shared.autograd
# causes extensor.mojo to be compiled twice with distinct type identities,
# producing "ExTensor cannot be converted from ExTensor" errors during
# `mojo package shared`. Import these directly from shared.core:
#   from shared.core.pooling import maxpool2d_backward
#   from shared.core.dropout import dropout_backward
