"""
Optimizers.

Optimizer implementations for training neural networks

Includes:
- SGD (Stochastic Gradient Descent) with momentum
- Adam (Adaptive Moment Estimation)
- AdamW (Adam with Weight Decay)
- RMSprop (Root Mean Square Propagation)
- LARS (Layer-wise Adaptive Rate Scaling)

All optimizers follow pure functional design - caller manages state

Note:
    All symbols in this module are re-exported cleanly through the parent
    `projectodyssey.training` package. You may import directly from either location:

    ```mojo
    from projectodyssey.training.optimizers import sgd_step
    from projectodyssey.training import sgd_step  # also works
    ```

    No Mojo re-export limitation applies here (unlike `projectodyssey.training.callbacks`).
"""

# Export optimizer implementations

# SGD optimizer (functional implementation and in-place mutation)
from projectodyssey.training.optimizers.sgd import (
    sgd_step,
    sgd_step_simple,
    sgd_momentum_update_inplace,
    initialize_velocities,
    initialize_velocities_from_params,
)

# Adam optimizer (functional implementation)
from projectodyssey.training.optimizers.adam import adam_step, adam_step_simple

# AdamW optimizer (functional implementation with decoupled weight decay)
from projectodyssey.training.optimizers.adamw import (
    adamw_step,
    adamw_step_simple,
)

# RMSprop optimizer (functional implementation)
from projectodyssey.training.optimizers.rmsprop import (
    rmsprop_step,
    rmsprop_step_simple,
)

# LARS optimizer (Layer-wise Adaptive Rate Scaling)
from projectodyssey.training.optimizers.lars import lars_step, lars_step_simple

# Muon optimizer (Newton-Schulz orthogonalized momentum for matrix-shaped parameters)
from projectodyssey.training.optimizers.muon import (
    muon_step,
    muon_step_simple,
    newton_schulz_orthogonalize,
    is_muon_eligible,
)

# Optimizer utilities (common helper functions)
from projectodyssey.training.optimizers.optimizer_utils import (
    initialize_optimizer_state,
    initialize_optimizer_state_from_params,
    compute_weight_decay_term,
    apply_weight_decay,
    scale_tensor,
    scale_tensor_inplace,
    compute_tensor_norm,
    compute_global_norm,
    normalize_tensor_to_unit_norm,
    clip_tensor_norm,
    clip_global_norm,
    apply_bias_correction,
    validate_optimizer_state,
)
