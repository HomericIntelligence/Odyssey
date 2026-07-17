"""
Optimizers.

Optimizer implementations for training neural networks

Includes:
- SGD (Stochastic Gradient Descent) with momentum
- Adam (Adaptive Moment Estimation)
- AdamW (Adam with Weight Decay)
- RMSprop (Root Mean Square Propagation)
- LARS (Layer-wise Adaptive Rate Scaling)
- Muon (Newton-Schulz Orthogonalized Momentum — Jordan et al. 2024)
- NorMuon (Muon with per-parameter normalization)
- MGUP-Muon (Muon with selective / max-utilization updates)
- Muon Hyperball (norm-constrained Muon — Frobenius-ball clamps on update and weights)
- Lion (EvoLved Sign Momentum — Chen et al. 2023)
- ADOPT (Modified Adam, optimal convergence for any beta2 — Taniguchi et al. 2024)
- Adan (Adaptive Nesterov Momentum — Xie et al. 2022)
- Sophia update step (clipped preconditioned step only, caller-supplied
  Hessian-diagonal estimates; Sophia-H/G estimators not included — Liu et al. 2023)
- FTRL-Proximal (online learning with L1 sparsity — McMahan et al. 2013)
- Shampoo (two-sided matrix preconditioner via Newton-Schulz inverse fourth root — Anil et al. 2020)
- SOAP (Shampoo + Adam in the preconditioner eigenbasis)

All optimizers follow pure functional design - caller manages state

Note:
    All symbols in this module are re-exported cleanly through the parent
    `odyssey.training` package. You may import directly from either location:

    ```mojo
    from odyssey.training.optimizers import sgd_step
    from odyssey.training import sgd_step  # also works
    ```

    No Mojo re-export limitation applies here (unlike `odyssey.training.callbacks`).
"""

# Export optimizer implementations

# SGD optimizer (functional implementation and in-place mutation)
from odyssey.training.optimizers.sgd import (
    sgd_step,
    sgd_step_simple,
    sgd_momentum_update_inplace,
    initialize_velocities,
    initialize_velocities_from_params,
)

# Adam optimizer (functional implementation)
from odyssey.training.optimizers.adam import adam_step, adam_step_simple

# AdamW optimizer (functional implementation with decoupled weight decay)
from odyssey.training.optimizers.adamw import (
    adamw_step,
    adamw_step_simple,
)

# RMSprop optimizer (functional implementation)
from odyssey.training.optimizers.rmsprop import (
    rmsprop_step,
    rmsprop_step_simple,
)

# LARS optimizer (Layer-wise Adaptive Rate Scaling)
from odyssey.training.optimizers.lars import lars_step, lars_step_simple

# Muon optimizer (Newton-Schulz orthogonalized momentum for matrix-shaped parameters)
from odyssey.training.optimizers.muon import (
    muon_step,
    muon_step_simple,
    newton_schulz_orthogonalize,
    is_muon_eligible,
)

# NorMuon optimizer (Muon with per-parameter normalization)
from odyssey.training.optimizers.normuon import (
    normuon_step,
    normuon_step_simple,
)

# MGUP-Muon optimizer (Muon with selective / max-utilization updates)
from odyssey.training.optimizers.mgup_muon import (
    mgup_muon_step,
    mgup_muon_step_simple,
)

# Muon Hyperball optimizer (norm-constrained Muon)
from odyssey.training.optimizers.muon_hyperball import (
    muon_hyperball_step,
    muon_hyperball_step_simple,
)

# Lion optimizer (EvoLved Sign Momentum — Chen et al. 2023)
from odyssey.training.optimizers.lion import lion_step, lion_step_simple

# ADOPT optimizer (modified Adam with the optimal convergence rate for any beta2)
from odyssey.training.optimizers.adopt import adopt_step, adopt_step_simple

# LionMuon optimizer (alternating Lion / Muon, separate per-rule buffers)
from odyssey.training.optimizers.lionmuon import (
    lionmuon_step,
    lionmuon_step_simple,
)

# Sophia clipped-preconditioned update step (caller-supplied Hessian estimates
# — Liu et al. 2023)
from odyssey.training.optimizers.sophia import (
    sophia_step,
    sophia_step_simple,
    sophia_update_hessian_moment,
)

# Adan optimizer (adaptive Nesterov momentum — Xie et al. 2022)
from odyssey.training.optimizers.adan import adan_step, adan_step_simple

# FTRL-Proximal optimizer (online learning with L1 sparsity — McMahan et al. 2013)
from odyssey.training.optimizers.ftrl import ftrl_step, ftrl_step_simple

# Shampoo optimizer (two-sided matrix preconditioner via Newton-Schulz inverse fourth root)
from odyssey.training.optimizers.shampoo import (
    shampoo_step,
    shampoo_step_simple,
    newton_schulz_inv_fourth_root,
    is_shampoo_eligible,
    initialize_shampoo_state,
)

# SOAP optimizer (Shampoo + Adam in the preconditioner eigenbasis)
from odyssey.training.optimizers.soap import (
    soap_step,
    init_soap_state,
)

# Optimizer utilities (common helper functions)
from odyssey.training.optimizers.optimizer_utils import (
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
