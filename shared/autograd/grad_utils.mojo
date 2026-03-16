"""Gradient clipping utilities for preventing exploding gradients.

Re-exports from shared.core.grad_utils to maintain backward compatibility.

Note:
    The implementation was moved to shared.core.grad_utils to avoid a circular
    type resolution issue in Mojo v0.26.1: importing shared.core.extensor from
    within shared.autograd causes extensor.mojo to be compiled twice with
    distinct type identities, breaking ExTensor operator overloads during
    `mojo package shared`.

    Import directly from shared.core.grad_utils or shared.autograd — both work:
        from shared.core.grad_utils import clip_grad_value_
        from shared.autograd import clip_grad_value_

References:
    - On the difficulty of training Recurrent Neural Networks (Pascanu et al., 2013)
      https://arxiv.org/abs/1211.1541
    - ADR-009: Heap Corruption Workaround
    - Issue #4513: ExTensor circular type resolution
"""

from shared.core.grad_utils import (
    clip_grad_value_,
    clip_grad_norm_,
    clip_grad_global_norm_,
)
