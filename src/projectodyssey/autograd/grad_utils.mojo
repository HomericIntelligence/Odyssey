"""Gradient clipping utilities for preventing exploding gradients.

Re-exports from projectodyssey.core.grad_utils to maintain backward compatibility.

Note:
    The implementation was moved to projectodyssey.core.grad_utils to avoid a circular
    type resolution issue in Mojo v0.26.1: importing projectodyssey.core.any_tensor from
    within projectodyssey.autograd causes any_tensor.mojo to be compiled twice with
    distinct type identities, breaking AnyTensor operator overloads during
    `mojo package shared`.

    Import directly from projectodyssey.core.grad_utils or projectodyssey.autograd — both work:
        from projectodyssey.core.grad_utils import clip_grad_value_
        from projectodyssey.autograd import clip_grad_value_

References:
    - On the difficulty of training Recurrent Neural Networks (Pascanu et al., 2013)
      https://arxiv.org/abs/1211.1541
    - Issue #4513: AnyTensor circular type resolution
"""

from projectodyssey.core.grad_utils import (
    clip_grad_value_,
    clip_grad_norm_,
    clip_grad_global_norm_,
)
