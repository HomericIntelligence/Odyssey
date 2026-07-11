"""Gradient clipping utilities for preventing exploding gradients.

Re-exports from odyssey.core.grad_utils to maintain backward compatibility.

Note:
    The implementation was moved to odyssey.core.grad_utils to avoid a circular
    type resolution issue in Mojo v0.26.1: importing odyssey.core.any_tensor from
    within odyssey.autograd causes any_tensor.mojo to be compiled twice with
    distinct type identities, breaking AnyTensor operator overloads during
    `mojo package shared`.

    Import directly from odyssey.core.grad_utils or odyssey.autograd — both work:
        from odyssey.core.grad_utils import clip_grad_value_
        from odyssey.autograd import clip_grad_value_

References:
    - On the difficulty of training Recurrent Neural Networks (Pascanu et al., 2013)
      https://arxiv.org/abs/1211.1541
    - Issue #4513: AnyTensor circular type resolution
"""

from odyssey.core.grad_utils import (
    clip_grad_value_,
    clip_grad_norm_,
    clip_grad_global_norm_,
)
