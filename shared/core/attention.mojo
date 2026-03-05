"""Attention mechanisms for transformer architectures.

This module provides pure functional implementations of attention operations.
All operations are stateless - caller manages parameters and state.

The core operation is scaled dot-product attention:
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

Where:
    Q: Query tensor of shape (batch, seq_len, d_k).
    K: Key tensor of shape (batch, seq_len, d_k).
    V: Value tensor of shape (batch, seq_len, d_v).
    d_k: Key dimension (used for scaling).
"""

from shared.core.extensor import ExTensor, zeros, zeros_like, ones
from shared.core.matrix import matmul, transpose
from shared.core.activation import softmax
from shared.core.arithmetic import multiply, divide, add
from shared.core.gradient_types import GradientTriple, GradientQuad
from math import sqrt


fn scaled_dot_product_attention(
    query: ExTensor,
    key: ExTensor,
    value: ExTensor,
    dropout_p: Float64 = 0.0,
) raises -> ExTensor:
    """Scaled dot-product attention without mask.

    Raises:
            Error: If operation fails.

    See scaled_dot_product_attention_masked for version with mask support.
    """
    var empty_shape = List[Int]()
    var empty_mask = zeros(empty_shape, DType.float32)
    return scaled_dot_product_attention_masked(
        query, key, value, empty_mask, dropout_p
    )


fn scaled_dot_product_attention_masked(
    query: ExTensor,
    key: ExTensor,
    value: ExTensor,
    mask: ExTensor,
    dropout_p: Float64 = 0.0,
) raises -> ExTensor:
    """Scaled dot-product attention.

        Computes attention weights from query-key similarity and applies them to values.
        This is the fundamental building block of transformer architectures.

    Args:
            query: Query tensor of shape (batch, seq_len, d_k) or (batch, heads, seq_len, d_k).
            key: Key tensor of shape (batch, seq_len, d_k) or (batch, heads, seq_len, d_k).
            value: Value tensor of shape (batch, seq_len, d_v) or (batch, heads, seq_len, d_v).
            mask: Optional attention mask. Use large negative values (-1e9) for positions
                   to ignore. Shape: (batch, seq_len, seq_len) or (batch, heads, seq_len, seq_len).
            dropout_p: Dropout probability (not applied in this implementation).

    Returns:
            Attention output of shape (batch, seq_len, d_v) or (batch, heads, seq_len, d_v).

    Raises:
            Error: If operation fails.

        Example:
            ```mojo
            from shared.core import scaled_dot_product_attention

            # Create Q, K, V tensors: (batch=2, seq_len=10, d_k=64)
            var query = ones([2, 10, 64], DType.float32)
            var key = ones([2, 10, 64], DType.float32)
            var value = ones([2, 10, 64], DType.float32)

            # Compute attention
            var output = scaled_dot_product_attention(query, key, value)
            # output shape: (2, 10, 64)
            ```

        Formula:
            Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V.

    Note:
            - The scaling factor sqrt(d_k) prevents dot products from growing too large.
            - Masking is applied before softmax (additive masking).
            - For causal (autoregressive) attention, use a lower-triangular mask.
    """
    var q_shape = query.shape()
    var k_shape = key.shape()
    var v_shape = value.shape()

    # Support both 3D (batch, seq, dim) and 4D (batch, heads, seq, dim) inputs
    var ndim = len(q_shape)
    if ndim != 3 and ndim != 4:
        raise Error("Error: query must be 3D or 4D tensor.")

    # Get d_k for scaling (last dimension of query/key)
    var d_k = q_shape[ndim - 1]
    var scale = Float64(1.0) / sqrt(Float64(d_k))

    # Compute QK^T
    # For 3D: (batch, seq_q, d_k) @ (batch, d_k, seq_k) -> (batch, seq_q, seq_k)
    # For 4D: (batch, heads, seq_q, d_k) @ (batch, heads, d_k, seq_k) -> (batch, heads, seq_q, seq_k)
    var key_t = transpose(key)
    var scores = matmul(query, key_t)

    # Scale scores
    var scale_tensor = zeros_like(scores)
    var scale_ptr = scale_tensor._data
    var numel = scores.numel()

    if scores.dtype() == DType.float32:
        var scale_f32 = Float32(scale)
        for i in range(numel):
            scale_ptr.bitcast[Float32]()[i] = scale_f32
    elif scores.dtype() == DType.float64:
        for i in range(numel):
            scale_ptr.bitcast[Float64]()[i] = scale
    else:
        raise Error("Error: only float32/64 supported.")

    var scaled_scores = multiply(scores, scale_tensor)

    # Apply mask if provided (additive masking)
    var mask_shape = mask.shape()
    if len(mask_shape) > 0 and mask.numel() > 0:
        scaled_scores = add(scaled_scores, mask)

    # Apply softmax along the last dimension (key sequence length)
    var attention_weights = softmax(scaled_scores)

    # Apply attention weights to values
    # (batch, seq_q, seq_k) @ (batch, seq_k, d_v) -> (batch, seq_q, d_v)
    var output = matmul(attention_weights, value)

    return output


# Type comptime for scaled dot-product attention backward results
# Maps to GradientTriple with field name mapping:
#   grad_input  -> grad_query  (query gradient)
#   grad_weights -> grad_key    (key gradient)
#   grad_bias  -> grad_value   (value gradient)
comptime ScaledDotProductAttentionBackwardResult = GradientTriple


fn scaled_dot_product_attention_backward(
    grad_output: ExTensor,
    query: ExTensor,
    key: ExTensor,
    value: ExTensor,
    attention_weights: ExTensor,
) raises -> GradientTriple:
    """Backward pass for scaled dot-product attention without mask.

    Raises:
            Error: If operation fails.
    """
    var empty_shape = List[Int]()
    var empty_mask = zeros(empty_shape, DType.float32)
    return scaled_dot_product_attention_backward_masked(
        grad_output, query, key, value, attention_weights, empty_mask
    )


fn scaled_dot_product_attention_backward_masked(
    grad_output: ExTensor,
    query: ExTensor,
    key: ExTensor,
    value: ExTensor,
    attention_weights: ExTensor,
    mask: ExTensor,
) raises -> GradientTriple:
    """Backward pass for scaled dot-product attention.

        Computes gradients with respect to query, key, and value tensors.

    Args:
            grad_output: Gradient w.r.t. attention output.
            query: Original query tensor.
            key: Original key tensor.
            value: Original value tensor.
            attention_weights: Attention weights from forward pass (after softmax).
            mask: Optional attention mask (same as forward pass).

    Returns:
            GradientTriple containing gradients for query, key, and value.
            Field mapping for backward results:
                - grad_input  -> gradient w.r.t. query.
                - grad_weights -> gradient w.r.t. key.
                - grad_bias  -> gradient w.r.t. value.

    Raises:
            Error: If operation fails.

        Example:
            ```mojo
            from shared.core import (
                scaled_dot_product_attention,
                scaled_dot_product_attention_backward,
            )

            # Forward pass
            var output = scaled_dot_product_attention(query, key, value)

            # ... compute loss and grad_output ...

            # Backward pass
            var result = scaled_dot_product_attention_backward(
                grad_output, query, key, value, attention_weights
            )
            # result.grad_input, result.grad_weights, result.grad_bias
            ```

    Note:
            Caller must save attention_weights from forward pass for use in backward.
            Pure functional: returns new tensors, does not modify inputs.
    """
    var q_shape = query.shape()
    var ndim = len(q_shape)
    var d_k = q_shape[ndim - 1]
    var scale = Float64(1.0) / sqrt(Float64(d_k))

    # Gradient w.r.t. value: attention_weights^T @ grad_output
    var attention_weights_t = transpose(attention_weights)
    var grad_value = matmul(attention_weights_t, grad_output)

    # Gradient w.r.t. attention weights: grad_output @ value^T
    var value_t = transpose(value)
    var grad_attention = matmul(grad_output, value_t)

    # Gradient through softmax
    # d_softmax/d_input = softmax * (grad - sum(grad * softmax))
    var grad_softmax = _softmax_backward(grad_attention, attention_weights)

    # Scale the gradient
    var scale_tensor = zeros_like(grad_softmax)
    var scale_ptr = scale_tensor._data
    var numel = grad_softmax.numel()

    if grad_softmax.dtype() == DType.float32:
        var scale_f32 = Float32(scale)
        for i in range(numel):
            scale_ptr.bitcast[Float32]()[i] = scale_f32
    elif grad_softmax.dtype() == DType.float64:
        for i in range(numel):
            scale_ptr.bitcast[Float64]()[i] = scale
    else:
        raise Error("Error: only float32/64 supported.")

    var grad_scores = multiply(grad_softmax, scale_tensor)

    # Gradient w.r.t. query: grad_scores @ key
    var grad_query = matmul(grad_scores, key)

    # Gradient w.r.t. key: grad_scores^T @ query
    var grad_scores_t = transpose(grad_scores)
    var grad_key = matmul(grad_scores_t, query)

    return GradientTriple(grad_query, grad_key, grad_value)


fn _softmax_backward(
    grad_output: ExTensor, softmax_output: ExTensor
) raises -> ExTensor:
    """Internal helper for softmax backward pass.

    Computes gradient through softmax: d_softmax/d_input = s * (g - sum(g * s)).
    where s = softmax_output and g = grad_output.

    Raises:
            Error: If operation fails.
    """
    var shape = grad_output.shape()
    var result = zeros_like(grad_output)

    var grad_ptr = grad_output._data
    var softmax_ptr = softmax_output._data
    var result_ptr = result._data

    # Get dimensions for the last axis (softmax is applied along last axis)
    var ndim = len(shape)
    var last_dim = shape[ndim - 1]
    var batch_size = grad_output.numel() // last_dim

    if grad_output.dtype() == DType.float32:
        for b in range(batch_size):
            var offset = b * last_dim

            # Compute sum(grad * softmax) for this batch element
            var dot_sum = Float32(0.0)
            for i in range(last_dim):
                dot_sum += (
                    grad_ptr.bitcast[Float32]()[offset + i]
                    * softmax_ptr.bitcast[Float32]()[offset + i]
                )

            # Compute gradient: softmax * (grad - dot_sum)
            for i in range(last_dim):
                var s = softmax_ptr.bitcast[Float32]()[offset + i]
                var g = grad_ptr.bitcast[Float32]()[offset + i]
                result_ptr.bitcast[Float32]()[offset + i] = s * (g - dot_sum)

    elif grad_output.dtype() == DType.float64:
        for b in range(batch_size):
            var offset = b * last_dim

            var dot_sum = Float64(0.0)
            for i in range(last_dim):
                dot_sum += (
                    grad_ptr.bitcast[Float64]()[offset + i]
                    * softmax_ptr.bitcast[Float64]()[offset + i]
                )

            for i in range(last_dim):
                var s = softmax_ptr.bitcast[Float64]()[offset + i]
                var g = grad_ptr.bitcast[Float64]()[offset + i]
                result_ptr.bitcast[Float64]()[offset + i] = s * (g - dot_sum)

    else:
        raise Error("Error: only float32/64 supported.")

    return result


fn create_causal_mask(
    seq_len: Int, dtype: DType = DType.float32
) raises -> ExTensor:
    """Create a causal (lower-triangular) attention mask.

        Returns a mask where positions that should be ignored have large negative
        values (-1e9) and valid positions have 0.

    Args:
            seq_len: Sequence length for the mask.
            dtype: Data type for the mask tensor.

    Returns:
            Mask tensor of shape (seq_len, seq_len) suitable for attention.

    Raises:
            Error: If operation fails.

        Example:
            ```mojo
            from shared.core import create_causal_mask, scaled_dot_product_attention

            var mask = create_causal_mask(10)
            var output = scaled_dot_product_attention(query, key, value, mask=mask)
            ```

    Note:
            The mask is designed for additive masking before softmax.
            Position (i, j) is masked (set to -1e9) if j > i (future position).
    """
    var shape = List[Int]()
    shape.append(seq_len)
    shape.append(seq_len)

    var mask = zeros(shape, dtype)
    var mask_ptr = mask._data

    var neg_inf = Float64(-1e9)

    if dtype == DType.float32:
        var neg_inf_f32 = Float32(neg_inf)
        for i in range(seq_len):
            for j in range(seq_len):
                var idx = i * seq_len + j
                if j > i:
                    # Future position - mask it out
                    mask_ptr.bitcast[Float32]()[idx] = neg_inf_f32
                # else: keep as 0 (no masking)

    elif dtype == DType.float64:
        for i in range(seq_len):
            for j in range(seq_len):
                var idx = i * seq_len + j
                if j > i:
                    mask_ptr.bitcast[Float64]()[idx] = neg_inf

    else:
        raise Error("Error: only float32/64 supported.")

    return mask


# ============================================================================
# Multi-Head Attention
# ============================================================================


struct MultiHeadAttentionWeights(Movable):
    """Container for multi-head attention weight matrices.

    Holds the projection matrices for Q, K, V and output projection.
    """

    var wq: ExTensor
    """Query projection weight matrix of shape (d_model, d_model)."""
    var wk: ExTensor
    """Key projection weight matrix of shape (d_model, d_model)."""
    var wv: ExTensor
    """Value projection weight matrix of shape (d_model, d_model)."""
    var wo: ExTensor
    """Output projection weight matrix of shape (d_model, d_model)."""

    fn __init__(
        out self,
        wq: ExTensor,
        wk: ExTensor,
        wv: ExTensor,
        wo: ExTensor,
    ):
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

    fn __moveinit__(out self, deinit existing: Self):
        self.wq = existing.wq^
        self.wk = existing.wk^
        self.wv = existing.wv^
        self.wo = existing.wo^


struct MultiHeadAttentionResult(Movable):
    """Result container for multi_head_attention.

    Contains output and attention weights for visualization/analysis.
    """

    var output: ExTensor
    """Attention output of shape (batch, seq_len, d_model)."""
    var attention_weights: ExTensor
    """Attention weights for visualization and analysis."""

    fn __init__(out self, output: ExTensor, attention_weights: ExTensor):
        self.output = output
        self.attention_weights = attention_weights

    fn __moveinit__(out self, deinit existing: Self):
        self.output = existing.output^
        self.attention_weights = existing.attention_weights^


fn multi_head_attention(
    query: ExTensor,
    key: ExTensor,
    value: ExTensor,
    weights: MultiHeadAttentionWeights,
    num_heads: Int,
) raises -> MultiHeadAttentionResult:
    """Multi-head attention mechanism without mask.

    Raises:
            Error: If operation fails.

    See multi_head_attention_masked for version with mask support.
    """
    var empty_shape = List[Int]()
    var empty_mask = zeros(empty_shape, DType.float32)
    return multi_head_attention_masked(
        query, key, value, weights, num_heads, empty_mask
    )


fn multi_head_attention_masked(
    query: ExTensor,
    key: ExTensor,
    value: ExTensor,
    weights: MultiHeadAttentionWeights,
    num_heads: Int,
    mask: ExTensor,
) raises -> MultiHeadAttentionResult:
    """Multi-head attention mechanism.

        Projects inputs through multiple attention heads in parallel, then
        concatenates and projects the results. This is the core mechanism in
        transformer architectures.

    Args:
            query: Query tensor of shape (batch, seq_len, d_model).
            key: Key tensor of shape (batch, seq_len, d_model).
            value: Value tensor of shape (batch, seq_len, d_model).
            weights: MultiHeadAttentionWeights containing Wq, Wk, Wv, Wo.
            num_heads: Number of attention heads.
            mask: Optional attention mask.

    Returns:
            MultiHeadAttentionResult containing:
                - output: Attended output of shape (batch, seq_len, d_model).
                - attention_weights: Attention weights for visualization.

    Raises:
            Error: If operation fails.

        Example:
            ```mojo
            from shared.core import multi_head_attention, MultiHeadAttentionWeights

            # Initialize weights (normally from model)
            var weights = MultiHeadAttentionWeights(wq, wk, wv, wo)

            # Compute multi-head attention
            var result = multi_head_attention(
                query, key, value, weights, num_heads=8
            )
            var output = result.output
            ```

        Formula:
            MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * Wo.
            where head_i = Attention(Q * Wq_i, K * Wk_i, V * Wv_i).

    Note:
            - d_model must be divisible by num_heads.
            - Each head operates on d_k = d_model / num_heads dimensions.
    """
    var q_shape = query.shape()
    if len(q_shape) != 3:
        raise Error("Error: query must be 3D (batch, seq, d_model).")

    var batch = q_shape[0]
    var seq_len = q_shape[1]
    var d_model = q_shape[2]

    if d_model % num_heads != 0:
        raise Error("Error: d_model must be divisible by num_heads.")

    var d_k = d_model // num_heads

    # Project Q, K, V through weight matrices
    # (batch, seq, d_model) @ (d_model, d_model) -> (batch, seq, d_model)
    var q_proj = matmul(query, weights.wq)
    var k_proj = matmul(key, weights.wk)
    var v_proj = matmul(value, weights.wv)

    # Reshape to (batch, seq, num_heads, d_k) then transpose to (batch, num_heads, seq, d_k)
    var q_heads = _reshape_for_heads(q_proj, batch, seq_len, num_heads, d_k)
    var k_heads = _reshape_for_heads(k_proj, batch, seq_len, num_heads, d_k)
    var v_heads = _reshape_for_heads(v_proj, batch, seq_len, num_heads, d_k)

    # Apply scaled dot-product attention per head
    # NOTE: scaled_dot_product_attention handles 4D tensors (batch, heads, seq, d_k)
    var scale = Float64(1.0) / sqrt(Float64(d_k))

    # Compute attention scores: (batch, heads, seq, d_k) @ (batch, heads, d_k, seq)
    var k_heads_t = transpose(k_heads)
    var scores = matmul(q_heads, k_heads_t)

    # Scale scores
    var scale_tensor = zeros_like(scores)
    var scale_ptr = scale_tensor._data
    var numel = scores.numel()

    if scores.dtype() == DType.float32:
        var scale_f32 = Float32(scale)
        for i in range(numel):
            scale_ptr.bitcast[Float32]()[i] = scale_f32
    else:
        for i in range(numel):
            scale_ptr.bitcast[Float64]()[i] = scale

    var scaled_scores = multiply(scores, scale_tensor)

    # Apply mask if provided
    var mask_shape = mask.shape()
    if len(mask_shape) > 0 and mask.numel() > 0:
        scaled_scores = add(scaled_scores, mask)

    # Softmax over last dimension
    var attention_weights = softmax(scaled_scores)

    # Apply attention to values
    var attended = matmul(attention_weights, v_heads)

    # Reshape back: (batch, heads, seq, d_k) -> (batch, seq, d_model)
    var concat_heads = _reshape_from_heads(
        attended, batch, seq_len, num_heads, d_k
    )

    # Final output projection
    var output = matmul(concat_heads, weights.wo)

    return MultiHeadAttentionResult(output, attention_weights)


fn _reshape_for_heads(
    x: ExTensor, batch: Int, seq_len: Int, num_heads: Int, d_k: Int
) raises -> ExTensor:
    """Reshape from (batch, seq, d_model) to (batch, num_heads, seq, d_k).

    Internal helper for multi-head attention.

    Raises:
            Error: If operation fails.
    """
    # x shape: (batch, seq_len, d_model) where d_model = num_heads * d_k
    # Target: (batch, num_heads, seq_len, d_k)
    var d_model = num_heads * d_k

    var out_shape = List[Int]()
    out_shape.append(batch)
    out_shape.append(num_heads)
    out_shape.append(seq_len)
    out_shape.append(d_k)

    var result = zeros(out_shape, x.dtype())
    var x_ptr = x._data
    var result_ptr = result._data

    if x.dtype() == DType.float32:
        for b in range(batch):
            for s in range(seq_len):
                for h in range(num_heads):
                    for k in range(d_k):
                        # Source: (b, s, h * d_k + k)
                        var src_idx = (
                            b * (seq_len * d_model) + s * d_model + h * d_k + k
                        )
                        # Dest: (b, h, s, k)
                        var dst_idx = (
                            b * (num_heads * seq_len * d_k)
                            + h * (seq_len * d_k)
                            + s * d_k
                            + k
                        )
                        result_ptr.bitcast[Float32]()[dst_idx] = x_ptr.bitcast[
                            Float32
                        ]()[src_idx]
    else:
        for b in range(batch):
            for s in range(seq_len):
                for h in range(num_heads):
                    for k in range(d_k):
                        var src_idx = (
                            b * (seq_len * d_model) + s * d_model + h * d_k + k
                        )
                        var dst_idx = (
                            b * (num_heads * seq_len * d_k)
                            + h * (seq_len * d_k)
                            + s * d_k
                            + k
                        )
                        result_ptr.bitcast[Float64]()[dst_idx] = x_ptr.bitcast[
                            Float64
                        ]()[src_idx]

    return result


fn _reshape_from_heads(
    x: ExTensor, batch: Int, seq_len: Int, num_heads: Int, d_k: Int
) raises -> ExTensor:
    """Reshape from (batch, num_heads, seq, d_k) to (batch, seq, d_model).

    Internal helper for multi-head attention.

    Raises:
            Error: If operation fails.
    """
    # x shape: (batch, num_heads, seq_len, d_k)
    # Target: (batch, seq_len, d_model) where d_model = num_heads * d_k
    var d_model = num_heads * d_k

    var out_shape = List[Int]()
    out_shape.append(batch)
    out_shape.append(seq_len)
    out_shape.append(d_model)

    var result = zeros(out_shape, x.dtype())
    var x_ptr = x._data
    var result_ptr = result._data

    if x.dtype() == DType.float32:
        for b in range(batch):
            for s in range(seq_len):
                for h in range(num_heads):
                    for k in range(d_k):
                        # Source: (b, h, s, k)
                        var src_idx = (
                            b * (num_heads * seq_len * d_k)
                            + h * (seq_len * d_k)
                            + s * d_k
                            + k
                        )
                        # Dest: (b, s, h * d_k + k)
                        var dst_idx = (
                            b * (seq_len * d_model) + s * d_model + h * d_k + k
                        )
                        result_ptr.bitcast[Float32]()[dst_idx] = x_ptr.bitcast[
                            Float32
                        ]()[src_idx]
    else:
        for b in range(batch):
            for s in range(seq_len):
                for h in range(num_heads):
                    for k in range(d_k):
                        var src_idx = (
                            b * (num_heads * seq_len * d_k)
                            + h * (seq_len * d_k)
                            + s * d_k
                            + k
                        )
                        var dst_idx = (
                            b * (seq_len * d_model) + s * d_model + h * d_k + k
                        )
                        result_ptr.bitcast[Float64]()[dst_idx] = x_ptr.bitcast[
                            Float64
                        ]()[src_idx]

    return result


struct MultiHeadAttentionBackwardResult(Movable):
    """Result container for multi_head_attention_backward.

    Contains gradients for all inputs and weight matrices.
    """

    var grad_query: ExTensor
    """Gradient with respect to query tensor."""
    var grad_key: ExTensor
    """Gradient with respect to key tensor."""
    var grad_value: ExTensor
    """Gradient with respect to value tensor."""
    var grad_wq: ExTensor
    """Gradient with respect to query projection weight matrix."""
    var grad_wk: ExTensor
    """Gradient with respect to key projection weight matrix."""
    var grad_wv: ExTensor
    """Gradient with respect to value projection weight matrix."""
    var grad_wo: ExTensor
    """Gradient with respect to output projection weight matrix."""

    fn __init__(
        out self,
        grad_query: ExTensor,
        grad_key: ExTensor,
        grad_value: ExTensor,
        grad_wq: ExTensor,
        grad_wk: ExTensor,
        grad_wv: ExTensor,
        grad_wo: ExTensor,
    ):
        self.grad_query = grad_query
        self.grad_key = grad_key
        self.grad_value = grad_value
        self.grad_wq = grad_wq
        self.grad_wk = grad_wk
        self.grad_wv = grad_wv
        self.grad_wo = grad_wo

    fn __moveinit__(out self, deinit existing: Self):
        self.grad_query = existing.grad_query^
        self.grad_key = existing.grad_key^
        self.grad_value = existing.grad_value^
        self.grad_wq = existing.grad_wq^
        self.grad_wk = existing.grad_wk^
        self.grad_wv = existing.grad_wv^
        self.grad_wo = existing.grad_wo^


fn multi_head_attention_backward(
    grad_output: ExTensor,
    query: ExTensor,
    key: ExTensor,
    value: ExTensor,
    weights: MultiHeadAttentionWeights,
    attention_weights: ExTensor,
    num_heads: Int,
) raises -> MultiHeadAttentionBackwardResult:
    """Backward pass for multi-head attention.

        Computes gradients with respect to all inputs and weight matrices.

    Args:
            grad_output: Gradient w.r.t. output (batch, seq_len, d_model).
            query: Original query tensor (batch, seq_len, d_model).
            key: Original key tensor (batch, seq_len, d_model).
            value: Original value tensor (batch, seq_len, d_model).
            weights: MultiHeadAttentionWeights used in forward pass.
            attention_weights: Attention weights from forward pass.
            num_heads: Number of attention heads.

    Returns:
            MultiHeadAttentionBackwardResult containing gradients for all inputs/weights.

    Raises:
            Error: If operation fails.

    Note:
            Caller must save attention_weights from forward pass.
            Pure functional: returns new tensors, does not modify inputs.
    """
    var q_shape = query.shape()
    var batch = q_shape[0]
    var seq_len = q_shape[1]
    var d_model = q_shape[2]
    var d_k = d_model // num_heads

    # Recompute projections for backward
    var q_proj = matmul(query, weights.wq)
    var k_proj = matmul(key, weights.wk)
    var v_proj = matmul(value, weights.wv)

    var q_heads = _reshape_for_heads(q_proj, batch, seq_len, num_heads, d_k)
    var k_heads = _reshape_for_heads(k_proj, batch, seq_len, num_heads, d_k)
    var v_heads = _reshape_for_heads(v_proj, batch, seq_len, num_heads, d_k)

    # Gradient through output projection: output = concat_heads @ Wo
    # grad_concat_heads = grad_output @ Wo^T
    var wo_t = transpose(weights.wo)
    var grad_concat = matmul(grad_output, wo_t)

    # grad_Wo = concat_heads^T @ grad_output
    var attended = matmul(attention_weights, v_heads)
    var concat_heads = _reshape_from_heads(
        attended, batch, seq_len, num_heads, d_k
    )
    var concat_heads_t = transpose(concat_heads)
    var grad_wo = matmul(concat_heads_t, grad_output)

    # Reshape gradient for heads
    var grad_attended = _reshape_for_heads(
        grad_concat, batch, seq_len, num_heads, d_k
    )

    # Gradient through attention: attended = attention_weights @ v_heads
    # grad_v_heads = attention_weights^T @ grad_attended
    var attention_weights_t = transpose(attention_weights)
    var grad_v_heads = matmul(attention_weights_t, grad_attended)

    # grad_attention_weights = grad_attended @ v_heads^T
    var v_heads_t = transpose(v_heads)
    var grad_attn_weights = matmul(grad_attended, v_heads_t)

    # Gradient through softmax
    var grad_scaled_scores = _softmax_backward(
        grad_attn_weights, attention_weights
    )

    # Gradient through scaling
    var scale = Float64(1.0) / sqrt(Float64(d_k))
    var scale_tensor = zeros_like(grad_scaled_scores)
    var scale_ptr = scale_tensor._data
    var numel = grad_scaled_scores.numel()

    if grad_scaled_scores.dtype() == DType.float32:
        var scale_f32 = Float32(scale)
        for i in range(numel):
            scale_ptr.bitcast[Float32]()[i] = scale_f32
    else:
        for i in range(numel):
            scale_ptr.bitcast[Float64]()[i] = scale

    var grad_scores = multiply(grad_scaled_scores, scale_tensor)

    # Gradient through matmul: scores = q_heads @ k_heads^T
    # grad_q_heads = grad_scores @ k_heads
    var grad_q_heads = matmul(grad_scores, k_heads)

    # grad_k_heads = grad_scores^T @ q_heads
    var grad_scores_t = transpose(grad_scores)
    var grad_k_heads = matmul(grad_scores_t, q_heads)

    # Reshape gradients back to (batch, seq, d_model)
    var grad_q_proj = _reshape_from_heads(
        grad_q_heads, batch, seq_len, num_heads, d_k
    )
    var grad_k_proj = _reshape_from_heads(
        grad_k_heads, batch, seq_len, num_heads, d_k
    )
    var grad_v_proj = _reshape_from_heads(
        grad_v_heads, batch, seq_len, num_heads, d_k
    )

    # Gradient through input projections
    # grad_query = grad_q_proj @ Wq^T
    var wq_t = transpose(weights.wq)
    var grad_query = matmul(grad_q_proj, wq_t)

    var wk_t = transpose(weights.wk)
    var grad_key = matmul(grad_k_proj, wk_t)

    var wv_t = transpose(weights.wv)
    var grad_value = matmul(grad_v_proj, wv_t)

    # Gradient w.r.t. weight matrices
    # grad_Wq = query^T @ grad_q_proj
    var query_t = transpose(query)
    var grad_wq = matmul(query_t, grad_q_proj)

    var key_t = transpose(key)
    var grad_wk = matmul(key_t, grad_k_proj)

    var value_t = transpose(value)
    var grad_wv = matmul(value_t, grad_v_proj)

    return MultiHeadAttentionBackwardResult(
        grad_query, grad_key, grad_value, grad_wq, grad_wk, grad_wv, grad_wo
    )
