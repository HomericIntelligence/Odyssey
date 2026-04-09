"""Example: Custom Layers - Multi-Head Attention

This example implements multi-head self-attention mechanism.

Usage:
    pixi run mojo run examples/custom_layers/attention_layer.mojo

See documentation: docs/advanced/custom-layers.md
"""

from shared.core.module import Module
from shared.core.layers import Linear
from shared.tensor.any_tensor import AnyTensor, randn
from shared.core.activation import softmax
from shared.core.matrix import matmul, transpose


struct MultiHeadAttention(Module):
    """Multi-head self-attention mechanism.

    Used in Transformers and other attention-based models.
    """

    var num_heads: Int
    var head_dim: Int
    var embed_dim: Int

    var q_proj: Linear  # Query projection
    var k_proj: Linear  # Key projection
    var v_proj: Linear  # Value projection
    var out_proj: Linear  # Output projection

    def __init__(out self, embed_dim: Int, num_heads: Int) raises:
        """Initialize multi-head attention.

        Args:
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
        """
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Projection layers
        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim)

    def forward(mut self, x: AnyTensor) raises -> AnyTensor:
        """Compute multi-head attention.

        Args:
            x: Input tensor, shape [batch, seq_len, embed_dim].

        Returns:
            Output tensor, shape [batch, seq_len, embed_dim].
        """
        var batch_size = x.shape()[0]
        var seq_len = x.shape()[1]

        # Project to Q, K, V
        var q = self.q_proj.forward(x)
        var k = self.k_proj.forward(x)
        var v = self.v_proj.forward(x)

        # Reshape for multi-head attention
        # [batch, seq_len, embed_dim] -> [batch, num_heads, seq_len, head_dim]
        q = q.reshape([batch_size, seq_len, self.num_heads, self.head_dim])
        var axes_0213 = List[Int]()
        axes_0213.append(0)
        axes_0213.append(2)
        axes_0213.append(1)
        axes_0213.append(3)
        q = transpose(q, axes_0213.copy())
        k = k.reshape([batch_size, seq_len, self.num_heads, self.head_dim])
        k = transpose(k, axes_0213.copy())
        v = v.reshape([batch_size, seq_len, self.num_heads, self.head_dim])
        v = transpose(v, axes_0213.copy())

        # Scaled dot-product attention
        # Transpose k: swap last two dimensions [batch, num_heads, seq_len, head_dim] -> [batch, num_heads, head_dim, seq_len]
        var axes_0132 = List[Int]()
        axes_0132.append(0)
        axes_0132.append(1)
        axes_0132.append(3)
        axes_0132.append(2)
        var k_transposed = transpose(k, axes_0132^)
        # Note: Scaling by sqrt(head_dim) would require element-wise division
        # For this example, we skip the scaling step (simplified version)
        var scores = q @ k_transposed
        var attn_weights = softmax(scores, axis=-1)
        var attn_output = attn_weights @ v

        # Reshape back
        # [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, embed_dim]
        # First transpose: [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, num_heads, head_dim]
        attn_output = transpose(attn_output, axes_0213^)
        attn_output = attn_output.reshape([batch_size, seq_len, self.embed_dim])

        # Output projection
        return self.out_proj.forward(attn_output)

    def parameters(self) raises -> List[AnyTensor]:
        var params = List[AnyTensor]()
        params.extend(self.q_proj.parameters())
        params.extend(self.k_proj.parameters())
        params.extend(self.v_proj.parameters())
        params.extend(self.out_proj.parameters())
        return params^

    def train(mut self):
        """Switch to training mode (no-op for this layer)."""
        pass

    def eval(mut self):
        """Switch to evaluation mode (no-op for this layer)."""
        pass


def main() raises:
    """Demonstrate multi-head attention."""

    # Create attention layer
    var attention = MultiHeadAttention(embed_dim=512, num_heads=8)

    # Test with sample input
    var input = randn(
        [32, 100, 512], DType.float32
    )  # [batch, seq_len, embed_dim]
    print("Input shape:", input.shape())

    var output = attention.forward(input)
    print("Output shape:", output.shape())
    print("Expected: same as input [32, 100, 512]")

    # Show parameter count
    var params = attention.parameters()
    print("Number of parameter tensors:", len(params))

    print("\nMulti-head attention example complete!")
