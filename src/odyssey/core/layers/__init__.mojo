"""
Neural Network Layers Module.

This module contains fundamental neural network layer implementations used across
paper reproductions. All layers are implemented in Mojo for maximum performance.

Components:
    - Linear: Fully connected (dense) layers
    - Conv2D: 2D convolutional layers
    - ReLU: Rectified Linear Unit activation
    - RNNCell: Vanilla (Elman) recurrent cell (tanh)
    - LTCCell: Liquid Time-constant recurrent cell (fused ODE solver)
    - FeedForward: Transformer position-wise feed-forward (Linear->act->Linear)
    - MultiHeadAttention: scaled dot-product self-attention block (Vaswani 2017)
    - SparseAttention: strided factorized sparse self-attention (Child et al. 2019)
    - DiagonalSSM: Diagonal state-space (S4-style) sequence block
    - TransformerEncoderBlock: pre-LN Transformer block (attention + FFN)
    - LinearAttention: Linear (kernel-feature) self-attention (arXiv:2006.16236)
    - MambaBlock: Selective state-space (S6) block (input-dependent B/C/Delta)
    - MLPMixerBlock: 1-layer MLP-Mixer block (token-mixing + channel-mixing MLPs)
    - KAN: 1-layer Kolmogorov-Arnold Network block (per-edge B-spline activations)
    - DeepSetsEquivariant: Permutation-equivariant Deep Sets linear block
    - Sigmoid: Sigmoid activation function
    - Tanh: Hyperbolic tangent activation
    - BatchNorm: Batch normalization
    - LayerNorm: Layer normalization
    - MaxPool2D: 2D max pooling
    - AvgPool2D: 2D average pooling

Example:
    ```mojo
    from odyssey.core.layers import Linear, ReLU

    struct MLP:
        var fc1: Linear
        var relu: ReLU
        var fc2: Linear

        def __init__(out self):
            self.fc1 = Linear(784, 128)
            self.relu = ReLU()
            self.fc2 = Linear(128, 10)
    ```
"""

# Layer exports
from odyssey.core.layers.linear import Linear
from odyssey.core.layers.conv2d import Conv2dLayer
from odyssey.core.layers.batchnorm import BatchNorm2dLayer
from odyssey.core.layers.relu import ReLULayer
from odyssey.core.layers.dropout import DropoutLayer
from odyssey.core.layers.rnn import RNNCell
from odyssey.core.layers.ltc import LTCCell
from odyssey.core.layers.lstm import LSTMCell
from odyssey.core.layers.layernorm import LayerNorm
from odyssey.core.layers.gru import GRUCell
from odyssey.core.layers.ssm import DiagonalSSM
from odyssey.core.layers.feedforward import FeedForward
from odyssey.core.layers.attention import MultiHeadAttention
from odyssey.core.layers.sparse_attention import SparseAttention
from odyssey.core.layers.transformer import TransformerEncoderBlock
from odyssey.core.layers.linear_attention import LinearAttention
from odyssey.core.layers.mamba import MambaBlock
from odyssey.core.layers.mlp_mixer import MLPMixerBlock
from odyssey.core.layers.kan import KAN
from odyssey.core.layers.deepsets import DeepSetsEquivariant

# from .activation import ReLU, Sigmoid, Tanh
# from .pooling import MaxPool2D, AvgPool2D
