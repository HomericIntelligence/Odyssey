"""
Neural Network Layers Module.

This module contains fundamental neural network layer implementations used across
paper reproductions. All layers are implemented in Mojo for maximum performance.

Components:
    - Linear: Fully connected (dense) layers
    - Conv2D: 2D convolutional layers
    - ReLU: Rectified Linear Unit activation
    - RNNCell: Vanilla (Elman) recurrent cell (tanh)
    - FeedForward: Transformer position-wise feed-forward (Linear->act->Linear)
    - KAN: 1-layer Kolmogorov-Arnold Network block (per-edge B-spline activations)
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
from odyssey.core.layers.lstm import LSTMCell
from odyssey.core.layers.layernorm import LayerNorm
from odyssey.core.layers.gru import GRUCell
from odyssey.core.layers.feedforward import FeedForward
from odyssey.core.layers.kan import KAN

# from .activation import ReLU, Sigmoid, Tanh
# from .pooling import MaxPool2D, AvgPool2D
