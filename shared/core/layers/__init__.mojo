"""
Neural Network Layers Module.

This module contains fundamental neural network layer implementations used across
paper reproductions. All layers are implemented in Mojo for maximum performance.

Components:
    - Linear: Fully connected (dense) layers
    - Conv2D: 2D convolutional layers
    - ReLU: Rectified Linear Unit activation
    - Sigmoid: Sigmoid activation function
    - Tanh: Hyperbolic tangent activation
    - BatchNorm: Batch normalization
    - LayerNorm: Layer normalization
    - MaxPool2D: 2D max pooling
    - AvgPool2D: 2D average pooling

Example:
    ```mojo
    from shared.core.layers import Linear, ReLU

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
from shared.core.layers.linear import Linear
from shared.core.layers.conv2d import Conv2dLayer
from shared.core.layers.batchnorm import BatchNorm2dLayer
from shared.core.layers.relu import ReLULayer
from shared.core.layers.dropout import DropoutLayer

# from .activation import ReLU, Sigmoid, Tanh
# from .pooling import MaxPool2D, AvgPool2D
