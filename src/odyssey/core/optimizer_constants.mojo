"""Default optimizer hyperparameters following PyTorch and TensorFlow conventions.

This module provides centralized default values for optimizer hyperparameters,
including learning rates, momentum, and beta coefficients.

Constants:
    DEFAULT_LEARNING_RATE_SGD: Default learning rate for SGD (0.01)
    DEFAULT_LEARNING_RATE_ADAM: Default learning rate for Adam (0.001)
    DEFAULT_MOMENTUM: Default momentum for SGD and related optimizers (0.9)
    DEFAULT_ADAM_BETA1: Default exponential decay rate for first moment (0.9)
    DEFAULT_ADAM_BETA2: Default exponential decay rate for second moment (0.999)
    DEFAULT_ADAM_EPSILON: Default epsilon for Adam optimizer (1e-8)
    DEFAULT_RMSPROP_ALPHA: Default decay rate for RMSprop (0.99)
    DEFAULT_RMSPROP_EPSILON: Default epsilon for RMSprop (1e-8)
    DEFAULT_ADAGRAD_EPSILON: Default epsilon for AdaGrad (1e-10)
"""

# SGD learning rate following common deep learning convention
comptime DEFAULT_LEARNING_RATE_SGD: Float64 = 0.01

# Adam learning rate following PyTorch defaults
comptime DEFAULT_LEARNING_RATE_ADAM: Float64 = 0.001

# Momentum for SGD and variants
comptime DEFAULT_MOMENTUM: Float64 = 0.9

# Adam beta coefficients
comptime DEFAULT_ADAM_BETA1: Float64 = 0.9
comptime DEFAULT_ADAM_BETA2: Float64 = 0.999

# Adam epsilon for numerical stability
comptime DEFAULT_ADAM_EPSILON: Float64 = 1e-8

# RMSprop alpha (decay rate) and epsilon
comptime DEFAULT_RMSPROP_ALPHA: Float64 = 0.99
comptime DEFAULT_RMSPROP_EPSILON: Float64 = 1e-8

# AdaGrad epsilon for numerical stability
comptime DEFAULT_ADAGRAD_EPSILON: Float64 = 1e-10
