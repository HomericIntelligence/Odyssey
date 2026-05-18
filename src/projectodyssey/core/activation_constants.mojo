"""Activation function constants for neural network operations.

This module provides centralized constants for activation function parameters,
including bounds, thresholds, and scaling factors used across the codebase.

Constants:
    RELU6_UPPER_BOUND: Upper bound for ReLU6 activation (6.0)
    SIGMOID_CLIP_THRESHOLD: Clipping threshold for numerical stability in sigmoid (20.0)
    HARD_SIGMOID_OFFSET: Offset in hard sigmoid formula (3.0)
    HARD_SIGMOID_SCALE: Scale factor in hard sigmoid formula (6.0)
    HARD_TANH_LOWER_BOUND: Lower bound for hard tanh (-1.0)
    HARD_TANH_UPPER_BOUND: Upper bound for hard tanh (1.0)
"""

# ReLU family bounds
comptime RELU6_UPPER_BOUND: Float64 = 6.0

# Sigmoid numerical stability clipping threshold
comptime SIGMOID_CLIP_THRESHOLD: Float64 = 20.0

# Hard sigmoid formula: clip((x + OFFSET) / SCALE, 0, 1)
comptime HARD_SIGMOID_OFFSET: Float64 = 3.0
comptime HARD_SIGMOID_SCALE: Float64 = 6.0

# Hard tanh bounds: clip(x, -1, 1)
comptime HARD_TANH_LOWER_BOUND: Float64 = -1.0
comptime HARD_TANH_UPPER_BOUND: Float64 = 1.0
