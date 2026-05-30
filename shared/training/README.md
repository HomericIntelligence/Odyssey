# Training Utilities

This directory contains components for training neural networks across different paper implementations.

## Purpose

Training utilities provide common functionality needed for model training:

- **Optimizers** - Optimization algorithms (SGD, Adam, etc.)
- **Schedulers** - Learning rate scheduling strategies
- **Training Loops** - Common training and validation patterns
- **Callbacks** - Training callbacks for logging, checkpointing, etc.
- **Loss Functions** - Standard loss functions used across models

## Guidelines

- Ensure compatibility with different model architectures
- Provide clear configuration options for training parameters
- Include logging and progress tracking
- Write tests for training components with small synthetic datasets
