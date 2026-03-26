#!/usr/bin/env bash

# VGG-16 CIFAR-10 Complete Workflow Script
#
# This script runs the complete VGG-16 training and evaluation pipeline:
# 1. Download and prepare CIFAR-10 dataset
# 2. Train VGG-16 model for 200 epochs
# 3. Evaluate trained model on test set
#
# Usage:
#   chmod +x examples/vgg16_cifar10/run_example.sh
#   ./examples/vgg16_cifar10/run_example.sh

set -e  # Exit on error

echo "========================================="
echo "VGG-16 CIFAR-10 Complete Workflow"
echo "========================================="
echo ""

# Step 1: Download dataset
echo "Step 1: Downloading CIFAR-10 dataset..."
echo "-----------------------------------------"
python3 examples/vgg16_cifar10/download_cifar10.py
echo ""
echo "✓ Dataset downloaded and converted to IDX format"
echo ""

# Step 2: Train model
echo "Step 2: Training VGG-16 model..."
echo "-----------------------------------------"
echo "Configuration:"
echo "  - Epochs: 200"
echo "  - Batch size: 128"
echo "  - Initial learning rate: 0.01"
echo "  - LR decay: 5x every 60 epochs"
echo "  - Optimizer: SGD with momentum (0.9)"
echo "  - Dropout: 0.5"
echo ""
echo "Expected training time: ~30-40 hours on CPU"
echo ""

# Run training
mojo run examples/vgg16_cifar10/train.mojo \
    --epochs 200 \
    --batch-size 128 \
    --lr 0.01 \
    --momentum 0.9 \
    --data-dir datasets/cifar10 \
    --weights-dir vgg16_weights

echo ""
echo "✓ Training completed"
echo ""

# Step 3: Evaluate on test set
echo "Step 3: Evaluating on test set..."
echo "-----------------------------------------"
mojo run examples/vgg16_cifar10/inference.mojo \
    --weights-dir vgg16_weights \
    --data-dir datasets/cifar10

echo ""
echo "========================================="
echo "VGG-16 Workflow Completed!"
echo "========================================="
echo ""
echo "Results:"
echo "  - Trained model weights: vgg16_weights/"
echo "  - Expected accuracy: 91-93% (with proper training)"
echo ""
echo "Next steps:"
echo "  - Review training logs for convergence"
echo "  - Try different hyperparameters"
echo "  - Add data augmentation for better accuracy"
echo ""
