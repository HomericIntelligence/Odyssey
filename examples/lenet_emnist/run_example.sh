#!/usr/bin/env bash
# Example runner script for LeNet-5 EMNIST
#
# This script demonstrates the complete workflow:
# 1. Download dataset
# 2. Train model (when available)
# 3. Run inference (when available)
#
# Usage:
#   bash examples/lenet_emnist/run_example.sh

set -e  # Exit on error

echo "======================================================================"
echo "LeNet-5 on EMNIST - Complete Example Workflow"
echo "======================================================================"
echo ""

# Step 1: Download dataset
echo "[Step 1/3] Downloading EMNIST dataset..."
echo "----------------------------------------------------------------------"
python3 scripts/download_emnist.py --split balanced
echo ""

# Step 2: Train model (when available)
echo "[Step 2/3] Training LeNet-5 model..."
echo "----------------------------------------------------------------------"
echo "Note: Training not yet available - waiting for stable Mojo file I/O"
echo "When available, run:"
echo "  mojo run examples/lenet_emnist/train.mojo --epochs 10 --batch-size 32 --lr 0.01"
# mojo run examples/lenet_emnist/train.mojo --epochs 10 --batch-size 32 --lr 0.01
echo ""

# Step 3: Run inference (when available)
echo "[Step 3/3] Running inference..."
echo "----------------------------------------------------------------------"
echo "Note: Inference not yet available - waiting for stable Mojo file I/O"
echo "When available, run:"
echo "  mojo run examples/lenet_emnist/inference.mojo --weights lenet5_emnist.weights"
# mojo run examples/lenet_emnist/inference.mojo --weights lenet5_emnist.weights
echo ""

echo "======================================================================"
echo "Example workflow complete!"
echo "======================================================================"
echo ""
echo "Current Status: Skeleton implementation demonstrating structure"
echo "Full functionality will be available when Mojo stdlib stabilizes."
