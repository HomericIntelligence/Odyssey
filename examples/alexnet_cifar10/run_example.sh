#!/bin/bash

###############################################################################
# AlexNet on CIFAR-10 - Complete Workflow Script
#
# This script demonstrates the full workflow for training and evaluating
# AlexNet on the CIFAR-10 dataset using ML Odyssey.
#
# Steps:
#   1. Download and prepare CIFAR-10 dataset
#   2. Train AlexNet model
#   3. Run inference on test set
#
# Usage:
#   ./examples/alexnet_cifar10/run_example.sh
#
# Requirements:
#   - Python 3.7+ with numpy
#   - Mojo compiler
#   - ML Odyssey shared library
###############################################################################

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DATA_DIR="datasets/cifar10"
WEIGHTS_DIR="alexnet_weights"
EPOCHS=100
BATCH_SIZE=128
LEARNING_RATE=0.01
MOMENTUM=0.9

echo "================================================================================"
echo "AlexNet on CIFAR-10 - Complete Workflow"
echo "================================================================================"
echo ""

#------------------------------------------------------------------------------
# Step 1: Check Python dependencies
#------------------------------------------------------------------------------
echo -e "${BLUE}[Step 1/4] Checking Python dependencies...${NC}"
if ! python3 -c "import numpy" &> /dev/null; then
    echo -e "${RED}Error: numpy is required. Install with: pip install numpy${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python dependencies satisfied${NC}"
echo ""

#------------------------------------------------------------------------------
# Step 2: Download and prepare CIFAR-10 dataset
#------------------------------------------------------------------------------
echo -e "${BLUE}[Step 2/4] Downloading and preparing CIFAR-10 dataset...${NC}"
if [ -d "$DATA_DIR" ] && [ -f "$DATA_DIR/test_batch_images.idx" ]; then
    echo -e "${YELLOW}Dataset already exists at $DATA_DIR${NC}"
    read -p "Re-download and prepare? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$DATA_DIR"
        python3 examples/alexnet_cifar10/download_cifar10.py --output-dir "$DATA_DIR"
    else
        echo "Using existing dataset"
    fi
else
    python3 examples/alexnet_cifar10/download_cifar10.py --output-dir "$DATA_DIR"
fi
echo -e "${GREEN}✓ Dataset ready${NC}"
echo ""

#------------------------------------------------------------------------------
# Step 3: Train AlexNet model
#------------------------------------------------------------------------------
echo -e "${BLUE}[Step 3/4] Training AlexNet model...${NC}"
echo "Configuration:"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Momentum: $MOMENTUM"
echo "  Data Directory: $DATA_DIR"
echo "  Weights Directory: $WEIGHTS_DIR"
echo ""

if [ -d "$WEIGHTS_DIR" ]; then
    echo -e "${YELLOW}Weights directory already exists: $WEIGHTS_DIR${NC}"
    read -p "Re-train model? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$WEIGHTS_DIR"
        mojo run examples/alexnet_cifar10/train.mojo \
            --epochs "$EPOCHS" \
            --batch-size "$BATCH_SIZE" \
            --lr "$LEARNING_RATE" \
            --momentum "$MOMENTUM" \
            --data-dir "$DATA_DIR" \
            --weights-dir "$WEIGHTS_DIR"
    else
        echo "Skipping training"
    fi
else
    mojo run examples/alexnet_cifar10/train.mojo \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --lr "$LEARNING_RATE" \
        --momentum "$MOMENTUM" \
        --data-dir "$DATA_DIR" \
        --weights-dir "$WEIGHTS_DIR"
fi
echo -e "${GREEN}✓ Training complete${NC}"
echo ""

#------------------------------------------------------------------------------
# Step 4: Run inference on test set
#------------------------------------------------------------------------------
echo -e "${BLUE}[Step 4/4] Running inference on test set...${NC}"
mojo run examples/alexnet_cifar10/inference.mojo \
    --weights-dir "$WEIGHTS_DIR" \
    --data-dir "$DATA_DIR"
echo -e "${GREEN}✓ Inference complete${NC}"
echo ""

#------------------------------------------------------------------------------
# Summary
#------------------------------------------------------------------------------
echo "================================================================================"
echo -e "${GREEN}AlexNet on CIFAR-10 - Workflow Complete!${NC}"
echo "================================================================================"
echo ""
echo "Summary:"
echo "  Dataset: $DATA_DIR"
echo "  Trained Weights: $WEIGHTS_DIR"
echo ""
echo "Next Steps:"
echo "  - Review training logs for loss and accuracy"
echo "  - Adjust hyperparameters (epochs, learning rate, momentum)"
echo "  - Add learning rate decay for better convergence"
echo "  - Implement data augmentation for improved accuracy"
echo ""
echo "For more information, see: examples/alexnet_cifar10/README.md"
echo ""
