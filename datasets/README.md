# Datasets Directory

## Overview

This directory stores datasets used for training and evaluating ML models in the ML Odyssey project.

## Structure

```text
datasets/
├── mnist/            # MNIST handwritten digits dataset
├── cifar10/          # CIFAR-10 image classification dataset
├── synthetic/        # Synthetic datasets for testing
└── custom/           # Custom datasets for specific papers
```text

## Quick Start

Download MNIST dataset:

```python
from projectodyssey.data import download_mnist
download_mnist("datasets/mnist/")
```text

## Usage

### Dataset Organization

Each dataset should be organized in its own subdirectory with:

- `raw/` - Original downloaded data
- `processed/` - Preprocessed data ready for training
- `README.md` - Dataset documentation
- `metadata.json` - Dataset metadata and statistics

### Adding New Datasets

1. Create dataset subdirectory
1. Add download script
1. Document preprocessing steps
1. Include dataset statistics
1. Add usage examples

### Dataset Format

All datasets should provide:

- Training data
- Validation data
- Test data
- Metadata (shape, classes, etc.)

## Available Datasets

### MNIST

- 60,000 training images
- 10,000 test images
- 28x28 grayscale images
- 10 digit classes (0-9)

### CIFAR-10

- 50,000 training images
- 10,000 test images
- 32x32 color images
- 10 object classes

### Synthetic

- Generated datasets for testing
- Configurable size and complexity
- Used for unit tests

## Data Loading

Example loading MNIST:

```python
from projectodyssey.data import load_dataset

train_data, train_labels = load_dataset("mnist", split="train")
test_data, test_labels = load_dataset("mnist", split="test")
```text

## Storage Guidelines

- Keep raw data in `raw/` subdirectory
- Process and save to `processed/`
- Use appropriate compression
- Document all preprocessing steps
- Include checksums for verification

## Privacy and Ethics

- Only use publicly available datasets
- Respect dataset licenses
- Document data sources
- Consider privacy implications
- Follow ethical AI guidelines
