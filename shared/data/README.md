# Data Processing

This directory contains utilities for loading, preprocessing, and augmenting data across different paper implementations.

## Purpose

Data processing utilities provide common functionality for working with datasets:

- **Data Loaders** - Dataset loading and batching
- **Preprocessors** - Data normalization and preprocessing
- **Augmentation** - Data augmentation techniques
- **Utilities** - Helper functions for data manipulation

## Guidelines

- Design loaders to be memory-efficient
- Implement augmentation as composable transforms
- Support both training and inference modes
- Include documentation on expected data formats
- Write tests with small sample datasets
