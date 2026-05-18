# Memory Requirements and Batch Size Guidelines

## Overview

This document provides guidelines for memory management when training neural networks with ML Odyssey, specifically
addressing the LeNet-5 EMNIST example and general best practices.

## Memory Calculation Basics

### Tensor Memory Formula

```text
Memory (bytes) = num_elements × bytes_per_element
```text

Where:

- `num_elements` = product of all shape dimensions
- `bytes_per_element` depends on data type:
  - `DType.float32`: 4 bytes
  - `DType.float64`: 8 bytes
  - `DType.float16`: 2 bytes
  - `DType.int32`: 4 bytes
  - `DType.uint8`: 1 byte

### Example: EMNIST Images

Single EMNIST image: `(1, 28, 28)` with `float32`

```text
Memory = 1 × 28 × 28 × 4 bytes = 3,136 bytes ≈ 3.1 KB
```text

Batch of 32 images:

```text
Memory = 32 × 1 × 28 × 28 × 4 bytes = 100,352 bytes ≈ 98 KB
```text

Full training set (112,800 images):

```text
Memory = 112,800 × 1 × 28 × 28 × 4 bytes = 353,894,400 bytes ≈ 337 MB
```text

## LeNet-5 Memory Requirements

### Forward Pass Memory

For a single forward pass with batch size `B`:

1. **Input**: `(B, 1, 28, 28)` = `B × 3,136` bytes
2. **Conv1 Output**: `(B, 6, 24, 24)` = `B × 13,824` bytes
3. **Pool1 Output**: `(B, 6, 12, 12)` = `B × 3,456` bytes
4. **Conv2 Output**: `(B, 16, 8, 8)` = `B × 4,096` bytes
5. **Pool2 Output**: `(B, 16, 4, 4)` = `B × 1,024` bytes
6. **FC1 Output**: `(B, 120)` = `B × 480` bytes
7. **FC2 Output**: `(B, 84)` = `B × 336` bytes
8. **FC3 Output**: `(B, 47)` = `B × 188` bytes

**Total per forward pass**: ~`B × 26,540 bytes` ≈ `B × 26 KB`

### Backward Pass Memory

Backward pass requires gradients for each layer, roughly **doubling** forward pass memory:

**Total with gradients**: ~`B × 53 KB`

### Model Parameters

LeNet-5 parameters (~61,706 parameters with `float32`):

```text
Parameters = 61,706 × 4 bytes = 246,824 bytes ≈ 241 KB
```text

### Full Training Iteration Memory

For batch size `B`:

```text
Total ≈ (B × 53 KB) + 241 KB + dataset_memory
```text

## Memory Limits and Safety

### Built-in Limits

ML Odyssey enforces the following limits (defined in `src/projectodyssey/core/extensor.mojo`):

- **MAX_TENSOR_BYTES**: 2,000,000,000 bytes (2 GB) - Hard limit per tensor
- **WARN_TENSOR_BYTES**: 500,000,000 bytes (500 MB) - Warning threshold

Any attempt to create a tensor exceeding `MAX_TENSOR_BYTES` will raise an error:

```text
Error: Tensor too large: 1500000000 bytes exceeds maximum 2000000000 bytes.
Consider using smaller batch sizes.
```text

### Calculating Maximum Batch Size

Use the `calculate_max_batch_size()` helper function:

```mojo
from projectodyssey.core.extensor import calculate_max_batch_size

# For EMNIST images (1, 28, 28) with float32
var sample_shape = List[Int]()
sample_shape.append(1)
sample_shape.append(28)
sample_shape.append(28)

var max_batch = calculate_max_batch_size(
    sample_shape,
    DType.float32,
    max_memory_bytes=500_000_000  # 500 MB limit
)
print("Maximum batch size:", max_batch)  # ~159,439 samples
```text

## Recommended Batch Sizes

### EMNIST Dataset (47 classes, 28×28 images)

| Batch Size | Input Memory | Forward Memory | Total Estimate | Recommendation |
| --- | --- | --- | --- | --- |
| 32 | 98 KB | 832 KB | ~1.7 MB | Ideal for testing |
| 64 | 196 KB | 1.6 MB | ~3.4 MB | Good for training |
| 128 | 393 KB | 3.3 MB | ~6.8 MB | Good for training |
| 256 | 786 KB | 6.6 MB | ~13.6 MB | Good for training |
| 512 | 1.6 MB | 13.2 MB | ~27 MB | Monitor memory |
| 1024 | 3.1 MB | 26.5 MB | ~54 MB | Monitor memory |
| 112,800 | 337 MB | 2.9 GB | ~5.9 GB | Exceeds limits! |

### General Guidelines

1. **Start small**: Begin with batch size 32 for testing
2. **Increase gradually**: Double batch size until you reach memory warnings
3. **Monitor output**: Watch for "Warning: Large tensor allocation" messages
4. **Leave headroom**: Don't use 100% of available memory (leave ~20% free)
5. **Consider GPU memory**: If using GPU, memory is typically more constrained

## Error Handling

### Out of Memory Errors

If you encounter an error like:

```text
Error: Tensor too large: 1500000000 bytes exceeds maximum 2000000000 bytes.
Consider using smaller batch sizes.
```text

**Solutions**:

1. Reduce batch size (try half the current size)
2. Use mixed precision (float16 instead of float32)
3. Process dataset in smaller chunks

### Memory Warnings

If you see:

```text
Warning: Large tensor allocation: 600000000 bytes
```text

**Actions**:

- This is informational - the allocation succeeded
- Consider reducing batch size if you encounter crashes later
- Monitor system memory usage

## Best Practices

### 1. Use Mini-Batch Processing

**DON'T** process the entire dataset at once:

```mojo
# ❌ BAD: Processes all 112,800 samples
var output = model.forward(all_training_data)
```text

**DO** use mini-batches with slicing:

```mojo
# ✅ GOOD: Process in batches of 32
for batch_idx in range(num_batches):
    var start = batch_idx * 32
    var end = min(start + 32, num_samples)
    var batch = train_images.slice(start, end, axis=0)
    var output = model.forward(batch)
```text

### 2. Calculate Batch Size Programmatically

```mojo
# Calculate safe batch size based on available memory
var sample_shape = List[Int]()
sample_shape.append(1)
sample_shape.append(28)
sample_shape.append(28)

var max_batch = calculate_max_batch_size(sample_shape, DType.float32)
var batch_size = min(max_batch, 256)  # Cap at 256 for training stability
```text

### 3. Use Views for Slicing

The `slice()` method creates memory-efficient views:

```mojo
# Creates a view (shares memory, zero copy)
var batch = dataset.slice(0, 32, axis=0)
```text

This avoids copying data, saving both time and memory.

### 4. Monitor Memory During Development

Add logging to track memory usage:

```mojo
var sample_shape = List[Int]()
// ... set up shape ...
var bytes_per_sample = calculate_bytes_per_sample(sample_shape, dtype)
var batch_memory = batch_size * bytes_per_sample

print("Batch memory requirement:", batch_memory, "bytes")
if batch_memory > 100_000_000:  # 100 MB
    print("⚠️  Large batch - consider reducing size")
```text

## Platform-Specific Considerations

### Desktop/Workstation

- Typical RAM: 16-64 GB
- Safe batch sizes: 128-512 for EMNIST
- Can handle larger models

### Laptop

- Typical RAM: 8-16 GB
- Safe batch sizes: 32-128 for EMNIST
- Be conservative with batch sizes

### Cloud/Server

- Typical RAM: 32-256 GB
- Safe batch sizes: 256-2048 for EMNIST
- Can handle very large batches

## Troubleshooting

### Segmentation Fault (Segfault)

**Cause**: Attempting to allocate more memory than available

**Solution**:

1. Check if batch size × sample size exceeds `MAX_TENSOR_BYTES`
2. Reduce batch size significantly (try 32 or 64)
3. Verify you're using `slice()` for batching, not processing full dataset

### Training Extremely Slow

**Cause**: Batch size too small (excessive overhead) or too large (memory swapping)

**Solution**:

1. Try batch sizes in range [64, 256] for EMNIST
2. Monitor system memory usage (`htop` or `top` on Linux)
3. Aim for ~50-80% memory utilization

### Inconsistent Results Between Runs

**Cause**: Memory errors or data corruption from oversized batches

**Solution**:

1. Use recommended batch sizes
2. Validate that all tensors are properly allocated
3. Check for memory warnings in output

## References

- LeNet-5 Paper: LeCun et al. (1998), "Gradient-based learning applied to document recognition"
- EMNIST Dataset: Cohen et al. (2017), "EMNIST: an extension of MNIST to handwritten letters"
- ML Odyssey Documentation: [Repository README](index.md)

## Quick Reference

```python
# Calculate memory for a tensor
memory_bytes = product(shape) * dtype_size

# EMNIST recommended batch sizes
batch_sizes = {
    "testing": 32,
    "training": 128,
    "maximum_safe": 256
}

# Memory limits
MAX_TENSOR_BYTES = 2_000_000_000  # 2 GB
WARN_TENSOR_BYTES = 500_000_000   # 500 MB
```text
