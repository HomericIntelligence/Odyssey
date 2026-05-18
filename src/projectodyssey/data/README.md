# Data Processing Library (WIP)

The `data` library provides data loading, preprocessing, and augmentation utilities for training neural networks
in ML Odyssey. All components are implemented in Mojo for maximum performance.

## Purpose

This library provides:

- Dataset abstractions for various data sources
- High-performance data loaders with batching and shuffling
- Data transformation and augmentation pipelines
- Efficient memory management for large datasets

**Key Principle**: Provide flexible, composable data processing components that work seamlessly with the training
pipeline while maintaining high performance.

## Directory Organization

```text
data/
├── __init__.mojo                    # Package root - exports main components
├── README.md                        # This file
├── datasets.mojo                    # Dataset abstractions and implementations
├── _datasets_core.mojo              # EMNIST and related implementations
├── loaders.mojo                     # Data loaders and samplers
├── transforms.mojo                  # Data transformation and augmentation
├── dataset_with_transform.mojo      # Transform wrapper for datasets
├── prefetch.mojo                    # Prefetch buffer infrastructure
├── cache.mojo                       # Caching layer for datasets
├── samplers.mojo                    # Sampling strategies
├── batch_utils.mojo                 # Batch utility functions
├── formats/                         # Low-level format loaders (IDX, CIFAR)
└── datasets/                        # Dataset implementations
```

## What Belongs in Data

### Include

- Dataset abstractions (base traits/interfaces)
- Common dataset implementations (TensorDataset, ImageDataset)
- Data loaders with batching and shuffling
- Standard transforms (normalization, resizing, cropping)
- Data augmentation (random flips, rotations, crops)
- Sampling strategies (sequential, random, weighted)

### Exclude

- Paper-specific datasets (belongs in papers/\*/data/)
- Paper-specific data preprocessing (belongs with the paper)
- Custom file formats without broad applicability
- One-off data manipulation code

## Architecture Overview

The data pipeline uses a modular, composable architecture:

```text
Raw Data → Dataset → TransformedDataset → CachedDataset → BatchLoader → Training
                         ↑                                      ↑
                    (Optional)                             (Sampler controls order)
```

### Key Design Principles

1. **Composability**: Components chain together seamlessly
2. **Lazy Evaluation**: Data is processed on-demand, not eagerly
3. **Type Safety**: Trait-based interfaces with compile-time checking
4. **Performance**: SIMD-optimized operations where appropriate
5. **Memory Efficiency**: Streaming data without loading everything upfront

## Components

### Datasets (`datasets.mojo`)

Base abstractions and common implementations for datasets.

#### Dataset Trait

```mojo
trait Dataset:
    """Base interface for all datasets."""
    fn __len__(self) -> Int
    fn __getitem__(self, index: Int) -> (Tensor, Tensor)
```

#### TensorDataset

In-memory dataset for tensor data:

```mojo
struct TensorDataset(Dataset):
    """Dataset wrapping tensors."""
    var data: Tensor
    var targets: Tensor
    var transform: Optional[Transform]

    fn __init__(out self, data: Tensor, targets: Tensor, transform: Optional[Transform] = None):
        """Create dataset from tensors."""
        self.data = data
        self.targets = targets
        self.transform = transform

    fn __len__(self) -> Int:
        """Return number of samples."""
        return self.data.shape[0]

    fn __getitem__(self, index: Int) -> (Tensor, Tensor):
        """Get sample at index."""
        var sample = self.data[index]
        var target = self.targets[index]

        if self.transform:
            sample = self.transform.value()(sample)

        return (sample, target)
```

**Use Case**: Training on MNIST, CIFAR-10, or any in-memory dataset

#### ImageDataset

Dataset for loading images from disk:

```mojo
struct ImageDataset(Dataset):
    """Dataset for loading images from files."""
    var image_paths: List[String]
    var labels: List[Int]
    var transform: Optional[Transform]

    fn __init__(out self, image_paths: List[String], labels: List[Int], transform: Optional[Transform] = None):
        """Create image dataset from file paths."""
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    fn __len__(self) -> Int:
        return len(self.image_paths)

    fn __getitem__(self, index: Int) -> (Tensor, Tensor):
        """Load and transform image at index."""
        var image = load_image(self.image_paths[index])
        var label = self.labels[index]

        if self.transform:
            image = self.transform.value()(image)

        return (image, label)
```

**Use Case**: ImageNet, custom image datasets

#### CSVDataset

Dataset for loading data from CSV files:

```mojo
struct CSVDataset(Dataset):
    """Dataset for loading structured data from CSV."""
    var data: Tensor
    var targets: Tensor

    fn __init__(out self, filepath: String, target_column: Int):
        """Load CSV file and split features from targets."""
        # Load CSV and convert to tensors
        pass
```

**Use Case**: Tabular data, benchmarks

### Dataset Wrappers

#### TransformedDataset (`dataset_with_transform.mojo`)

Wraps a dataset with a transform pipeline applied during data loading:

```mojo
struct TransformedDataset[D: Dataset, T](Dataset):
    """Applies transform to data, leaves labels unchanged."""
    var dataset: D
    var transform: T

    fn __getitem__(self, index: Int) -> (AnyTensor, AnyTensor):
        var data, labels = self.dataset[index]
        var transformed = self.transform(data)
        return (transformed, labels)
```

**Use Cases**:

- Data augmentation during training (Flip, Rotate, Crop)
- Normalization with dataset-specific statistics
- Custom preprocessing pipelines

**Example**:

```mojo
var dataset = AnyTensorDataset(images, labels)
var normalize = Normalize(mean=0.5, std=0.5)
var augmented = TransformedDataset(dataset, normalize)
# augmented[0] returns (normalized_data, original_labels)
```

#### CachedDataset (`cache.mojo`)

Caches samples to avoid repeated I/O:

```mojo
struct CachedDataset[D: Dataset](Dataset):
    """Caches loaded samples up to max_cache_size."""
    var dataset: D
    var cache: Dict[Int, Tuple[AnyTensor, AnyTensor]]
    var max_cache_size: Int

    fn __getitem__(mut self, index: Int) -> (AnyTensor, AnyTensor):
        if index in cache:
            return cache[index]  # Cache hit
        # Load from dataset and cache
        var sample = self.dataset[index]
        if cache.size() < max_cache_size:
            cache[index] = sample
        return sample
```

**Use Cases**:

- Small datasets that fit in memory (< 1GB)
- Expensive preprocessing (transforms, file I/O)
- Repeated iteration over same data

**Example**:

```mojo
var dataset = FileDataset(paths, labels)  # Slow I/O
var cached = CachedDataset(dataset, max_cache_size=-1)
cached._preload_cache()  # Load all samples
# Subsequent accesses are instant cache hits
```

### Data Loaders (`loaders.mojo`)

Efficient batch loading with parallel data loading and prefetching.

#### DataLoader

Main data loader with batching and shuffling:

```mojo
struct DataLoader:
    """Iterate over dataset in batches."""
    var dataset: Dataset
    var batch_size: Int
    var shuffle: Bool
    var drop_last: Bool
    var num_workers: Int

    fn __init__(
        out self,
        dataset: Dataset,
        batch_size: Int = 1,
        shuffle: Bool = False,
        drop_last: Bool = False,
        num_workers: Int = 0
    ):
        """Initialize data loader."""
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers

    fn __iter__(self) -> DataLoaderIterator:
        """Return iterator over batches."""
        var sampler = RandomSampler(len(self.dataset)) if self.shuffle else SequentialSampler(len(self.dataset))
        var batch_sampler = BatchSampler(sampler, self.batch_size, self.drop_last)
        return DataLoaderIterator(self.dataset, batch_sampler)
```

### Features

- Batching: Group samples into batches
- Shuffling: Randomize sample order per epoch (with RandomSampler)
- Drop last: Handle incomplete final batch
- Multi-worker: Parallel data loading (future enhancement)
- Prefetching: Pre-compute batches ahead of consumption

#### Sampling Strategies (`samplers.mojo`)

Control data iteration order:

```mojo
trait Sampler:
    fn __len__(self) -> Int
    fn __iter__(self) raises -> List[Int]

struct SequentialSampler(Sampler):
    """Sequential sampling (0, 1, 2, ..., N-1) - no shuffling."""

struct RandomSampler(Sampler):
    """Random permutation - enables epoch shuffling."""
    var num_samples: Int
    var shuffle: Bool

struct WeightedSampler(Sampler):
    """Weighted sampling with replacement."""
    var weights: List[Float32]
```

**Use RandomSampler for training** to enable shuffling between epochs.

#### Prefetching (`prefetch.mojo`)

Pre-computes batches to improve pipeline efficiency:

```mojo
struct PrefetchDataLoader[D: Dataset, S: Sampler]:
    """Wraps BatchLoader with batch prefetching."""
    var base_loader: BatchLoader[D, S]
    var prefetch_factor: Int

    fn __iter__(mut self) -> List[Batch]:
        """Pre-compute all batches for this epoch."""
        return self.base_loader.__iter__()
```

**Note**: Currently uses synchronous pre-computation. Can be upgraded to
async when Mojo adds Task primitives and thread-safe queues.

**Benefits**:

- Separates data loading logic from training logic
- Clear interface for future async upgrades
- Easier debugging and profiling

#### Batch

Container for batch data:

```mojo
struct Batch:
    """Container for batch of data."""
    var inputs: Tensor
    var targets: Tensor
    var indices: List[Int]

    fn __init__(out self, inputs: Tensor, targets: Tensor, indices: List[Int]):
        """Create batch container."""
        self.inputs = inputs
        self.targets = targets
        self.indices = indices
```

#### Samplers

Control sampling order:

```mojo
trait Sampler:
    """Base sampler interface."""
    fn __iter__(self) -> SamplerIterator
    fn __len__(self) -> Int

struct SequentialSampler(Sampler):
    """Sample elements sequentially."""
    var dataset_size: Int

struct RandomSampler(Sampler):
    """Sample elements randomly."""
    var dataset_size: Int
    var generator: RandomNumberGenerator

struct WeightedRandomSampler(Sampler):
    """Sample elements with given probabilities."""
    var weights: Tensor
    var num_samples: Int
```

### Transforms (`transforms.mojo`)

Data transformation and augmentation pipeline.

#### Transform Trait

```mojo
trait Transform:
    """Base interface for transforms."""
    fn __call__(self, x: Tensor) -> Tensor
```

#### Core Transforms

**ToTensor**: Convert data to tensor format

```mojo
struct ToTensor(Transform):
    """Convert input to Tensor."""
    fn __call__(self, x: Any) -> Tensor:
        """Convert to tensor."""
        return to_tensor(x)
```

**Normalize**: Normalize tensor values

```mojo
struct Normalize(Transform):
    """Normalize tensor with mean and std."""
    var mean: Float32
    var std: Float32

    fn __init__(out self, mean: Float32, std: Float32):
        self.mean = mean
        self.std = std

    fn __call__(self, x: Tensor) -> Tensor:
        """Apply normalization: (x - mean) / std."""
        return (x - self.mean) / self.std
```

**Compose**: Chain multiple transforms

```mojo
struct Compose(Transform):
    """Compose multiple transforms."""
    var transforms: List[Transform]

    fn __init__(out self, transforms: List[Transform]):
        self.transforms = transforms

    fn __call__(self, x: Tensor) -> Tensor:
        """Apply transforms sequentially."""
        var output = x
        for transform in self.transforms:
            output = transform(output)
        return output
```

#### Image Transforms

**Resize**: Resize images

```mojo
struct Resize(Transform):
    """Resize image to target size."""
    var height: Int
    var width: Int
    var interpolation: InterpolationMode

    fn __call__(self, x: Tensor) -> Tensor:
        """Resize image."""
        return resize_image(x, self.height, self.width, self.interpolation)
```

**CenterCrop**: Crop center region

```mojo
struct CenterCrop(Transform):
    """Crop center region of image."""
    var size: Int

    fn __call__(self, x: Tensor) -> Tensor:
        """Crop center."""
        return center_crop(x, self.size)
```

#### Data Augmentation

**RandomCrop**: Random crop for augmentation

```mojo
struct RandomCrop(Transform):
    """Randomly crop image."""
    var size: Int
    var padding: Int

    fn __call__(self, x: Tensor) -> Tensor:
        """Apply random crop."""
        var padded = pad_image(x, self.padding) if self.padding > 0 else x
        return random_crop(padded, self.size)
```

**RandomHorizontalFlip**: Random horizontal flip

```mojo
struct RandomHorizontalFlip(Transform):
    """Randomly flip image horizontally."""
    var p: Float32  # Probability

    fn __call__(self, x: Tensor) -> Tensor:
        """Apply random flip with probability p."""
        if random() < self.p:
            return horizontal_flip(x)
        return x
```

**RandomRotation**: Random rotation

```mojo
struct RandomRotation(Transform):
    """Randomly rotate image."""
    var degrees: (Float32, Float32)  # Angle range

    fn __call__(self, x: Tensor) -> Tensor:
        """Apply random rotation."""
        var angle = random_uniform(self.degrees[0], self.degrees[1])
        return rotate_image(x, angle)
```

## Usage Examples

### Basic Usage with Modern Architecture

```mojo
from projectodyssey.data import (
    AnyTensorDataset, BatchLoader, RandomSampler, Normalize,
    TransformedDataset
)

# Create base dataset
var dataset = AnyTensorDataset(images, labels)

# (Optional) Add transforms for augmentation
var normalize = Normalize(mean=0.5, std=0.5)
var augmented = TransformedDataset(dataset, normalize)

# Create sampler (RandomSampler enables shuffling)
var sampler = RandomSampler(augmented.__len__())

# Create loader
var loader = BatchLoader(
    augmented,
    sampler,
    batch_size=32,
    drop_last=False
)

# Iterate over batches
var batches = loader.__iter__()
for batch in batches:
    var data = batch.data
    var labels = batch.labels
    # ... training code
```

### Complete Pipeline with Caching and Prefetching

```mojo
from projectodyssey.data import (
    AnyTensorDataset, BatchLoader, RandomSampler, CachedDataset,
    TransformedDataset, PrefetchDataLoader, Normalize
)

# Step 1: Create dataset
var dataset = AnyTensorDataset(images, labels)

# Step 2: Add transforms
var normalize = Normalize(mean=0.5, std=0.5)
var transformed = TransformedDataset(dataset, normalize)

# Step 3: Add caching (for small datasets)
var cached = CachedDataset(transformed, max_cache_size=-1)
cached._preload_cache()  # Pre-populate cache

# Step 4: Create sampler
var sampler = RandomSampler(cached.__len__())

# Step 5: Create loader
var loader = BatchLoader(cached, sampler, batch_size=32)

# Step 6: Create prefetch loader
var prefetch = PrefetchDataLoader(loader, prefetch_factor=2)

# Training loop
for epoch in range(num_epochs):
    var batches = prefetch.__iter__()
    for batch in batches:
        # Forward/backward pass
        var loss = train_step(model, batch.data, batch.labels)

    # Check cache stats
    var cache_size, hits, misses = cached.get_cache_stats()
    print(f"Epoch {epoch}: cache hits={hits}, misses={misses}")
```

### With Transform Composition

```mojo
from projectodyssey.data import (
    AnyTensorDataset, TransformedDataset, RandomHorizontalFlip,
    RandomCrop, Normalize, Compose
)

# Create transform pipeline
var transforms = Compose([
    RandomCrop(32, padding=4),
    RandomHorizontalFlip(p=0.5),
    Normalize(mean=0.5, std=0.5),
])

# Create dataset
var dataset = AnyTensorDataset(images, labels)

# Apply transforms
var augmented = TransformedDataset(dataset, transforms)

# Use in loader
var sampler = RandomSampler(augmented.__len__())
var loader = BatchLoader(augmented, sampler, batch_size=128)
```

### Custom Dataset

```mojo
from projectodyssey.data import Dataset, DataLoader

struct MyDataset(Dataset):
    """Custom dataset implementation."""
    var data: MyDataType

    fn __len__(self) -> Int:
        return len(self.data)

    fn __getitem__(self, index: Int) -> (Tensor, Tensor):
        # Custom loading logic
        var sample = self.data[index]
        var features = preprocess(sample)
        var label = extract_label(sample)
        return (features, label)

var dataset = MyDataset(my_data)
var loader = DataLoader(dataset, batch_size=32)
```

## Performance Considerations

### Memory Management

**Problem**: Loading large datasets can exhaust memory

### Solutions

1. **Lazy Loading**: Load data on-demand rather than all at once
1. **Memory Mapping**: Use memory-mapped files for large datasets
1. **Streaming**: Process data in streaming fashion without full load

### Data Loading Bottlenecks

**Problem**: Data loading can be slower than model computation

### Solutions

1. **Prefetching**: Load next batch while training current batch
1. **Multi-worker Loading**: Parallel data loading (future feature)
1. **Caching**: Cache frequently accessed data in memory

### Transform Performance

**Problem**: Complex transforms slow down data loading

### Solutions

1. **SIMD Optimization**: Vectorize transform operations
1. **Batch Transforms**: Apply transforms to entire batches
1. **Precompute**: Cache transformed data when possible

## Best Practices

### Dataset Design

1. **Keep It Simple**: Implement only essential loading logic in dataset
1. **Lazy Loading**: Don't load all data in `__init__`
1. **Validation**: Validate data integrity during loading
1. **Error Handling**: Gracefully handle missing or corrupted data

### Transform Design

1. **Composability**: Design transforms to be composable
1. **Determinism**: Support reproducible transforms with seed
1. **Type Safety**: Ensure transforms preserve tensor properties
1. **Documentation**: Document expected input/output formats

### Loader Configuration

1. **Batch Size**: Choose based on GPU memory and model size
1. **Shuffle**: Always shuffle training data
1. **Drop Last**: Drop incomplete batches for consistent shapes
1. **Num Workers**: Start with 0, increase if data loading is bottleneck

## Testing

Data processing components should be tested for:

1. **Correctness**: Transforms produce expected outputs
1. **Shape Preservation**: Batch shapes are consistent
1. **Memory Safety**: No leaks or corruption
1. **Performance**: Loading speed meets requirements
1. **Edge Cases**: Empty datasets, single samples, odd batch sizes

See `tests/shared/data/` for comprehensive test suite.

## Integration with Training

Data loaders integrate seamlessly with training loops:

```mojo
from projectodyssey.data import DataLoader
from projectodyssey.training import train_epoch, validate_epoch

# Create loaders
var train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
var val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Training loop
for epoch in range(num_epochs):
    var train_loss = train_epoch(model, optimizer, train_loader, loss_fn)
    var val_loss = validate_epoch(model, val_loader, loss_fn)
```

## Recent Enhancements (Issue #2637)

Completed in the Data Pipeline Architecture redesign:

1. ✅ **Dataset Wrappers**: TransformedDataset for composable transforms
1. ✅ **Caching Layer**: CachedDataset with hit rate tracking and statistics
1. ✅ **Prefetching Infrastructure**: PrefetchDataLoader and PrefetchBuffer
1. ✅ **Unified DataLoader**: BatchLoader exported for consistent usage
1. ✅ **Sampling Strategies**: RandomSampler for epoch shuffling
1. ✅ **Comprehensive Examples**: data_pipeline_demo.mojo showcasing all components

## Future Enhancements

Planned features for future releases:

1. **Async Prefetching**: Upgrade to true async when Mojo adds Task primitives
1. **Multi-worker Data Loading**: Parallel loading with worker processes
1. **Distributed Data Loading**: Sharding for distributed training
1. **Advanced Samplers**: Stratified, cluster, importance sampling
1. **Streaming Datasets**: Support for infinite data streams
1. **SIMD-Optimized Transforms**: Vectorized augmentation operations

## References

- [PyTorch DataLoader](https://pytorch.org/docs/stable/data.html)
- [TensorFlow Dataset API](https://www.tensorflow.org/guide/data)
- [Mojo Performance Guide](https://docs.modular.com/mojo/faq/#performance)

## Contributing

When adding new data processing components:

1. Follow the Dataset or Transform trait interface
1. Add comprehensive tests in `tests/shared/data/`
1. Document usage with examples
1. Optimize hot paths with SIMD
1. Update this README with new components
