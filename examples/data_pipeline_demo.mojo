"""Demonstration of efficient data pipeline architecture.

This example showcases the complete data pipeline for ML training:
1. Dataset: In-memory tensor storage
2. Transforms: Data augmentation (Normalize)
3. Sampling: RandomSampler for shuffling
4. Batching: BatchLoader for efficient mini-batches
5. Caching: CachedDataset for small datasets
6. Prefetching: PrefetchDataLoader for pre-computation

The pipeline demonstrates best practices for data loading efficiency
without requiring actual dataset files.

Usage:
    mojo run examples/data_pipeline_demo.mojo

Key Points:
- Modular composition allows mixing components as needed
- Transforms enable augmentation during training
- RandomSampler provides shuffling between epochs
- BatchLoader handles batching with configurable size
- CachedDataset improves I/O for small datasets
- PrefetchDataLoader separates data loading from training

See Also:
    src/projectodyssey/data/README.md - Data pipeline architecture overview
    src/projectodyssey/data/loaders.mojo - BatchLoader implementation
    src/projectodyssey/data/transforms.mojo - Available transforms
"""

from projectodyssey.data.cache import CachedDataset
from projectodyssey.data.dataset_with_transform import TransformedDataset
from projectodyssey.data.datasets import AnyTensorDataset
from projectodyssey.data.loaders import Batch, BatchLoader
from projectodyssey.data.prefetch import PrefetchDataLoader
from projectodyssey.data.samplers import RandomSampler, SequentialSampler
from projectodyssey.data.transforms import Normalize, Compose
from projectodyssey.tensor.any_tensor import AnyTensor, ones, zeros
from std.collections import List


# ============================================================================
# Phase 1: Create Synthetic Dataset
# ============================================================================


def create_demo_dataset(
    num_samples: Int = 32, num_classes: Int = 10
) raises -> Tuple[AnyTensor, AnyTensor]:
    """Create synthetic training data for demonstration.

    Args:
        num_samples: Number of training samples.
        num_classes: Number of classes for labels.

    Returns:
        Tuple of (images, labels) tensors.
    """
    # Create batch of random-like data (all ones for simplicity)
    var image_shape: List[Int] = [num_samples, 3, 32, 32]
    var label_shape: List[Int] = [num_samples, num_classes]

    var images = ones(image_shape, DType.float32)
    var labels = zeros(label_shape, DType.float32)

    # Fill labels with one-hot encoding (sample 0 -> class 0, sample 1 -> class 1, etc.)
    var labels_ptr = labels._data.bitcast[Float32]()
    for i in range(num_samples):
        labels_ptr[i * num_classes + (i % num_classes)] = Float32(1.0)

    print(
        "Creating synthetic dataset: "
        + String(num_samples)
        + " samples, "
        + String(num_classes)
        + " classes"
    )
    return (images, labels)


# ============================================================================
# Phase 2: Basic Data Pipeline (BatchLoader only)
# ============================================================================


def demo_basic_pipeline() raises:
    """Demonstrate basic data loading without transforms or caching."""
    print("\n" + "=" * 70)
    print("Phase 1: Basic Data Pipeline (BatchLoader)")
    print("=" * 70)

    var images, labels = create_demo_dataset(num_samples=16, num_classes=10)

    # Create dataset
    var dataset = AnyTensorDataset(images^, labels^)
    print("✓ Created dataset with " + String(dataset.__len__()) + " samples")

    # Create sampler (Sequential = no shuffling)
    var sampler = SequentialSampler(dataset.__len__())
    print("✓ Created sequential sampler")

    # Create loader
    var loader = BatchLoader(dataset^, sampler^, batch_size=4, drop_last=False)
    print("✓ Created batch loader (batch_size=4)")

    # Iterate and print batch information
    var batches = loader.__iter__()
    print("✓ Generated " + String(batches.__len__()) + " batches")

    for i in range(min(2, batches.__len__())):  # Print first 2 batches
        var batch_data_shape = batches[i].data.shape()[0]
        var batch_labels_shape_0 = batches[i].labels.shape()[0]
        var batch_labels_shape_1 = batches[i].labels.shape()[1]
        var batch_indices_len = batches[i].indices.__len__()
        print(
            "  Batch "
            + String(i)
            + ": data shape="
            + String(batch_data_shape)
            + " samples, labels shape=["
            + String(batch_labels_shape_0)
            + ", "
            + String(batch_labels_shape_1)
            + "], indices="
            + String(batch_indices_len)
        )


# ============================================================================
# Phase 3: Data Pipeline with Transforms
# ============================================================================


def demo_transform_pipeline() raises:
    """Demonstrate pipeline with data augmentation transforms."""
    print("\n" + "=" * 70)
    print("Phase 2: Data Pipeline with Transforms")
    print("=" * 70)

    var images, labels = create_demo_dataset(num_samples=16, num_classes=10)

    # Create base dataset
    var dataset = AnyTensorDataset(images^, labels^)
    print("✓ Created dataset with " + String(dataset.__len__()) + " samples")

    # Create and apply transform
    var normalize = Normalize(mean=0.5, std=0.5)
    var transformed_dataset = TransformedDataset(dataset^, normalize^)
    print("✓ Applied Normalize transform (mean=0.5, std=0.5)")

    # Create sampler and loader
    var sampler = SequentialSampler(transformed_dataset.__len__())
    var loader = BatchLoader(transformed_dataset^, sampler^, batch_size=4)
    print("✓ Created loader with transform pipeline")

    # Verify transform was applied
    var batches = loader.__iter__()
    var batch_shape_0 = batches[0].data.shape()[0]
    var batch_shape_1 = batches[0].data.shape()[1]
    var batch_shape_2 = batches[0].data.shape()[2]
    var batch_shape_3 = batches[0].data.shape()[3]
    print(
        "✓ Batch data after transform: shape=["
        + String(batch_shape_0)
        + ", "
        + String(batch_shape_1)
        + ", "
        + String(batch_shape_2)
        + ", "
        + String(batch_shape_3)
        + "]"
    )


# ============================================================================
# Phase 4: Data Pipeline with Shuffling
# ============================================================================


def demo_shuffling_pipeline() raises:
    """Demonstrate pipeline with random sampling and shuffling."""
    print("\n" + "=" * 70)
    print("Phase 3: Data Pipeline with Shuffling")
    print("=" * 70)

    var images, labels = create_demo_dataset(num_samples=16, num_classes=10)

    # Create dataset
    var dataset = AnyTensorDataset(images^, labels^)
    print("✓ Created dataset with " + String(dataset.__len__()) + " samples")

    # Create sampler with shuffling
    var sampler = RandomSampler(dataset.__len__())
    print("✓ Created random sampler (enables shuffling)")

    # Create loader
    var loader = BatchLoader(dataset^, sampler^, batch_size=4, drop_last=False)

    # Simulate multiple epochs to show shuffling
    print("✓ Simulating multiple epochs:")
    for epoch in range(2):
        var batches = loader.__iter__()
        print(
            "  Epoch "
            + String(epoch)
            + ": "
            + String(batches.__len__())
            + " batches"
        )


# ============================================================================
# Phase 5: Complete Pipeline with All Components
# ============================================================================


def demo_complete_pipeline() raises:
    """Demonstrate complete pipeline with all components combined."""
    print("\n" + "=" * 70)
    print("Phase 4: Complete Pipeline (All Components)")
    print("=" * 70)

    var images, labels = create_demo_dataset(num_samples=32, num_classes=10)

    # Step 1: Create dataset
    var dataset = AnyTensorDataset(images^, labels^)
    print(
        "✓ Step 1: Created dataset (" + String(dataset.__len__()) + " samples)"
    )

    # Step 2: Apply transforms
    var normalize = Normalize(mean=0.5, std=0.5)
    var transformed = TransformedDataset(dataset^, normalize^)
    print("✓ Step 2: Applied Normalize transform")

    # Step 3: Create sampler with shuffling
    var sampler = RandomSampler(transformed.__len__())
    print("✓ Step 3: Created random sampler")

    # Step 4: Create batch loader
    var loader = BatchLoader(
        transformed^, sampler^, batch_size=4, drop_last=False
    )
    print("✓ Step 4: Created batch loader (batch_size=4)")

    # Step 5: Create prefetch loader
    var prefetch = PrefetchDataLoader(loader^, prefetch_factor=2)
    print("✓ Step 5: Created prefetch loader (prefetch_factor=2)")

    # Simulate training with prefetching
    print("✓ Simulating training iteration:")
    for epoch in range(2):
        var batches = prefetch.__iter__()
        print("  Epoch " + String(epoch) + ":")
        print("    - Total batches: " + String(batches.__len__()))


# ============================================================================
# Phase 6: Performance Comparison
# ============================================================================


def demo_performance_comparison() raises:
    """Demonstrate performance characteristics of different configurations."""
    print("\n" + "=" * 70)
    print("Phase 5: Performance Characteristics")
    print("=" * 70)

    # Configuration 1: No caching
    print("\nConfiguration 1: Sequential, no caching, no transforms")
    var images1, labels1 = create_demo_dataset(num_samples=64, num_classes=10)
    var dataset1 = AnyTensorDataset(images1^, labels1^)
    var sampler1 = SequentialSampler(dataset1.__len__())
    var loader1 = BatchLoader(dataset1^, sampler1^, batch_size=16)
    var batches1 = loader1.__iter__()
    print("  ✓ " + String(batches1.__len__()) + " batches created")

    # Configuration 2: With transforms
    print("\nConfiguration 2: Sequential with transforms")
    var images2, labels2 = create_demo_dataset(num_samples=64, num_classes=10)
    var dataset2 = AnyTensorDataset(images2^, labels2^)
    var normalize2 = Normalize(mean=0.5, std=0.5)
    var transformed2 = TransformedDataset(dataset2^, normalize2^)
    var sampler2 = SequentialSampler(transformed2.__len__())
    var loader2 = BatchLoader(transformed2^, sampler2^, batch_size=16)
    var batches2 = loader2.__iter__()
    print(
        "  ✓ " + String(batches2.__len__()) + " batches created with transforms"
    )

    # Configuration 3: With shuffling and caching
    print("\nConfiguration 3: Random sampling with caching")
    var images3, labels3 = create_demo_dataset(num_samples=64, num_classes=10)
    var dataset3 = AnyTensorDataset(images3^, labels3^)
    var cached3 = CachedDataset(dataset3^, max_cache_size=-1)
    var sampler3 = RandomSampler(cached3.__len__())
    var loader3 = BatchLoader(cached3^, sampler3^, batch_size=16)
    var batches3 = loader3.__iter__()
    print(
        "  ✓ "
        + String(batches3.__len__())
        + " batches created with shuffling & caching"
    )

    print("\nConfiguration Summary:")
    print(
        "  All configurations produce "
        + String(batches1.__len__())
        + " batches"
    )
    print("  Config 1: Simple sequential pipeline")
    print("  Config 2: Adds augmentation (transforms)")
    print("  Config 3: Adds shuffling & caching for better training")


# ============================================================================
# Main Entry Point
# ============================================================================


def main() raises:
    """Run all pipeline demonstrations."""
    print("=" * 70)
    print("ML Data Pipeline Architecture Demonstration")
    print("=" * 70)
    print("\nThis example showcases best practices for efficient data loading:")
    print("  - Modular components (Dataset, Sampler, BatchLoader)")
    print("  - Data augmentation (Transforms)")
    print("  - Batch shuffling (RandomSampler)")
    print("  - Sample caching (CachedDataset)")
    print("  - Batch prefetching (PrefetchDataLoader)")

    # Run demonstrations
    demo_basic_pipeline()
    demo_transform_pipeline()
    demo_shuffling_pipeline()
    demo_complete_pipeline()
    demo_performance_comparison()

    print("\n" + "=" * 70)
    print("Summary: Efficient Data Pipeline Architecture")
    print("=" * 70)
    print(
        """
Key Takeaways:
1. Compose components (Dataset -> Transform -> Cache -> Sampler -> Loader)
2. Use RandomSampler for training (enables epoch shuffling)
3. Apply transforms for augmentation during loading
4. Cache small datasets to avoid repeated I/O
5. Use prefetch to separate data prep from training

Architecture Benefits:
- Clean separation of concerns
- Easy to add/remove components
- Reusable building blocks
- Type-safe interfaces
- Ready for async upgrade when Mojo adds Task primitives

Next Steps:
- Use BatchLoader in training scripts instead of extract_batch_pair()
- Apply transforms in data pipelines
- Enable shuffling with RandomSampler
- Cache datasets that fit in memory
    """
    )

    print("=" * 70)
    print("Demo complete!")
    print("=" * 70)
