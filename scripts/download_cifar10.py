#!/usr/bin/env python3
"""
Download and prepare CIFAR-10 dataset for ML Odyssey AlexNet example.

This script:
1. Downloads CIFAR-10 dataset (Python version) from official source
2. Extracts the tar.gz archive
3. Converts Python pickle batches to IDX format for Mojo compatibility
4. Normalizes images and saves in ML Odyssey format

Usage:
    python examples/alexnet-cifar10/download_cifar10.py [--output-dir datasets/cifar10]

Requirements:
    - Python 3.7+
    - numpy

References:
    - CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html
    - Download: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
"""

import argparse
import pickle
import struct
import sys
import tarfile
from pathlib import Path
from typing import Tuple
from urllib.request import urlretrieve

try:
    import numpy as np
except ImportError:
    print("Error: numpy is required. Install with: pip install numpy")
    sys.exit(1)


# CIFAR-10 dataset URL
CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_MD5 = "c58f30108f718f92721af3b95e74349a"

# CIFAR-10 structure
TRAIN_BATCHES = [
    "data_batch_1",
    "data_batch_2",
    "data_batch_3",
    "data_batch_4",
    "data_batch_5",
]
TEST_BATCH = "test_batch"


def download_progress(block_num: int, block_size: int, total_size: int):
    """Display download progress."""
    downloaded = block_num * block_size
    percent = min(100, downloaded * 100 / total_size)
    bar_length = 50
    filled = int(bar_length * downloaded / total_size)
    bar = "=" * filled + "-" * (bar_length - filled)
    print(f"\rDownloading: [{bar}] {percent:.1f}%", end="", flush=True)


def download_cifar10(output_dir: Path) -> Path:
    """Download CIFAR-10 dataset.

    Args:
        output_dir: Directory to save downloaded files

    Returns:
        Path to downloaded tar.gz file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tar_path = output_dir / "cifar-10-python.tar.gz"

    if tar_path.exists():
        print(f"CIFAR-10 archive already exists: {tar_path}")
        return tar_path

    print(f"Downloading CIFAR-10 from {CIFAR10_URL}...")
    urlretrieve(CIFAR10_URL, tar_path, reporthook=download_progress)
    print("\nDownload complete!")

    return tar_path


def extract_cifar10(tar_path: Path, output_dir: Path) -> Path:
    """Extract CIFAR-10 tar.gz archive.

    Args:
        tar_path: Path to tar.gz file
        output_dir: Directory to extract to

    Returns:
        Path to extracted directory
    """
    extract_dir = output_dir / "cifar-10-batches-py"

    if extract_dir.exists():
        print(f"CIFAR-10 already extracted: {extract_dir}")
        return extract_dir

    print(f"Extracting {tar_path}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(output_dir)
    print("Extraction complete!")

    return extract_dir


def load_cifar10_batch(filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load a single CIFAR-10 pickle batch.

    Args:
        filepath: Path to batch file

    Returns:
        Tuple of (images, labels):
        - images: (N, 3, 32, 32) uint8 array
        - labels: (N,) uint8 array

    Note:
        CIFAR-10 pickle format:
        - dict['data']: (10000, 3072) uint8 array (3072 = 32*32*3)
        - dict['labels']: (10000,) list of ints
        - Pixel order: R channel, G channel, B channel (all 1024 pixels each).
    """
    with open(filepath, "rb") as f:
        batch = pickle.load(f, encoding="bytes")

    # Extract images and labels
    images_flat = batch[b"data"]  # (10000, 3072)
    labels = batch[b"labels"]  # (10000,)

    # Reshape to (N, 3, 32, 32)
    images = images_flat.reshape(-1, 3, 32, 32)
    labels = np.array(labels, dtype=np.uint8)

    return images, labels


def save_idx_labels(labels: np.ndarray, filepath: Path):
    """Save labels in IDX format.

    Args:
        labels: (N,) uint8 array
        filepath: Output file path

    IDX Format:
        [magic(4B)][count(4B)][label_data...]
        Magic number: 2049 (0x00000801).
    """
    magic = 2049
    count = len(labels)

    with open(filepath, "wb") as f:
        # Write header
        f.write(struct.pack(">I", magic))  # Magic number (big-endian)
        f.write(struct.pack(">I", count))  # Number of items

        # Write label data
        f.write(labels.tobytes())

    print(f"  Saved labels: {filepath} ({count} labels)")


def save_idx_images_rgb(images: np.ndarray, filepath: Path):
    """Save RGB images in IDX format.

    Args:
        images: (N, 3, 32, 32) uint8 array
        filepath: Output file path

    IDX Format:
        [magic(4B)][count(4B)][channels(4B)][rows(4B)][cols(4B)][pixel_data...]
        Magic number: 2052 (custom extension for RGB).
    """
    magic = 2052  # Custom magic for RGB images
    count, channels, rows, cols = images.shape

    with open(filepath, "wb") as f:
        # Write header
        f.write(struct.pack(">I", magic))  # Magic number
        f.write(struct.pack(">I", count))  # Number of images
        f.write(struct.pack(">I", channels))  # Channels (3 for RGB)
        f.write(struct.pack(">I", rows))  # Image height
        f.write(struct.pack(">I", cols))  # Image width

        # Write image data
        f.write(images.tobytes())

    print(f"  Saved images: {filepath} ({count} images, {channels}×{rows}×{cols})")


def convert_to_idx(batch_dir: Path, output_dir: Path):
    """Convert CIFAR-10 pickle batches to IDX format.

    Args:
        batch_dir: Directory containing CIFAR-10 pickle batches
        output_dir: Directory to save IDX files
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nConverting CIFAR-10 batches to IDX format...")

    # Convert training batches
    for i, batch_name in enumerate(TRAIN_BATCHES, start=1):
        print(f"\nProcessing training batch {i}/5: {batch_name}")
        images, labels = load_cifar10_batch(batch_dir / batch_name)

        # Save as IDX
        save_idx_images_rgb(images, output_dir / f"train_batch_{i}_images.idx")
        save_idx_labels(labels, output_dir / f"train_batch_{i}_labels.idx")

    # Convert test batch
    print(f"\nProcessing test batch: {TEST_BATCH}")
    images, labels = load_cifar10_batch(batch_dir / TEST_BATCH)

    save_idx_images_rgb(images, output_dir / "test_batch_images.idx")
    save_idx_labels(labels, output_dir / "test_batch_labels.idx")

    print("\nConversion complete!")


def verify_dataset(output_dir: Path):
    """Verify that all required IDX files exist.

    Args:
        output_dir: Directory containing IDX files
    """
    print("\nVerifying dataset...")

    required_files = []
    for i in range(1, 6):
        required_files.append(f"train_batch_{i}_images.idx")
        required_files.append(f"train_batch_{i}_labels.idx")
    required_files.append("test_batch_images.idx")
    required_files.append("test_batch_labels.idx")

    missing = []
    for filename in required_files:
        filepath = output_dir / filename
        if not filepath.exists():
            missing.append(filename)

    if missing:
        print(f"Error: Missing files: {missing}")
        sys.exit(1)

    print("✓ All dataset files present")
    print(f"\nDataset ready at: {output_dir}")
    print("\nTo train AlexNet:")
    print("  mojo run examples/alexnet-cifar10/train.mojo --epochs 100 --batch-size 128 --lr 0.01")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Download and prepare CIFAR-10 dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets/cifar10",
        help="Output directory for dataset (default: datasets/cifar10)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("CIFAR-10 Dataset Download and Preparation")
    print("=" * 60)
    print()

    # Step 1: Download
    tar_path = download_cifar10(output_dir)

    # Step 2: Extract
    batch_dir = extract_cifar10(tar_path, output_dir)

    # Step 3: Convert to IDX
    idx_dir = output_dir
    convert_to_idx(batch_dir, idx_dir)

    # Step 4: Verify
    verify_dataset(idx_dir)

    print("\n" + "=" * 60)
    print("CIFAR-10 dataset preparation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
