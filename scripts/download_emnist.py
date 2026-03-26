#!/usr/bin/env python3
"""
Download and extract EMNIST dataset.

This script downloads the EMNIST dataset in MATLAB format and extracts it
to the datasets/emnist/ directory.

ADR-001 Justification: Python required for:
- subprocess for downloading with wget/curl
- gzip/tarfile for extraction
- No Mojo stdlib support for these operations yet

Usage:
    python scripts/download_emnist.py [--split SPLIT]

References:
    - EMNIST Dataset: https://www.nist.gov/itl/products-and-services/emnist-dataset
    - Paper: Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017).
             EMNIST: an extension of MNIST to handwritten letters.
             arXiv:1702.05373v1
"""

import argparse
import gzip
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

# Enable importing from scripts/common.py
sys.path.insert(0, str(Path(__file__).parent))
from common import get_repo_root
from typing import Optional, Tuple

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None  # type: ignore[assignment]

# EMNIST download URLs (with fallbacks)
# Primary URL (NIST official - may have availability issues from GitHub Actions)
EMNIST_PRIMARY_URL = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"

# Fallback mirrors (if primary fails - especially for CI environments)
EMNIST_FALLBACK_URLS = [
    # Kaggle mirror (alternative source)
    "https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip",
]

# All URLs to try in order
EMNIST_URLS = [EMNIST_PRIMARY_URL] + EMNIST_FALLBACK_URLS

# User-Agent header to avoid bot blocking (especially from NIST server)
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAYS = [1, 2, 4]  # Exponential backoff: 1s, 2s, 4s

# Available splits
EMNIST_SPLITS = [
    "balanced",  # 131,600 chars, 47 balanced classes (recommended)
    "byclass",  # 814,255 chars, 62 unbalanced classes
    "bymerge",  # 814,255 chars, 47 unbalanced classes
    "digits",  # 280,000 chars, 10 balanced classes
    "letters",  # 145,600 chars, 26 balanced classes
    "mnist",  # 70,000 chars, 10 balanced classes
]


def download_file_with_retry(url: str, output_path: Path, max_retries: int = MAX_RETRIES) -> None:
    """
    Download file using wget or curl with retry logic and User-Agent header.

    Uses exponential backoff for retries to handle transient failures
    (especially 403 errors from NIST server in GitHub Actions).

    Args:
        url: URL to download from
        output_path: Path to save downloaded file
        max_retries: Maximum number of retry attempts

    Raises:
        RuntimeError: If download fails after all retries
    """
    last_error: Optional[str] = None

    for attempt in range(max_retries):
        if attempt > 0:
            delay = RETRY_DELAYS[min(attempt - 1, len(RETRY_DELAYS) - 1)]
            print(f"  Retry {attempt}/{max_retries - 1} after {delay}s delay...")
            time.sleep(delay)

        print(f"Downloading {url}..." + (f" (attempt {attempt + 1}/{max_retries})" if attempt > 0 else ""))

        # Try wget first with User-Agent header
        result = subprocess.run(
            ["wget", "-q", "-O", str(output_path), f"--user-agent={USER_AGENT}", url],
            capture_output=True,
        )

        if result.returncode == 0:
            print(f"Downloaded to {output_path}")
            return

        # Fall back to curl with User-Agent header
        result = subprocess.run(
            ["curl", "-s", "-L", "-A", USER_AGENT, "-o", str(output_path), url],
            capture_output=True,
        )

        if result.returncode == 0:
            print(f"Downloaded to {output_path}")
            return

        # Check for specific error codes
        stderr = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""
        last_error = f"wget/curl failed with exit code {result.returncode}"
        if "403" in stderr or "Forbidden" in stderr:
            last_error = "HTTP 403 Forbidden (server may be blocking automated requests)"
        elif "404" in stderr or "Not Found" in stderr:
            last_error = "HTTP 404 Not Found"
        elif "Connection refused" in stderr:
            last_error = "Connection refused"

        print(f"  Download failed: {last_error}")

    raise RuntimeError(f"Failed to download {url} after {max_retries} attempts. Last error: {last_error}")


def extract_gzip(gzip_path: Path, output_dir: Path) -> None:
    """Extract gzip.zip file containing EMNIST binary files."""
    print(f"Extracting {gzip_path}...")

    # First unzip the .zip file
    result = subprocess.run(
        ["unzip", "-q", "-o", str(gzip_path), "-d", str(output_dir)],
        capture_output=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Failed to extract {gzip_path}")

    # Then extract all .gz files in the gzip directory
    gzip_dir = output_dir / "gzip"
    if gzip_dir.exists():
        for gz_file in gzip_dir.glob("*.gz"):
            print(f"Extracting {gz_file.name}...")
            with gzip.open(gz_file, "rb") as f_in:
                output_file = output_dir / gz_file.stem
                with open(output_file, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

    print(f"Extracted to {output_dir}")


def read_idx_labels(filename: Path):
    """Read IDX file format for labels."""
    if not HAS_NUMPY:
        raise ImportError("NumPy is required for dataset verification. Install with: pip install numpy")

    with open(filename, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        if magic != 2049:  # Label file magic number
            raise ValueError(f"Invalid magic number in {filename}")

        num_items = struct.unpack(">I", f.read(4))[0]
        labels = np.frombuffer(f.read(), dtype=np.uint8)

        if len(labels) != num_items:
            raise ValueError(f"Label count mismatch in {filename}")

        return labels


def read_idx_images(filename: Path):
    """Read IDX file format for images."""
    if not HAS_NUMPY:
        raise ImportError("NumPy is required for dataset verification. Install with: pip install numpy")

    with open(filename, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        if magic != 2051:  # Image file magic number
            raise ValueError(f"Invalid magic number in {filename}")

        num_images = struct.unpack(">I", f.read(4))[0]
        num_rows = struct.unpack(">I", f.read(4))[0]
        num_cols = struct.unpack(">I", f.read(4))[0]

        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, num_rows, num_cols)

        return images


def verify_dataset(data_dir: Path, split: str) -> Tuple[int, int, int]:
    """Verify dataset was extracted correctly and return stats."""
    train_images = data_dir / f"emnist-{split}-train-images-idx3-ubyte"
    train_labels = data_dir / f"emnist-{split}-train-labels-idx1-ubyte"
    test_images = data_dir / f"emnist-{split}-test-images-idx3-ubyte"
    test_labels = data_dir / f"emnist-{split}-test-labels-idx1-ubyte"

    if not all(f.exists() for f in [train_images, train_labels, test_images, test_labels]):
        raise FileNotFoundError(f"Missing dataset files for split '{split}'")

    # Read and verify
    train_imgs = read_idx_images(train_images)
    train_lbls = read_idx_labels(train_labels)
    test_imgs = read_idx_images(test_images)
    test_lbls = read_idx_labels(test_labels)

    if len(train_imgs) != len(train_lbls):
        raise ValueError("Training images and labels count mismatch")

    if len(test_imgs) != len(test_lbls):
        raise ValueError("Test images and labels count mismatch")

    num_classes = len(np.unique(np.concatenate([train_lbls, test_lbls])))

    print("\nDataset verification successful:")
    print(f"  Split: {split}")
    print(f"  Training samples: {len(train_imgs)}")
    print(f"  Test samples: {len(test_imgs)}")
    print(f"  Image shape: {train_imgs.shape[1:]} (height x width)")
    print(f"  Number of classes: {num_classes}")

    return len(train_imgs), len(test_imgs), num_classes


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Download and extract EMNIST dataset")
    parser.add_argument(
        "--split",
        type=str,
        default="balanced",
        choices=EMNIST_SPLITS,
        help="EMNIST split to use (default: balanced)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=get_repo_root() / "datasets" / "emnist",
        help="Output directory for dataset (default: datasets/emnist/)",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Check if dataset already exists
    train_images = args.output_dir / f"emnist-{args.split}-train-images-idx3-ubyte"
    if train_images.exists():
        print(f"Dataset already exists in {args.output_dir}")
        try:
            verify_dataset(args.output_dir, args.split)
            print("\nDataset is ready to use!")
            return 0
        except Exception as e:
            print(f"Verification failed: {e}")
            print("Re-downloading dataset...")

    # Download gzip.zip (try all URLs until one works)
    gzip_zip = args.output_dir / "gzip.zip"
    download_success = False
    last_error = None

    for url in EMNIST_URLS:
        try:
            download_file_with_retry(url, gzip_zip)
            download_success = True
            break
        except Exception as e:
            last_error = e
            print(f"Failed to download from {url}: {e}")
            print("Trying next mirror...")
            continue

    if not download_success:
        print("Error: All download URLs failed.", file=sys.stderr)
        print(f"Last error: {last_error}", file=sys.stderr)
        print("\nTroubleshooting:", file=sys.stderr)
        print("  1. Check your internet connection", file=sys.stderr)
        print("  2. Verify wget or curl is installed", file=sys.stderr)
        print("  3. Try downloading manually from:", file=sys.stderr)
        for url in EMNIST_URLS:
            print(f"     - {url}", file=sys.stderr)
        return 1

    try:
        extract_gzip(gzip_zip, args.output_dir)

        # Verify dataset
        verify_dataset(args.output_dir, args.split)

        # Cleanup
        gzip_zip.unlink(missing_ok=True)
        gzip_dir = args.output_dir / "gzip"
        if gzip_dir.exists():
            shutil.rmtree(gzip_dir)

        print(f"\nDataset successfully downloaded to {args.output_dir}")
        print("\nUsage:")
        print("  Training: examples/lenet_emnist/train.mojo")
        print("  Inference: examples/lenet_emnist/inference.mojo")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
