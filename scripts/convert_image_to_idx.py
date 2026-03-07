#!/usr/bin/env python3
"""Convert PNG/JPEG image to IDX format for LeNet-5 inference.

ADR-001 Justification: Python required for PIL image decoding
(not available in Mojo v0.26.1 stdlib).
See: docs/adr/ADR-001-language-selection-tooling.md

Usage:
    python scripts/convert_image_to_idx.py input.png output.idx
    python scripts/convert_image_to_idx.py input.jpg output.idx --no-emnist-transform
"""

import argparse
import struct
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("Error: Pillow not installed. Install with: pip install Pillow")
    sys.exit(1)


def load_and_preprocess(image_path: Path, emnist_transform: bool) -> bytes:
    """Load image and return 28x28 uint8 grayscale pixel bytes.

    Converts to grayscale, resizes to 28x28, and optionally applies
    the EMNIST transpose+flip transform to align with model weights.

    Args:
        image_path: Path to input PNG or JPEG file.
        emnist_transform: Whether to apply EMNIST transpose+flip.

    Returns:
        784 bytes of uint8 pixel values in row-major order.
    """
    img = Image.open(image_path).convert("L")
    img = img.resize((28, 28), Image.LANCZOS)
    if emnist_transform:
        img = img.transpose(Image.TRANSPOSE).transpose(Image.FLIP_LEFT_RIGHT)
    return bytes(img.getdata())


def write_idx_image(pixel_bytes: bytes, output_path: Path) -> None:
    """Write single 28x28 grayscale image in IDX format.

    IDX format (magic=2051, count=1, rows=28, cols=28):
        [4B magic][4B count][4B rows][4B cols][784B pixels]

    Args:
        pixel_bytes: 784 raw uint8 pixel bytes.
        output_path: Destination .idx file path.
    """
    header = struct.pack(">IIII", 2051, 1, 28, 28)
    output_path.write_bytes(header + pixel_bytes)


def main() -> int:
    """Parse arguments and run conversion. Returns exit code."""
    parser = argparse.ArgumentParser(description="Convert PNG/JPEG to IDX format for run_infer.mojo")
    parser.add_argument("input", type=Path, help="Input PNG or JPEG file")
    parser.add_argument("output", type=Path, help="Output .idx file")
    parser.add_argument(
        "--no-emnist-transform",
        action="store_true",
        help="Skip EMNIST transpose+flip (use for non-EMNIST models)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return 1

    pixel_bytes = load_and_preprocess(args.input, not args.no_emnist_transform)
    write_idx_image(pixel_bytes, args.output)
    print(f"Converted {args.input} -> {args.output} (28x28 grayscale IDX)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
