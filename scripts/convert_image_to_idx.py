#!/usr/bin/env python3
"""Convert PNG/JPEG image to IDX format for LeNet-5 inference.

ADR-001 Justification: Python required for PIL image decoding
(not available in Mojo v0.26.1 stdlib).
See: docs/adr/ADR-001-language-selection-tooling.md

Usage:
    python scripts/convert_image_to_idx.py input.png output.idx
    python scripts/convert_image_to_idx.py input.jpg output.idx --no-emnist-transform
    python scripts/convert_image_to_idx.py input.png output.idx --preview
"""

import argparse
import struct
import sys
from pathlib import Path

try:
    from PIL import Image, ImageChops
except ImportError:
    print("Error: Pillow not installed. Install with: pip install Pillow")
    sys.exit(1)


def _convert_luma(img: "Image.Image") -> "Image.Image":
    """Convert to grayscale using ITU-R 601 luma weighting (PIL default).

    Formula: 0.299R + 0.587G + 0.114B
    """
    return img.convert("L")


def _convert_average(img: "Image.Image") -> "Image.Image":
    """Convert to grayscale using equal-weight channel average.

    Formula: (R + G + B) / 3
    """
    return img.convert("RGB").convert("L", matrix=(1 / 3, 1 / 3, 1 / 3, 0))


def _convert_max(img: "Image.Image") -> "Image.Image":
    """Convert to grayscale using the brightest (max) channel.

    Formula: max(R, G, B)  — desaturation method.
    """
    rgb = img.convert("RGB")
    r, g, b = rgb.split()
    return ImageChops.lighter(ImageChops.lighter(r, g), b)


_GRAYSCALE_METHODS = {
    "luma": _convert_luma,
    "average": _convert_average,
    "max": _convert_max,
}


def load_and_preprocess(
    image_path: Path,
    emnist_transform: bool,
    grayscale_method: str = "luma",
) -> "Image.Image":
    """Load image and return 28x28 grayscale PIL Image.

    Converts to grayscale, resizes to 28x28, and optionally applies
    the EMNIST transpose+flip transform to align with model weights.

    Args:
        image_path: Path to input PNG or JPEG file.
        emnist_transform: Whether to apply EMNIST transpose+flip.
        grayscale_method: Grayscale conversion strategy — one of
            "luma" (ITU-R 601, default), "average" ((R+G+B)/3),
            or "max" (brightest channel).

    Returns:
        PIL Image object (28x28 grayscale).
    """
    img = Image.open(image_path)
    img = _GRAYSCALE_METHODS[grayscale_method](img)
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    if emnist_transform:
        img = img.transpose(Image.Transpose.TRANSPOSE).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return img


def preview_ascii_art(img: "Image.Image") -> None:
    """Display ASCII art thumbnail of 28x28 image.

    Uses unicode block characters to display a 14x14 character preview
    of the grayscale image. Darker pixels = darker characters.

    Args:
        img: PIL Image object (28x28 grayscale).
    """
    # Downsample to 14x14 for terminal preview (2x2 pixel blocks)
    thumb = img.resize((14, 14), Image.Resampling.BILINEAR)
    pixels = list(thumb.getdata())

    # Unicode block characters from darkest to lightest
    # Using different heights to show intensity variation
    chars = " ▁▂▃▄▅▆▇█"

    print("\n" + "=" * 32)
    print("PREPROCESSED IMAGE PREVIEW (28×28)")
    print("=" * 32)
    for y in range(14):
        row = ""
        for x in range(14):
            pixel_value = pixels[y * 14 + x]
            # Map 0-255 to 0-8 (length of chars string)
            char_idx = min(8, int(pixel_value / 256.0 * 9))
            row += chars[char_idx] + " "
        print(row)
    print("=" * 32 + "\n")


def write_idx_image(img: "Image.Image", output_path: Path) -> None:
    """Write single 28x28 grayscale image in IDX format.

    IDX format (magic=2051, count=1, rows=28, cols=28):
        [4B magic][4B count][4B rows][4B cols][784B pixels]

    Args:
        img: PIL Image object (28x28 grayscale).
        output_path: Destination .idx file path.
    """
    pixel_bytes = bytes(img.getdata())
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
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Display ASCII art preview of preprocessed image",
    )
    parser.add_argument(
        "--grayscale-method",
        choices=["luma", "average", "max"],
        default=None,
        help=(
            "Grayscale conversion strategy: luma (ITU-R 601, default), "
            "average ((R+G+B)/3), max (brightest channel)"
        ),
    )
    args = parser.parse_args()

    grayscale_method = args.grayscale_method if args.grayscale_method is not None else "luma"

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return 1

    img = load_and_preprocess(args.input, not args.no_emnist_transform, grayscale_method)

    if args.preview:
        preview_ascii_art(img)

    write_idx_image(img, args.output)
    print(f"Converted {args.input} -> {args.output} (28x28 grayscale IDX)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
