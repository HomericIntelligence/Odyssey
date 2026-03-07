#!/usr/bin/env python3
"""Tests for scripts/convert_image_to_idx.py.

Verifies IDX header correctness, pixel data, CLI flags,
and error handling using synthesized in-memory images.
"""

import struct
import sys
import tempfile
from pathlib import Path
from unittest import TestCase, main

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

pytestmark = pytest.mark.skipif(not PIL_AVAILABLE, reason="Pillow not installed")


def _make_png(path: Path, size: tuple = (32, 32)) -> None:
    """Create a minimal grayscale PNG for testing."""
    img = Image.new("L", size, color=128)
    img.save(path, format="PNG")


def _make_jpeg(path: Path, size: tuple = (32, 32)) -> None:
    """Create a minimal grayscale JPEG for testing."""
    img = Image.new("L", size, color=64)
    img.save(path, format="JPEG")


@pytest.mark.skipif(not PIL_AVAILABLE, reason="Pillow not installed")
class TestLoadAndPreprocess(TestCase):
    """Tests for load_and_preprocess()."""

    def setUp(self) -> None:
        from convert_image_to_idx import load_and_preprocess

        self.load_and_preprocess = load_and_preprocess
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmpdir.name)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_returns_784_bytes(self) -> None:
        """Output is always exactly 784 bytes (28x28)."""
        png = self.tmp / "img.png"
        _make_png(png)
        result = self.load_and_preprocess(png, emnist_transform=True)
        self.assertEqual(len(result), 784)

    def test_pixel_values_in_range(self) -> None:
        """All pixel values are uint8 [0, 255]."""
        png = self.tmp / "img.png"
        _make_png(png)
        result = self.load_and_preprocess(png, emnist_transform=False)
        for byte in result:
            self.assertGreaterEqual(byte, 0)
            self.assertLessEqual(byte, 255)

    def test_jpeg_accepted(self) -> None:
        """JPEG files are processed identically to PNG."""
        jpg = self.tmp / "img.jpg"
        _make_jpeg(jpg)
        result = self.load_and_preprocess(jpg, emnist_transform=False)
        self.assertEqual(len(result), 784)

    def test_emnist_transform_changes_pixels(self) -> None:
        """EMNIST transform produces different pixel order than no transform."""
        png = self.tmp / "asymmetric.png"
        # Create asymmetric image so transform is detectable
        img = Image.new("L", (32, 32), color=0)
        for i in range(16):
            img.putpixel((i, 0), 255)
        img.save(png)
        with_transform = self.load_and_preprocess(png, emnist_transform=True)
        without_transform = self.load_and_preprocess(png, emnist_transform=False)
        self.assertNotEqual(with_transform, without_transform)


@pytest.mark.skipif(not PIL_AVAILABLE, reason="Pillow not installed")
class TestWriteIdxImage(TestCase):
    """Tests for write_idx_image()."""

    def setUp(self) -> None:
        from convert_image_to_idx import write_idx_image

        self.write_idx_image = write_idx_image
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmpdir.name)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _write_dummy(self) -> Path:
        out = self.tmp / "out.idx"
        pixel_bytes = bytes(784)
        self.write_idx_image(pixel_bytes, out)
        return out

    def test_output_file_created(self) -> None:
        """Output .idx file is created."""
        out = self._write_dummy()
        self.assertTrue(out.exists())

    def test_file_size_is_800_bytes(self) -> None:
        """IDX file is exactly 16-byte header + 784 pixels = 800 bytes."""
        out = self._write_dummy()
        self.assertEqual(out.stat().st_size, 800)

    def test_idx_magic_number(self) -> None:
        """First 4 bytes encode magic number 2051 (0x00000803)."""
        out = self._write_dummy()
        data = out.read_bytes()
        (magic,) = struct.unpack(">I", data[:4])
        self.assertEqual(magic, 2051)

    def test_idx_count(self) -> None:
        """Bytes 4-7 encode count = 1."""
        out = self._write_dummy()
        data = out.read_bytes()
        (count,) = struct.unpack(">I", data[4:8])
        self.assertEqual(count, 1)

    def test_idx_rows(self) -> None:
        """Bytes 8-11 encode rows = 28."""
        out = self._write_dummy()
        data = out.read_bytes()
        (rows,) = struct.unpack(">I", data[8:12])
        self.assertEqual(rows, 28)

    def test_idx_cols(self) -> None:
        """Bytes 12-15 encode cols = 28."""
        out = self._write_dummy()
        data = out.read_bytes()
        (cols,) = struct.unpack(">I", data[12:16])
        self.assertEqual(cols, 28)

    def test_pixel_data_appended(self) -> None:
        """Pixel bytes are written verbatim after the header."""
        out = self.tmp / "out.idx"
        pixel_bytes = bytes(range(256)) * 3 + bytes(16)  # 784 bytes
        self.write_idx_image(pixel_bytes, out)
        data = out.read_bytes()
        self.assertEqual(data[16:], pixel_bytes)


@pytest.mark.skipif(not PIL_AVAILABLE, reason="Pillow not installed")
class TestMain(TestCase):
    """Integration tests for main() via subprocess-free invocation."""

    def setUp(self) -> None:
        from convert_image_to_idx import main

        self.main = main
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmpdir.name)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _run_main(self, args: list) -> int:
        old_argv = sys.argv
        try:
            sys.argv = ["convert_image_to_idx.py"] + args
            return self.main()
        finally:
            sys.argv = old_argv

    def test_end_to_end_png(self) -> None:
        """Full conversion of PNG produces valid 800-byte IDX file."""
        png = self.tmp / "digit.png"
        out = self.tmp / "digit.idx"
        _make_png(png)
        exit_code = self._run_main([str(png), str(out)])
        self.assertEqual(exit_code, 0)
        self.assertTrue(out.exists())
        self.assertEqual(out.stat().st_size, 800)

    def test_end_to_end_jpeg(self) -> None:
        """Full conversion of JPEG produces valid 800-byte IDX file."""
        jpg = self.tmp / "digit.jpg"
        out = self.tmp / "digit.idx"
        _make_jpeg(jpg)
        exit_code = self._run_main([str(jpg), str(out)])
        self.assertEqual(exit_code, 0)
        self.assertEqual(out.stat().st_size, 800)

    def test_missing_input_exits_nonzero(self) -> None:
        """Non-existent input file returns exit code 1."""
        out = self.tmp / "out.idx"
        exit_code = self._run_main([str(self.tmp / "nonexistent.png"), str(out)])
        self.assertEqual(exit_code, 1)
        self.assertFalse(out.exists())

    def test_no_emnist_transform_flag(self) -> None:
        """--no-emnist-transform flag is accepted and produces valid output."""
        png = self.tmp / "digit.png"
        out = self.tmp / "digit.idx"
        _make_png(png)
        exit_code = self._run_main([str(png), str(out), "--no-emnist-transform"])
        self.assertEqual(exit_code, 0)
        self.assertEqual(out.stat().st_size, 800)


if __name__ == "__main__":
    main()
