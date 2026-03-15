#!/usr/bin/env python3
"""Tests for batch conversion functionality in scripts/convert_image_to_idx.py.

Covers resolve_batch_inputs(), write_idx_images_batch(), and the --batch
CLI flag end-to-end.
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


def _make_png(path: Path, color: int = 128, size: tuple = (32, 32)) -> None:
    """Create a minimal grayscale PNG for testing."""
    img = Image.new("L", size, color=color)
    img.save(path, format="PNG")


def _make_jpeg(path: Path, color: int = 64, size: tuple = (32, 32)) -> None:
    """Create a minimal grayscale JPEG for testing."""
    img = Image.new("L", size, color=color)
    img.save(path, format="JPEG")


def _read_header(path: Path) -> tuple:
    """Read IDX header fields (magic, count, rows, cols) from file."""
    data = path.read_bytes()
    return struct.unpack(">IIII", data[:16])


@pytest.mark.skipif(not PIL_AVAILABLE, reason="Pillow not installed")
class TestResolveBatchInputs(TestCase):
    """Tests for resolve_batch_inputs()."""

    def setUp(self) -> None:
        from convert_image_to_idx import resolve_batch_inputs

        self.resolve_batch_inputs = resolve_batch_inputs
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmpdir.name)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_directory_returns_sorted_png_files(self) -> None:
        """Directory with PNGs returns sorted list of paths."""
        for name in ["c.png", "a.png", "b.png"]:
            _make_png(self.tmp / name)
        result = self.resolve_batch_inputs(str(self.tmp))
        names = [p.name for p in result]
        self.assertEqual(names, ["a.png", "b.png", "c.png"])

    def test_directory_includes_jpeg(self) -> None:
        """Directory with mixed PNG/JPEG returns both types."""
        _make_png(self.tmp / "a.png")
        _make_jpeg(self.tmp / "b.jpg")
        _make_jpeg(self.tmp / "c.jpeg")
        result = self.resolve_batch_inputs(str(self.tmp))
        self.assertEqual(len(result), 3)

    def test_directory_ignores_non_images(self) -> None:
        """Non-image files in directory are excluded."""
        _make_png(self.tmp / "img.png")
        (self.tmp / "notes.txt").write_text("ignored")
        (self.tmp / "data.idx").write_bytes(b"\x00" * 16)
        result = self.resolve_batch_inputs(str(self.tmp))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "img.png")

    def test_glob_pattern_matches_files(self) -> None:
        """Glob pattern expands to matching image files."""
        _make_png(self.tmp / "digit_0.png")
        _make_png(self.tmp / "digit_1.png")
        pattern = str(self.tmp / "digit_*.png")
        result = self.resolve_batch_inputs(pattern)
        self.assertEqual(len(result), 2)

    def test_glob_ignores_non_image_matches(self) -> None:
        """Glob matches that are not image files are excluded."""
        _make_png(self.tmp / "img.png")
        (self.tmp / "img.txt").write_text("not an image")
        pattern = str(self.tmp / "img.*")
        result = self.resolve_batch_inputs(pattern)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "img.png")

    def test_empty_directory_exits(self) -> None:
        """Empty directory causes SystemExit."""
        with self.assertRaises(SystemExit):
            self.resolve_batch_inputs(str(self.tmp))

    def test_no_matching_glob_exits(self) -> None:
        """Glob with no matches causes SystemExit."""
        with self.assertRaises(SystemExit):
            self.resolve_batch_inputs(str(self.tmp / "*.png"))

    def test_returns_path_objects(self) -> None:
        """Result is a list of Path objects."""
        _make_png(self.tmp / "img.png")
        result = self.resolve_batch_inputs(str(self.tmp))
        self.assertIsInstance(result[0], Path)


@pytest.mark.skipif(not PIL_AVAILABLE, reason="Pillow not installed")
class TestWriteIdxImagesBatch(TestCase):
    """Tests for write_idx_images_batch()."""

    def setUp(self) -> None:
        from convert_image_to_idx import write_idx_images_batch

        self.write_idx_images_batch = write_idx_images_batch
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmpdir.name)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _make_images(self, n: int, color: int = 128) -> list:
        """Create N synthetic 28x28 grayscale PIL Images."""
        return [Image.new("L", (28, 28), color=color + i) for i in range(n)]

    def test_file_created(self) -> None:
        """Output file is created after write."""
        out = self.tmp / "batch.idx"
        self.write_idx_images_batch(self._make_images(2), out)
        self.assertTrue(out.exists())

    def test_magic_number(self) -> None:
        """Magic number in header is 2051."""
        out = self.tmp / "batch.idx"
        self.write_idx_images_batch(self._make_images(1), out)
        magic, _, _, _ = _read_header(out)
        self.assertEqual(magic, 2051)

    def test_count_field_matches_n(self) -> None:
        """Count field equals number of images passed."""
        for n in [1, 3, 10]:
            out = self.tmp / f"batch_{n}.idx"
            self.write_idx_images_batch(self._make_images(n), out)
            _, count, _, _ = _read_header(out)
            self.assertEqual(count, n)

    def test_rows_field(self) -> None:
        """Rows field is 28."""
        out = self.tmp / "batch.idx"
        self.write_idx_images_batch(self._make_images(2), out)
        _, _, rows, _ = _read_header(out)
        self.assertEqual(rows, 28)

    def test_cols_field(self) -> None:
        """Cols field is 28."""
        out = self.tmp / "batch.idx"
        self.write_idx_images_batch(self._make_images(2), out)
        _, _, _, cols = _read_header(out)
        self.assertEqual(cols, 28)

    def test_file_size_formula(self) -> None:
        """File size equals 16 + N * 784."""
        for n in [1, 3, 5]:
            out = self.tmp / f"size_{n}.idx"
            self.write_idx_images_batch(self._make_images(n), out)
            self.assertEqual(out.stat().st_size, 16 + n * 784)

    def test_single_image_pixel_data_matches_write_idx_image(self) -> None:
        """Batch with N=1 produces identical pixel data to write_idx_image."""
        from convert_image_to_idx import write_idx_image

        img = Image.new("L", (28, 28), color=100)
        out_single = self.tmp / "single.idx"
        out_batch = self.tmp / "batch1.idx"

        write_idx_image(img, out_single)
        self.write_idx_images_batch([img], out_batch)

        # Both should have identical pixel bytes (offset 16 onward)
        self.assertEqual(out_single.read_bytes()[16:], out_batch.read_bytes()[16:])

    def test_pixel_data_order(self) -> None:
        """Images appear sequentially: first at offset 16, second at 16+784, etc."""
        # Create two images with distinct, uniform pixel values
        img0 = Image.new("L", (28, 28), color=10)
        img1 = Image.new("L", (28, 28), color=200)
        out = self.tmp / "order.idx"
        self.write_idx_images_batch([img0, img1], out)

        data = out.read_bytes()
        first_pixel_block = data[16 : 16 + 784]
        second_pixel_block = data[16 + 784 : 16 + 784 * 2]

        self.assertEqual(first_pixel_block, bytes([10] * 784))
        self.assertEqual(second_pixel_block, bytes([200] * 784))


@pytest.mark.skipif(not PIL_AVAILABLE, reason="Pillow not installed")
class TestBatchMain(TestCase):
    """Integration tests for main() with --batch flag."""

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

    def _populate_dir(self, n: int) -> Path:
        """Create a directory with N PNG images."""
        img_dir = self.tmp / "images"
        img_dir.mkdir()
        for i in range(n):
            _make_png(img_dir / f"digit_{i:02d}.png", color=i * 20)
        return img_dir

    def test_batch_flag_with_directory(self) -> None:
        """--batch with directory produces valid IDX with count=N."""
        img_dir = self._populate_dir(3)
        out = self.tmp / "batch.idx"
        exit_code = self._run_main([str(img_dir), str(out), "--batch"])
        self.assertEqual(exit_code, 0)
        self.assertTrue(out.exists())
        magic, count, rows, cols = _read_header(out)
        self.assertEqual(magic, 2051)
        self.assertEqual(count, 3)
        self.assertEqual(rows, 28)
        self.assertEqual(cols, 28)
        self.assertEqual(out.stat().st_size, 16 + 3 * 784)

    def test_batch_flag_with_glob(self) -> None:
        """--batch with glob pattern produces valid IDX."""
        img_dir = self._populate_dir(2)
        out = self.tmp / "glob.idx"
        pattern = str(img_dir / "*.png")
        exit_code = self._run_main([pattern, str(out), "--batch"])
        self.assertEqual(exit_code, 0)
        _, count, _, _ = _read_header(out)
        self.assertEqual(count, 2)

    def test_missing_directory_exits_nonzero(self) -> None:
        """Nonexistent directory with --batch exits with code 1."""
        out = self.tmp / "out.idx"
        with self.assertRaises(SystemExit) as cm:
            self._run_main([str(self.tmp / "nonexistent"), str(out), "--batch"])
        self.assertNotEqual(cm.exception.code, 0)

    def test_no_emnist_transform_in_batch(self) -> None:
        """--no-emnist-transform is accepted and produces valid batch output."""
        img_dir = self._populate_dir(2)
        out = self.tmp / "no_transform.idx"
        exit_code = self._run_main([str(img_dir), str(out), "--batch", "--no-emnist-transform"])
        self.assertEqual(exit_code, 0)
        _, count, _, _ = _read_header(out)
        self.assertEqual(count, 2)

    def test_single_mode_unchanged(self) -> None:
        """Without --batch, single-image mode works as before."""
        png = self.tmp / "single.png"
        out = self.tmp / "single.idx"
        _make_png(png)
        exit_code = self._run_main([str(png), str(out)])
        self.assertEqual(exit_code, 0)
        magic, count, rows, cols = _read_header(out)
        self.assertEqual(magic, 2051)
        self.assertEqual(count, 1)
        self.assertEqual(rows, 28)
        self.assertEqual(cols, 28)
        self.assertEqual(out.stat().st_size, 800)


if __name__ == "__main__":
    main()
