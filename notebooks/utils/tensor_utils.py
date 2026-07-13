"""Tensor conversion utilities between Mojo and NumPy.

Handles serialization/deserialization of tensors for
Mojo-Python interop.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import struct


def numpy_to_mojo_binary(
    array: np.ndarray,
    path: str,
) -> None:
    """Save NumPy array in Mojo-compatible binary format.

    Args:
        array: NumPy array to save
        path: Path to output binary file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save as raw binary data (Mojo can read with fromfile)
    with open(path, "wb") as f:
        array.astype(np.float32).tobytes()


def mojo_binary_to_numpy(
    path: str,
    shape: Tuple[int, ...],
    dtype: str = "float32",
) -> np.ndarray:
    """Load Mojo binary tensor as NumPy array.

    Args:
        path: Path to binary file
        shape: Shape of the tensor
        dtype: Data type string ('float32', 'float16', etc.)

    Returns:
        NumPy array with loaded data

    Raises:
        FileNotFoundError: If file does not exist
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Tensor file not found: {path}")

    dtype_map = {
        "float32": np.float32,
        "float16": np.float16,
        "int32": np.int32,
        "int64": np.int64,
        "uint8": np.uint8,
        "f32": np.float32,
        "f16": np.float16,
        "i32": np.int32,
    }

    np_dtype = dtype_map.get(dtype, np.float32)
    data = np.fromfile(file_path, dtype=np_dtype)
    return data.reshape(shape)


def save_tensor_to_json(array: np.ndarray, path: str) -> None:
    """Save tensor metadata as JSON (shape, dtype, min/max).

    Useful for saving alongside binary tensor files.

    Args:
        array: NumPy array to save metadata for
        path: Path to output JSON file
    """
    import json

    metadata = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "dtype_name": array.dtype.name,
        "size": int(np.prod(array.shape)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
    }

    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)


def load_tensor_metadata(path: str) -> dict:
    """Load tensor metadata from JSON file.

    Args:
        path: Path to metadata JSON file

    Returns:
        Dict with tensor metadata
    """
    import json

    with open(path) as f:
        return json.load(f)


def compare_tensors(
    a: np.ndarray,
    b: np.ndarray,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> dict:
    """Compare two tensors and return statistics.

    Useful for debugging Mojo vs Python implementations.

    Args:
        a: First tensor
        b: Second tensor
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        Dict with comparison statistics
    """
    if a.shape != b.shape:
        return {
            "equal": False,
            "error": f"Shape mismatch: {a.shape} vs {b.shape}",
        }

    diff = np.abs(a - b)
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))
    num_diff = int(np.sum(diff > atol))

    is_close = np.allclose(a, b, rtol=rtol, atol=atol)

    return {
        "equal": is_close,
        "max_difference": max_diff,
        "mean_difference": mean_diff,
        "num_different_elements": num_diff,
        "total_elements": int(np.prod(a.shape)),
        "percent_different": (100.0 * num_diff / np.prod(a.shape) if np.prod(a.shape) > 0 else 0.0),
    }


def parse_mojo_tensor_output(output: str) -> Optional[np.ndarray]:
    """Parse tensor from Mojo stdout output.

    Assumes Mojo printed tensor in format like:
    [[1.0, 2.0], [3.0, 4.0]]

    Args:
        output: Stdout from Mojo script

    Returns:
        NumPy array if parsing succeeds, None otherwise
    """
    try:
        # Find array-like text in output
        lines = output.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("[[") or line.startswith("["):
                # Try to parse as list
                tensor_list = eval(line)  # Use ast.literal_eval for safety in production
                return np.array(tensor_list)
        return None
    except Exception:
        return None
