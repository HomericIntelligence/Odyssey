"""Bridge for calling Mojo code from Jupyter notebooks.

This module provides utilities for:
- Compiling and running Mojo scripts
- Capturing output and parsing results
- Loading model weights and tensors
"""

import subprocess
import json
from pathlib import Path
from typing import Optional, Dict, Any, List


def run_mojo_script(
    script_path: str,
    args: Optional[List[str]] = None,
    capture_output: bool = True,
    timeout: int = 300,
) -> Dict[str, Any]:
    """Run a Mojo script and capture its output.

    Args:
        script_path: Path to .mojo file
        args: Command line arguments to pass to the script
        capture_output: Whether to capture stdout/stderr
        timeout: Execution timeout in seconds

    Returns:
        Dict with 'returncode', 'stdout', 'stderr'

    Raises:
        FileNotFoundError: If script does not exist
        subprocess.TimeoutExpired: If execution exceeds timeout
    """
    script = Path(script_path)
    if not script.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    cmd = ["uv", "run", "mojo", "run", "-I", ".", str(script)]
    if args:
        cmd.extend(args)

    try:
        result = subprocess.run(
            cmd,
            capture_output=capture_output,
            text=True,
            timeout=timeout,
            cwd=Path.cwd(),
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0,
        }
    except subprocess.TimeoutExpired as e:
        return {
            "returncode": -1,
            "stdout": e.stdout or "",
            "stderr": f"Timeout after {timeout}s",
            "success": False,
        }


def compile_mojo_binary(
    source_path: str,
    output_path: str,
    flags: Optional[List[str]] = None,
    timeout: int = 300,
) -> bool:
    """Compile a Mojo file to binary.

    Args:
        source_path: Path to .mojo source file
        output_path: Path for output binary
        flags: Additional compiler flags
        timeout: Compilation timeout in seconds

    Returns:
        True if compilation succeeded, False otherwise
    """
    source = Path(source_path)
    if not source.exists():
        print(f"Error: Source file not found: {source_path}")
        return False

    cmd = ["uv", "run", "mojo", "build", "-I", ".", str(source), "-o", output_path]
    if flags:
        cmd.extend(flags)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Path.cwd(),
        )
        if result.returncode != 0:
            print(f"Compilation failed: {result.stderr}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"Compilation timeout after {timeout}s")
        return False


def load_tensor_from_file(path: str, shape: tuple, dtype: str = "float32"):
    """Load a tensor saved by Mojo in binary format.

    Args:
        path: Path to binary file
        shape: Shape of the tensor
        dtype: Data type ('float32', 'float16', 'int32', etc)

    Returns:
        NumPy array with loaded data

    Raises:
        FileNotFoundError: If file does not exist
    """
    import numpy as np

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Tensor file not found: {path}")

    # Map Mojo dtype strings to NumPy dtypes
    dtype_map = {
        "float32": np.float32,
        "float16": np.float16,
        "int32": np.int32,
        "int64": np.int64,
        "uint8": np.uint8,
    }

    np_dtype = dtype_map.get(dtype, np.float32)
    data = np.fromfile(file_path, dtype=np_dtype)
    return data.reshape(shape)


def parse_json_from_mojo_output(output: str) -> Dict[str, Any]:
    """Parse JSON from Mojo stdout output.

    Useful for getting structured data from Mojo scripts
    that print JSON.

    Args:
        output: Stdout from Mojo script

    Returns:
        Parsed JSON dict, or empty dict if parsing fails
    """
    try:
        # Find JSON in output (it might be surrounded by other text)
        lines = output.split("\n")
        for line in lines:
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
        return {}
    except Exception:
        return {}
