#!/usr/bin/env python3
"""Tensor Operations Benchmark Suite.

Runs tensor operation benchmarks and outputs results in JSON format.

ADR-001 Justification: Python required for:
- subprocess output capture (Mojo limitation)
- JSON formatting for CI integration
- Cross-platform timing utilities

Usage:
    python benchmarks/tensor-ops/run_benchmark.py --output results.json
    python benchmarks/tensor-ops/run_benchmark.py  # stdout
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# Add scripts directory to path for common utilities
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
from hephaestus.utils import get_repo_root


def run_mojo_benchmark(benchmark_file: Path, timeout: int = 300) -> Dict[str, Any]:
    """Run a Mojo benchmark file and capture timing.

    Args:
        benchmark_file: Path to the benchmark .mojo file.
        timeout: Timeout in seconds.

    Returns:
        Dictionary with benchmark results.
    """
    if not benchmark_file.exists():
        return {
            "name": benchmark_file.stem,
            "status": "not_found",
            "error": f"File not found: {benchmark_file}",
        }

    start_time = time.perf_counter()
    try:
        result = subprocess.run(
            ["uv", "run", "mojo", "run", str(benchmark_file)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=get_repo_root(),
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return {
            "name": benchmark_file.stem,
            "status": "success" if result.returncode == 0 else "failed",
            "duration_ms": round(elapsed_ms, 2),
            "returncode": result.returncode,
            "stdout_lines": len(result.stdout.splitlines()) if result.stdout else 0,
            "stderr_preview": result.stderr[:200] if result.stderr else "",
        }
    except subprocess.TimeoutExpired:
        return {
            "name": benchmark_file.stem,
            "status": "timeout",
            "error": f"Benchmark timed out after {timeout}s",
        }
    except Exception as e:
        return {
            "name": benchmark_file.stem,
            "status": "error",
            "error": str(e),
        }


def run_simple_tensor_benchmarks() -> List[Dict[str, Any]]:
    """Run simple tensor operation benchmarks using direct Mojo execution.

    Returns:
        List of benchmark results.
    """
    results = []
    repo_root = get_repo_root()

    # Define tensor operation benchmarks to run
    benchmarks = [
        {
            "name": "tensor_creation",
            "description": "Create tensors of various sizes",
            "sizes": [(100, 100), (500, 500), (1000, 1000)],
        },
        {
            "name": "tensor_add",
            "description": "Element-wise tensor addition",
            "sizes": [(100, 100), (500, 500), (1000, 1000)],
        },
        {
            "name": "tensor_multiply",
            "description": "Element-wise tensor multiplication",
            "sizes": [(100, 100), (500, 500), (1000, 1000)],
        },
        {
            "name": "matmul",
            "description": "Matrix multiplication",
            "sizes": [(64, 64), (128, 128), (256, 256)],
        },
    ]

    # Check for existing benchmark files
    benchmark_files = {
        "bench_matmul": repo_root / "benchmarks" / "bench_matmul.mojo",
        "bench_simd": repo_root / "benchmarks" / "bench_simd.mojo",
    }

    # Run existing benchmark files
    for name, filepath in benchmark_files.items():
        if filepath.exists():
            print(f"Running {name}...", file=sys.stderr)
            result = run_mojo_benchmark(filepath, timeout=120)
            results.append(result)

    # Add placeholder results for benchmarks not yet implemented
    for bench in benchmarks:
        results.append(
            {
                "name": bench["name"],
                "description": bench["description"],
                "status": "placeholder",
                "sizes": [f"{s[0]}x{s[1]}" for s in bench["sizes"]],
                "note": "Detailed timing to be implemented in future iteration",
            }
        )

    return results


def get_environment_info() -> Dict[str, str]:
    """Get environment information for benchmark reproducibility.

    Returns:
        Dictionary with environment details.
    """
    import platform

    # Get Mojo version
    mojo_version = "unknown"
    try:
        result = subprocess.run(
            ["uv", "run", "mojo", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            mojo_version = result.stdout.strip().split("\n")[0]
    except Exception:
        pass

    # Get git commit
    git_commit = "unknown"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=get_repo_root(),
        )
        if result.returncode == 0:
            git_commit = result.stdout.strip()
    except Exception:
        pass

    return {
        "os": platform.system().lower(),
        "cpu": platform.machine(),
        "python_version": platform.python_version(),
        "mojo_version": mojo_version,
        "git_commit": git_commit,
    }


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run tensor operations benchmark suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output JSON file (default: stdout)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    print("Running tensor-ops benchmark suite...", file=sys.stderr)

    # Run benchmarks
    benchmark_results = run_simple_tensor_benchmarks()

    # Build output structure
    output = {
        "suite": "tensor-ops",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "environment": get_environment_info(),
        "benchmarks": benchmark_results,
        "summary": {
            "total": len(benchmark_results),
            "success": sum(1 for b in benchmark_results if b.get("status") == "success"),
            "failed": sum(1 for b in benchmark_results if b.get("status") == "failed"),
            "placeholder": sum(1 for b in benchmark_results if b.get("status") == "placeholder"),
        },
    }

    # Output results
    json_output = json.dumps(output, indent=2)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json_output)
        print(f"Results written to {args.output}", file=sys.stderr)
    else:
        print(json_output)

    print(f"Benchmark suite complete: {output['summary']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
