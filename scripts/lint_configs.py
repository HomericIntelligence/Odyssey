#!/usr/bin/env python3
"""CLI for linting ML Odyssey YAML configuration files.

The YAML-linting logic now lives in ``hephaestus.validation.config_lint``
(issue #5061). This script is a thin CLI wrapper that supplies ML Odyssey's
repo-specific lint rules (deprecated keys, required keys, performance
thresholds) and drives `hephaestus`'s `ConfigLinter` over the given paths.

Usage:
    python scripts/lint_configs.py configs/
    python scripts/lint_configs.py -v configs/experiments/lenet5/baseline.yaml
"""

import argparse
import sys
from pathlib import Path

from hephaestus.validation.config_lint import ConfigLinter

# ML Odyssey-specific deprecated-key rules. hephaestus's ConfigLinter is
# generic and accepts these as a constructor override; its deprecated-key
# check walks dotted paths, so it works with this repo's nested configs.
#
# `required_keys` and `perf_thresholds` are intentionally NOT passed:
# hephaestus's ConfigLinter checks both against TOP-LEVEL keys, but ML
# Odyssey configs nest everything under `model:` / `training:` / `optimizer:`
# and compose via `extends:`. Passing them produced false-positive "Missing
# required key 'architecture'" errors on configs that DO define it (nested).
# A flat per-file linter cannot validate required keys across an `extends:`
# inheritance chain anyway, so the check is omitted rather than made wrong.
_DEPRECATED_KEYS = {
    "optimizer.type": "optimizer.name",
    "model.num_layers": "model.layers",
    "lr": "learning_rate",
    "val_split": "validation_split",
}


def main() -> int:
    """Lint the YAML config files passed on the command line."""
    parser = argparse.ArgumentParser(description="Lint configuration files for ML Odyssey")
    parser.add_argument("paths", nargs="+", help="Configuration files or directories to lint")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Collect YAML files to lint.
    files_to_lint: list[Path] = []
    for path_str in args.paths:
        path = Path(path_str)
        if path.is_file():
            if path.suffix in (".yaml", ".yml"):
                files_to_lint.append(path)
        elif path.is_dir():
            files_to_lint.extend(path.rglob("*.yaml"))
            files_to_lint.extend(path.rglob("*.yml"))
        else:
            print(f"Warning: Path not found: {path}")

    if not files_to_lint:
        print("No configuration files found to lint")
        return 1

    linter = ConfigLinter(
        verbose=args.verbose,
        deprecated_keys=_DEPRECATED_KEYS,
    )
    failed_files: list[Path] = []
    total_errors = 0
    total_warnings = 0

    print(f"Linting {len(files_to_lint)} configuration file(s)...\n")

    for filepath in sorted(files_to_lint):
        if not linter.lint_file(filepath):
            failed_files.append(filepath)

        total_errors += len(linter.errors)
        total_warnings += len(linter.warnings)

        if linter.errors or linter.warnings or linter.suggestions:
            print(f"\n📄 {filepath}")
            linter.print_results()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Files checked: {len(files_to_lint)}")
    print(f"Files passed: {len(files_to_lint) - len(failed_files)}")
    print(f"Files failed: {len(failed_files)}")
    print(f"Total errors: {total_errors}")
    print(f"Total warnings: {total_warnings}")

    if failed_files:
        print("\nFailed files:")
        for filepath in failed_files:
            print(f"  - {filepath}")
        return 1

    print("\n✅ All configuration files passed linting!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
