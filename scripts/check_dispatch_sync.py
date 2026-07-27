#!/usr/bin/env python3
"""
Three-way consistency check for the optimizer dispatch registry.

Issue: #5682 (the 3-way dispatch consistency requirement, closed by PR #5706).

Validates that two sources of truth agree on the 24 functional optimizers
shipped under `odyssey.training.optimizers`:

  1. `configs/schemas/training.schema.yaml`
       — the YAML schema's `optimizer.name` enum (CLI/config source of truth)
  2. `src/odyssey/training/dispatch.mojo`
       — the `all_supported_optimizers()` function's `String("name")` entries

Plus a source-existence check:

  3. `src/odyssey/training/optimizers/<name>.mojo`
       — each `init_<name>_state(params_list, *, force_f64)` function must
       exist for every name in the dispatch roster.

When all three agree, the dispatch is internally consistent: a CLI driver can
trust that any name from the YAML enum will route to a registered optimizer,
and each registered optimizer has a matching `init_<name>_state` in source.

Usage:
    python scripts/check_dispatch_sync.py [--root PATH] [--quiet]

Exit codes:
    0 — all three sources agree (24 optimizers × 3 layers)
    1 — at least one inconsistency, drift between sources, or parse error
"""

import argparse
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print(
        "ERROR: PyYAML is required. Install with: pip install pyyaml",
        file=sys.stderr,
    )
    sys.exit(1)


# ============================================================================
# Layer 1 — YAML schema parser
# ============================================================================


def parse_yaml_enum(schema_path: Path) -> list[str]:
    """Extract the optimizer.name enum from the YAML schema.

    Args:
        schema_path: Path to `configs/schemas/training.schema.yaml`.

    Returns:
        Sorted list of optimizer names declared in the schema's enum.

    Raises:
        FileNotFoundError, yaml.YAMLError, KeyError on malformed input.
    """
    with schema_path.open() as f:
        schema = yaml.safe_load(f)
    enum = schema["properties"]["optimizer"]["properties"]["name"]["enum"]
    return sorted(enum)


# ============================================================================
# Layer 2 — dispatch.mojo all_supported_optimizers() parser
# ============================================================================

# Matches `String("name")` entries. Captures the optimizer name.
# Used only within the `all_supported_optimizers()` function body —
# see `parse_dispatch_registry` for the scoping logic.
_DISPATCH_NAME_RE = re.compile(
    r'String\(\s*"(?P<name>[a-z_]+)"\s*\)',
)

# Matches the `def all_supported_optimizers()` function header so we can
# scope the name extraction to only that function's body (avoids false
# positives from `String("...")` calls elsewhere in dispatch.mojo).
_ALL_SUPPORTED_FN_RE = re.compile(
    r"^def\s+all_supported_optimizers\s*\(",
    re.MULTILINE,
)


def parse_dispatch_registry(dispatch_path: Path) -> set[str]:
    """Extract optimizer names from `all_supported_optimizers()` in dispatch.mojo.

    The current dispatch.mojo API (PR #5708) uses `all_supported_optimizers()`
    returning `List[String]` with `String("name")` entries. This function
    extracts those names by scoping the regex to the function body only —
    `String("...")` calls elsewhere in dispatch.mojo (e.g., error messages)
    are NOT matched. It does NOT extract buffer counts — the new API
    delegates state allocation to `init_optimizer_state()` which dispatches
    to per-optimizer `init_<name>_state` functions at the source level.

    Args:
        dispatch_path: Path to `src/odyssey/training/dispatch.mojo`.

    Returns:
        Set of optimizer names declared in `all_supported_optimizers()`.

    Raises:
        FileNotFoundError if the path does not exist.
    """
    text = dispatch_path.read_text()

    # Scope to the `all_supported_optimizers()` function body only.
    fn_match = _ALL_SUPPORTED_FN_RE.search(text)
    if fn_match is None:
        return set()

    # The function body extends from after the signature's closing `:` line
    # to the next top-level `def ` (column 0) or EOF.
    pos = fn_match.end()
    while pos < len(text):
        nl = text.find("\n", pos)
        if nl == -1:
            nl = len(text)
        line = text[pos:nl]
        stripped = line.lstrip()
        if stripped.startswith(")") and stripped.endswith(":"):
            pos = nl + 1
            break
        pos = nl + 1

    end_match = re.search(r"^def ", text[pos:], re.MULTILINE)
    body_end = pos + end_match.start() if end_match else len(text)
    body = text[pos:body_end]

    names: set[str] = set()
    for m in _DISPATCH_NAME_RE.finditer(body):
        names.add(m.group("name"))
    return names


# ============================================================================
# Layer 3 — per-source `init_<name>_state` existence check
# ============================================================================

# Matches the function header for `init_<name>_state(...)`. Captures the
# optimizer name. We use this for a pure existence check (`has_init_state`).
# Buffer counts are NOT verified here — the dispatch itself does not declare
# them (post #5708 the dispatch API delegates state allocation to
# `init_optimizer_state()` at runtime), so a buffer-count comparison would
# be comparing across a layer that no longer exists.
_INIT_DEF_RE = re.compile(
    r"^def\s+init_(?P<name>[a-z_]+)_state\s*\(",
    re.MULTILINE,
)


def has_init_state(optimizer_source: Path, name: str) -> bool:
    """Return True if `init_<name>_state(...)` is defined in the source file.

    Args:
        optimizer_source: Path to `src/odyssey/training/optimizers/<name>.mojo`.
        name: Optimizer name (e.g., "shampoo").

    Returns:
        True if a `def init_<name>_state(` header is present in the file,
        False otherwise (missing function or wrong name).

    Note:
        Caller is expected to handle I/O errors (FileNotFoundError, OSError,
        UnicodeDecodeError) by treating them as "not found".
    """
    text = optimizer_source.read_text()
    m = _INIT_DEF_RE.search(text)
    return m is not None and m.group("name") == name


# ============================================================================
# Consistency check + reporting
# ============================================================================


def check_dispatch_sync(
    root: Path,
    *,
    quiet: bool = False,
) -> int:
    """Run the consistency check.

    Args:
        root: Repository root.
        quiet: Suppress per-optimizer informational output; print only
            the summary + errors.

    Returns:
        Exit code: 0 on success, 1 on any inconsistency.
    """
    schema_path = root / "configs" / "schemas" / "training.schema.yaml"
    dispatch_path = root / "src" / "odyssey" / "training" / "dispatch.mojo"
    optimizers_dir = root / "src" / "odyssey" / "training" / "optimizers"

    # Layer 1: YAML enum
    try:
        yaml_enum = parse_yaml_enum(schema_path)
    except Exception as e:
        print(f"ERROR: failed to parse YAML schema: {e}", file=sys.stderr)
        return 1

    # Layer 2: dispatch.mojo registry
    if not dispatch_path.exists():
        print(
            f"ERROR: dispatch module not found at {dispatch_path}",
            file=sys.stderr,
        )
        return 1
    try:
        registry = parse_dispatch_registry(dispatch_path)
    except Exception as e:
        print(f"ERROR: failed to parse dispatch.mojo: {e}", file=sys.stderr)
        return 1

    # Layer 3: per-source init_<name>_state existence check
    source_exists: dict[str, bool] = {}
    for name in sorted(registry):
        source_path = optimizers_dir / f"{name}.mojo"
        if not source_path.exists():
            source_exists[name] = False
            continue
        try:
            source_exists[name] = has_init_state(source_path, name)
        except (OSError, UnicodeDecodeError):
            # I/O errors (file deleted between .exists() and read_text,
            # permission denied, or a binary source file) → treat as
            # "not found". Other exceptions would indicate a genuine bug
            # in `has_init_state` or the regex — let those propagate so
            # CI fails fast rather than silently masking regressions.
            source_exists[name] = False

    # Layer-by-layer comparison.
    errors: list[str] = []
    yaml_set = set(yaml_enum)
    reg_set = set(registry)

    # 1. YAML <-> registry name set (name mismatches are always errors)
    only_in_yaml = yaml_set - reg_set
    only_in_reg = reg_set - yaml_set
    for name in sorted(only_in_yaml):
        errors.append(f"NAME MISMATCH: '{name}' is in YAML schema but NOT in dispatch.mojo")
    for name in sorted(only_in_reg):
        errors.append(f"NAME MISMATCH: '{name}' is in dispatch.mojo but NOT in YAML schema")

    # 2. dispatch registry <-> source existence
    if not quiet:
        print("\nPer-optimizer source-existence check (dispatch ↔ source):\n")
    for name in sorted(registry):
        exists = source_exists.get(name, False)
        if not quiet:
            status = "found" if exists else "MISSING"
            print(f"  {name:<22} init_{name}_state: {status}")
        if not exists:
            errors.append(
                f"SOURCE MISSING: '{name}' — dispatch lists it but init_<name>_state not found or unparsable in source"
            )

    # Summary
    print()
    print(
        f"YAML enum:         {len(yaml_enum)} names\n"
        f"dispatch.mojo:     {len(registry)} optimizer names\n"
        f"source-verified:   {sum(1 for v in source_exists.values() if v)} / "
        f"{len(registry)} init_<name>_state functions found"
    )

    if errors:
        print(f"\nFAIL: {len(errors)} consistency error(s):", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print("\nOK: all optimizer names consistent across YAML enum, dispatch.mojo, and source.")
    return 0


# ============================================================================
# CLI
# ============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Three-way consistency check across the optimizer dispatch "
            "registry (YAML schema enum ↔ dispatch.mojo "
            "all_supported_optimizers names ↔ per-source "
            "init_<name>_state existence)."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root (default: cwd).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-optimizer informational output.",
    )
    args = parser.parse_args()
    return check_dispatch_sync(
        args.root,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    sys.exit(main())
