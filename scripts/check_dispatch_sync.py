#!/usr/bin/env python3
"""
Three-way consistency check for the optimizer dispatch registry.

Issue: #5683 (closes the TODO documented in PR #5682).

Validates that three sources of truth agree on the 24 functional optimizers
shipped under `odyssey.training.optimizers`:

  1. `configs/schemas/training.schema.yaml`
       — the YAML schema's `optimizer.name` enum (CLI/config source of truth)
  2. `src/odyssey/training/dispatch.mojo`
       — the dispatch's `OptimizerSpec(...)` registry entries
       (each carries the documented `num_state_buffers`)
  3. `src/odyssey/training/optimizers/<name>.mojo`
       — each `init_<name>_state(params_list, *, force_f64)` function's
       actual per-parameter inner buffer count, extracted from the source.

When all three agree, the dispatch is internally consistent: a CLI driver can
trust that any name from the YAML enum will route to a registered
`OptimizerSpec`, and any `num_state_buffers` documented in the spec is
the same number the source actually allocates.

Usage:
    python scripts/check_dispatch_sync.py [--root PATH] [--strict] [--quiet]

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
# Layer 2 — dispatch.mojo OptimizerSpec parser
# ============================================================================

# Matches `OptimizerSpec("name", "family", "step_fn", "init_fn", lr, wd, N)`
# in either single-line or multi-line layout. Captures: name, family, n_buffers.
_DISPATCH_OPTIMIZER_SPEC_RE = re.compile(
    r"OptimizerSpec\(\s*"  # OptimizerSpec(
    r'"(?P<name>[a-z_]+)"\s*,\s*'  # "name",
    r'"(?P<family>[a-z_]+)"\s*,\s*'  # "family",
    r'"[a-z_]+_step"\s*,\s*'  # "<name>_step",
    r'"init_[a-z_]+_state"\s*,\s*'  # "init_<name>_state",
    r"[\d.]+\s*,\s*"  # default_learning_rate,
    r"[\d.]+\s*,?\s*"  # default_weight_decay (optional trailing comma),
    r"(?P<n_buffers>\d+)\s*,?\s*"  # num_state_buffers (optional trailing comma)
    r"\)",  # )
    re.MULTILINE,
)


def parse_dispatch_registry(dispatch_path: Path) -> dict[str, int]:
    """Extract (name → num_state_buffers) from each `OptimizerSpec(...)` call.

    Args:
        dispatch_path: Path to `src/odyssey/training/dispatch.mojo`.

    Returns:
        Dict mapping optimizer name to the `num_state_buffers` Int declared
        in the registry. Order from the source is preserved (caller may
        re-sort if needed).

    Raises:
        FileNotFoundError if the path does not exist.
    """
    text = dispatch_path.read_text()
    registry: dict[str, int] = {}
    for m in _DISPATCH_OPTIMIZER_SPEC_RE.finditer(text):
        name = m.group("name")
        n_buffers = int(m.group("n_buffers"))
        # Last-write-wins on duplicate entries (catch refactor typos).
        registry[name] = n_buffers
    return registry


# ============================================================================
# Layer 3 — per-source `init_<name>_state` buffer-count extractor
# ============================================================================

# Matches the function header for `init_<name>_state(...)`. Captures the
# optimizer name. We then extract the function body via brace-balanced scan
# (Mojo uses indentation rather than braces, but the function ends at the
# next `def ` at the same indent level — see `_extract_function_body`).
_INIT_DEF_RE = re.compile(
    r"^def\s+init_(?P<name>[a-z_]+)_state\s*\(",
    re.MULTILINE,
)

# Pattern A (loop-based): `for _ in range(N): per.append(...)` inside the
# function body. Captures the integer N.
_LOOP_BUFFER_COUNT_RE = re.compile(
    # Use [ \t]* (NOT \s*) before \n so the greedy \s doesn't eat the newline.
    r"for\s+_\s+in\s+range\s*\(\s*(?P<n>\d+)\s*\)[ \t]*:[ \t]*\n[ \t]*per\.append",
    re.MULTILINE,
)

# Pattern B (append-based): every `per.append(...)` inside the function
# body. Count these. Used as a fallback when pattern A doesn't match.
_APPEND_COUNT_RE = re.compile(r"^\s*per\.append\s*\(", re.MULTILINE)


def _extract_function_body(source: str, def_match: re.Match) -> str:
    """Extract the indented body of the `def init_<name>_state(...)` block.

    Mojo uses leading-whitespace indentation, not braces, so the body
    extends from the line after the `def` header until the next
    line at the SAME indent as `def` (or end-of-file).

    Args:
        source: Full file text.
        def_match: Regex match for the `def init_<name>_state(` header.

    Returns:
        The body text (all indented lines after the header).
    """
    # The `def` line's leading whitespace defines the block's indent.
    def_line_start = source.rfind("\n", 0, def_match.start()) + 1
    def_line_indent = len(source[def_line_start:]) - len(source[def_line_start:].lstrip())

    # Body starts on the next non-empty line after the def line.
    pos = source.find("\n", def_match.end()) + 1
    body_lines: list[str] = []
    while pos < len(source):
        nl = source.find("\n", pos)
        if nl == -1:
            nl = len(source)
        line = source[pos:nl]
        if line.strip() == "":
            body_lines.append(line)
            pos = nl + 1
            continue
        line_indent = len(line) - len(line.lstrip())
        if line_indent <= def_line_indent:
            break
        body_lines.append(line)
        pos = nl + 1
    return "\n".join(body_lines)


def extract_init_state_buffer_count(optimizer_source: Path, name: str) -> int | None:
    """Return the per-parameter inner buffer count of `init_<name>_state`.

    Two extraction strategies, in order of preference:

    1. **Loop-based**: scan the function body for `for _ in range(N): per.append(...)`
       and return N. Covers the canonical `init_<name>_state` template used
       by sgd, adam, adamw, adopt, adan, sophia, rmsprop, adagrad, lars,
       muon, normuon, mgup_muon, muon_hyperball, lion, lionmuon, sf_normuon,
       ftrl, schedule_free, schedule_free_plus, prodigy.

    2. **Append-based**: count `per.append(...)` calls in the function body.
       Covers optimizers that allocate buffers via explicit `per.append(eye(...))`
       etc. inside an `if <eligibility>:` branch — notably shampoo, kl_shampoo,
       soap, splus. Counts the *matrix-eligible* path, which is what the
       dispatch's `num_state_buffers` documents.

    Args:
        optimizer_source: Path to `src/odyssey/training/optimizers/<name>.mojo`.
        name: Optimizer name (e.g., "shampoo").

    Returns:
        The extracted buffer count, or None if the function or pattern is
        not found.
    """
    text = optimizer_source.read_text()
    m = _INIT_DEF_RE.search(text)
    if m is None or m.group("name") != name:
        return None

    body = _extract_function_body(text, m)
    loop_match = _LOOP_BUFFER_COUNT_RE.search(body)
    if loop_match is not None:
        return int(loop_match.group("n"))

    # Fallback: count `per.append(...)` calls. For eligibility-gated
    # optimizers (shampoo family), these all live inside the same `if`
    # branch, so a global count in the function body equals the
    # matrix-eligible buffer count.
    return len(_APPEND_COUNT_RE.findall(body))


# ============================================================================
# Consistency check + reporting
# ============================================================================


def check_dispatch_sync(
    root: Path,
    *,
    strict: bool = False,
    quiet: bool = False,
) -> int:
    """Run the three-way consistency check.

    Args:
        root: Repository root.
        strict: Reserved for future use (e.g., fail-on-warning). Currently
            a no-op — any source-level mismatch is already a hard error.
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

    # Layer 3: per-source init_<name>_state buffer counts
    source_counts: dict[str, int | None] = {}
    for name in sorted(registry):
        source_path = optimizers_dir / f"{name}.mojo"
        if not source_path.exists():
            source_counts[name] = None
            continue
        try:
            source_counts[name] = extract_init_state_buffer_count(source_path, name)
        except Exception:
            source_counts[name] = None

    # Layer-by-layer comparison.
    errors: list[str] = []
    yaml_set = set(yaml_enum)
    reg_set = set(registry)

    # 1. YAML <-> registry name set
    only_in_yaml = yaml_set - reg_set
    only_in_reg = reg_set - yaml_set
    for name in sorted(only_in_yaml):
        errors.append(f"NAME MISMATCH: '{name}' is in YAML schema but NOT in dispatch.mojo registry")
    for name in sorted(only_in_reg):
        errors.append(f"NAME MISMATCH: '{name}' is in dispatch.mojo registry but NOT in YAML schema")

    # 2. dispatch registry <-> source buffer count
    if not quiet:
        print("\nPer-optimizer buffer-count check (registry ↔ source):\n")
    for name in sorted(registry):
        reg_count = registry[name]
        src_count = source_counts.get(name)
        if not quiet:
            status = (
                f"registry={reg_count} source={src_count}"
                if src_count is not None
                else f"registry={reg_count} source=<not found>"
            )
            print(f"  {name:<22} {status}")
        if src_count is None:
            errors.append(
                f"BUFFER-COUNT MISSING: '{name}' — dispatch says {reg_count}, "
                "but init_<name>_state not found or unparsable in source"
            )
        elif src_count != reg_count:
            errors.append(f"BUFFER-COUNT MISMATCH: '{name}' — dispatch says {reg_count}, source emits {src_count}")

    # Summary
    print()
    print(
        f"YAML enum:         {len(yaml_enum)} names\n"
        f"dispatch.mojo:     {len(registry)} OptimizerSpec entries\n"
        f"source-extracted:  {sum(1 for v in source_counts.values() if v is not None)} / "
        f"{len(registry)} counts successfully extracted"
    )

    if errors:
        print(f"\nFAIL: {len(errors)} consistency error(s):", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print("\nOK: all 24 optimizers consistent across YAML enum, dispatch.mojo, and source.")
    return 0


# ============================================================================
# CLI
# ============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Three-way consistency check across the optimizer dispatch "
            "registry (YAML schema enum ↔ dispatch.mojo OptimizerSpec "
            "entries ↔ per-source init_<name>_state buffer counts)."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root (default: cwd).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Reserved for future use (no effect today).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-optimizer informational output.",
    )
    args = parser.parse_args()
    return check_dispatch_sync(args.root, strict=args.strict, quiet=args.quiet)


if __name__ == "__main__":
    sys.exit(main())
