#!/usr/bin/env python3
"""
Three-way consistency check for the optimizer dispatch registry.

Issue: #5682 (closes the TODO documented in PR #5682).

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

    Multi-line signatures (`def init_<n>_state(\\n    args\\n) raises ->
    ...:\\n    body`) need special handling: the `):` line is at the same
    indent as `def` (column 0), so a naive `line_indent <= def_line_indent`
    break fires at the signature's closing paren BEFORE the body ever
    starts. We scan forward to find that closing line, then collect the
    body from the next line onward.

    Args:
        source: Full file text.
        def_match: Regex match for the `def init_<name>_state(` header.

    Returns:
        The body text (everything between the signature's `):` line and
        the next top-level `def `, or end-of-file).
    """
    # Step 1 — find the signature's closing `):` line. The signature is
    # everything from the match end until a line containing `):` (possibly
    # with `raises ->` / `-> ReturnType:` decorations). After that line,
    # the actual function body begins.
    pos = def_match.end()
    while pos < len(source):
        nl = source.find("\n", pos)
        if nl == -1:
            nl = len(source)
        line = source[pos:nl]
        # Detect the signature's closing line. Could be `):`, `) raises -> ... :`,
        # `) -> ReturnType:`, etc. The line must (a) START with `)` after
        # lstrip AND (b) END with `:` (the colon that introduces the body).
        # This guards against false matches on comment lines that happen to
        # start with `)`, e.g. `# ) foo bar baz`.
        stripped = line.lstrip()
        if stripped.startswith(")") and stripped.endswith(":"):
            pos = nl + 1
            break
        pos = nl + 1

    # Step 2 — body extends until the next top-level `def ` (column 0) or EOF.
    end_match = re.search(r"^def ", source[pos:], re.MULTILINE)
    body_end = pos + end_match.start() if end_match else len(source)
    return source[pos:body_end]


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
    """Run the consistency check.

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

    # Layer 3: per-source init_<name>_state existence check
    source_exists: dict[str, bool] = {}
    for name in sorted(registry):
        source_path = optimizers_dir / f"{name}.mojo"
        if not source_path.exists():
            source_exists[name] = False
            continue
        try:
            count = extract_init_state_buffer_count(source_path, name)
            source_exists[name] = count is not None
        except Exception:
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
    return check_dispatch_sync(
        args.root,
        strict=args.strict,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    sys.exit(main())
