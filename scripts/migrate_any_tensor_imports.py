#!/usr/bin/env python3
"""Migrate `from ...any_tensor import ...` to per-module imports (issue #5159).

Idempotent: a second invocation produces 0 changes.
Aborts on:
  - partition-table drift between hardcoded sets and any_tensor.mojo re-export block
  - partial-migration in-tree state (same symbol imported from old AND new module)
  - any unparsable `from … any_tensor import …` statement
"""

from __future__ import annotations

import argparse
import io
import sys
import tokenize
from pathlib import Path
from token import COMMENT, DEDENT, ENCODING, INDENT, NAME, NEWLINE, NL, OP

DEFAULT_REPO = Path(__file__).resolve().parent.parent
ROOTS = ("src", "tests", "papers", "examples", "conda.recipe", "benchmarks")
EXCLUDE_PARTS = {"build", ".pixi", "worktrees", "node_modules", ".git"}
DEFAULT_SKIP_REL = {"src/odyssey/tensor/any_tensor.mojo"}

# Hardcoded sets — VALIDATED at startup against any_tensor.mojo re-export block.
CREATION = {
    "zeros",
    "ones",
    "full",
    "empty",
    "arange",
    "eye",
    "linspace",
    "ones_like",
    "zeros_like",
    "full_like",
    "nan_tensor",
    "inf_tensor",
    "neg_inf_tensor",
    "randn",
    "_dtype_to_string",
}
UTILS = {"calculate_max_batch_size", "copy", "clone", "item", "diff", "tolist", "contiguous"}

# ---------------------------------------------------------------- token helpers


def _tokenize(text: str):
    return list(tokenize.tokenize(io.BytesIO(text.encode("utf-8")).readline))


def _iter_import_statements(toks):
    """Yield (start_idx, end_idx, module_dotted_name, symbols)  for every
    `from <module> import <names>` statement in the token stream.

    Skips comments and string content by construction (only walks OP/NAME tokens).
    For `from .foo import X`, module is ".foo" (leading dots preserved).
    """
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.type == NAME and t.string == "from":
            start = i
            i += 1
            # Collect leading dots + dotted name.
            module_parts: list[str] = []
            while i < len(toks):
                tt = toks[i]
                if tt.type == OP and tt.string == ".":
                    module_parts.append(".")
                elif tt.type == NAME and tt.string != "import":
                    module_parts.append(tt.string)
                elif tt.type == OP and tt.string != ".":
                    break
                elif tt.type == NAME and tt.string == "import":
                    break
                i += 1
            module = "".join(module_parts)
            # Expect "import".
            if i >= len(toks) or toks[i].type != NAME or toks[i].string != "import":
                continue
            i += 1
            # Symbol list, possibly inside (...).
            symbols: list[str] = []
            paren_depth = 0
            current: list[str] = []

            def flush():
                if current:
                    symbols.append("".join(current).strip())
                    current.clear()

            while i < len(toks):
                tt = toks[i]
                if tt.type == OP and tt.string == "(":
                    paren_depth += 1
                elif tt.type == OP and tt.string == ")":
                    paren_depth -= 1
                    if paren_depth == 0:
                        flush()
                        i += 1
                        break
                elif tt.type == OP and tt.string == "," and paren_depth >= 0:
                    flush()
                elif tt.type == NAME:
                    if current and current[-1] not in (".", " "):
                        current.append(" ")
                    current.append(tt.string)
                elif tt.type in (NEWLINE, NL, COMMENT, INDENT, DEDENT, ENCODING):
                    if paren_depth == 0 and tt.type == NEWLINE:
                        flush()
                        i += 1
                        break
                i += 1
            end = i
            yield start, end, module, symbols
        else:
            i += 1


# ---------------------------------------------------------------- partition


def _validate_partition_table(any_tensor_mojo: Path) -> None:
    text = any_tensor_mojo.read_text()
    toks = _tokenize(text)
    found: dict[str, set[str]] = {"tensor_creation": set(), "tensor_utils": set()}
    for _, _, module, syms in _iter_import_statements(toks):
        if module == ".tensor_creation":
            found["tensor_creation"] |= {s.split(" as ")[0].strip() for s in syms}
        elif module == ".tensor_utils":
            found["tensor_utils"] |= {s.split(" as ")[0].strip() for s in syms}
    expected = {"tensor_creation": CREATION, "tensor_utils": UTILS}
    drift_msgs = []
    for mod, exp in expected.items():
        missing = exp - found[mod]
        extra = found[mod] - exp
        if missing or extra:
            drift_msgs.append(
                f"  {mod}: in CREATION/UTILS only={sorted(missing)} in any_tensor.mojo only={sorted(extra)}"
            )
    if drift_msgs:
        raise SystemExit(
            "partition-table drift between script hardcoded sets and "
            f"{any_tensor_mojo}:\n"
            + "\n".join(drift_msgs)
            + "\nUpdate the hardcoded CREATION/UTILS sets to match, then re-run."
        )


# ---------------------------------------------------------------- preflight


def _preflight_partial_state(files: list[Path], root: Path) -> None:
    """Abort if any file imports the same symbol from any_tensor AND from
    tensor_creation/tensor_utils — that signals a merge-conflict or partial
    migration. Operates on tokenized statements so comments cannot trigger.
    """
    offenders: list[str] = []
    for p in files:
        try:
            toks = _tokenize(p.read_text())
        except (tokenize.TokenError, SyntaxError):
            continue  # consumer file Mojo-only syntax — preflight is best-effort
        old, new = set(), set()
        for _, _, module, syms in _iter_import_statements(toks):
            bases = {s.split(" as ")[0].strip() for s in syms}
            if module.endswith("any_tensor"):
                old |= bases & (CREATION | UTILS)
            elif module.endswith("tensor_creation") or module.endswith("tensor_utils"):
                new |= bases
        clash = old & new
        if clash:
            offenders.append(f"  {p.relative_to(root)}: {sorted(clash)}")
    if offenders:
        raise SystemExit(
            "partial-migration state — same symbol imported from old AND new module:\n"
            + "\n".join(offenders)
            + "\nResolve manually, then re-run."
        )


# ---------------------------------------------------------------- rewrite


def _partition(symbols: list[str]) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = {
        "any_tensor": [],
        "tensor_creation": [],
        "tensor_utils": [],
    }
    for sym in symbols:
        base = sym.split(" as ")[0].strip()
        if base in CREATION:
            buckets["tensor_creation"].append(sym)
        elif base in UTILS:
            buckets["tensor_utils"].append(sym)
        else:
            buckets["any_tensor"].append(sym)
    return buckets


def _emit(indent: str, prefix: str, buckets: dict[str, list[str]]) -> str:
    out = []
    for mod in ("any_tensor", "tensor_creation", "tensor_utils"):
        if buckets[mod]:
            out.append(f"{indent}from {prefix}{mod} import {', '.join(buckets[mod])}")
    return "\n".join(out) + "\n"


def _process(path: Path) -> bool:
    text = path.read_text()
    if "any_tensor import" not in text:
        return False
    try:
        toks = _tokenize(text)
    except (tokenize.TokenError, SyntaxError) as e:
        raise SystemExit(f"tokenize failed for {path}: {e}")
    # Identify every any_tensor import statement and its source-line span.
    targets: list[tuple[int, int, str, str, list[str]]] = []  # (line_start, line_end, indent, prefix, syms)
    for start_i, end_i, module, syms in _iter_import_statements(toks):
        if not module.endswith("any_tensor"):
            continue
        prefix = module[: -len("any_tensor")]  # "odyssey.tensor." or "."
        first_tok = toks[start_i]
        last_tok = toks[end_i - 1]
        line_start = first_tok.start[0]
        line_end = last_tok.end[0]
        # Reconstruct indent from column of "from".
        indent = " " * first_tok.start[1]
        targets.append((line_start, line_end, indent, prefix, syms))
    if not targets:
        return False
    src_lines = text.splitlines(keepends=True)
    # Replace in reverse so earlier indices stay valid.
    for line_start, line_end, indent, prefix, syms in reversed(targets):
        buckets = _partition(syms)
        replacement = _emit(indent, prefix, buckets)
        # tokenize uses 1-based line numbers; splitlines is 0-based.
        src_lines[line_start - 1 : line_end] = [replacement]
    new = "".join(src_lines)
    if new == text:
        return False
    path.write_text(new)
    return True


# ---------------------------------------------------------------- walk


def _iter_mojo_files(root: Path, skip_rel: set[str]) -> list[Path]:
    files: list[Path] = []
    for sub in ROOTS:
        d = root / sub
        if not d.exists():
            continue
        for p in d.rglob("*.mojo"):
            rel = p.relative_to(root).as_posix()
            if rel in skip_rel:
                continue
            rel_parts = Path(rel).parts
            if any(part in EXCLUDE_PARTS for part in rel_parts):
                continue
            # Defense-in-depth: skip files that lie under a symlinked dir.
            if any(parent.is_symlink() for parent in p.parents if parent != root):
                continue
            files.append(p)
    return files


# ---------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root", type=Path, default=DEFAULT_REPO, help="repo root to sweep (default: script's grandparent)"
    )
    ap.add_argument(
        "--any-tensor-mojo",
        type=Path,
        default=None,
        help="path to any_tensor.mojo used for partition-table validation"
        " (default: <root>/src/odyssey/tensor/any_tensor.mojo)",
    )
    ap.add_argument(
        "--skip-preflight", action="store_true", help="skip partial-migration preflight (used by tests only)"
    )
    args = ap.parse_args(argv)
    root = args.root.resolve()
    any_tensor_mojo = (args.any_tensor_mojo or root / "src/odyssey/tensor/any_tensor.mojo").resolve()
    _validate_partition_table(any_tensor_mojo)
    skip_rel = set(DEFAULT_SKIP_REL)
    try:
        skip_rel.add(any_tensor_mojo.relative_to(root).as_posix())
    except ValueError:
        pass  # any_tensor_mojo outside --root (sandboxed test setup)
    files = _iter_mojo_files(root, skip_rel)
    if not args.skip_preflight:
        _preflight_partial_state(files, root)
    changed = 0
    for p in files:
        if _process(p):
            print(f"rewrote {p.relative_to(root)}")
            changed += 1
    print(f"\n{changed} files rewritten")
    return 0


if __name__ == "__main__":
    sys.exit(main())
