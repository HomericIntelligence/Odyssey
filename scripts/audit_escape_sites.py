#!/usr/bin/env python3
"""Refined escape audit: flag only sites where the owner is NOT used after the escape.

Dangerous (#6963): `var X = <expr>` then `X._data`/`X.data_ptr` escapes into a local,
and X is never referenced again in the function -> the compiler hoists X.__deinit__
to the escape line, so subsequent reads through the pointer hit freed memory.

Safe: X is referenced again later (or X is a borrowed param).

Usage: python3 scripts/audit_escape_sites.py <file|dir>...
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ESCAPE_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)(\._data|\.data_ptr)")
VAR_DECL_RE = re.compile(r"\bvar\s+([A-Za-z_][A-Za-z0-9_]*)\s*=")
FN_START_RE = re.compile(r"^\s*(?:fn|def)\s+[A-Za-z_][A-Za-z0-9_]*")


def split_functions(lines: list[str]) -> list[tuple[int, int]]:
    starts: list[int] = []
    for i, line in enumerate(lines):
        if FN_START_RE.match(line):
            starts.append(i)
    ranges: list[tuple[int, int]] = []
    for idx, s in enumerate(starts):
        e = starts[idx + 1] if idx + 1 < len(starts) else len(lines)
        ranges.append((s, e))
    return ranges


def is_owned_local(lines: list[str], fn_range: tuple[int, int], var: str) -> bool:
    """True if `var` is declared with `var X =` inside fn_range before any escape."""
    start, end = fn_range
    for i in range(start, end):
        m = VAR_DECL_RE.search(lines[i])
        if m and m.group(1) == var:
            return True
    return False


def main() -> None:
    targets = sys.argv[1:]
    if not targets:
        print("usage: audit_escape_sites.py <file|dir>...")
        sys.exit(2)

    total_danger = 0
    for t in targets:
        p = ROOT / t
        files = [p] if p.is_file() else sorted(p.rglob("*.mojo")) if p.is_dir() else []
        for f in files:
            if "__init__" in f.name or f.name.startswith("_"):
                continue
            lines = f.read_text().splitlines()
            fns = split_functions(lines)

            def fn_for(lineno: int):
                for s, e in fns:
                    if s <= lineno < e:
                        return (s, e)
                return None

            danger: list[tuple[int, str, str]] = []
            for i, line in enumerate(lines):
                for m in ESCAPE_RE.finditer(line):
                    var = m.group(1)
                    if var in ("self", "this"):
                        continue
                    fr = fn_for(i)
                    if fr is None:
                        continue
                    owned = is_owned_local(lines, fr, var)
                    if not owned:
                        continue  # borrowed param -> Class C safe
                    # Is `var` used again later in the same fn (as a bare ident,
                    # not as `var._data` / `var.data_ptr`)?
                    used_later = False
                    for j in range(i + 1, fr[1]):
                        for m2 in re.finditer(rf"\b{re.escape(var)}\b", lines[j]):
                            nxt = lines[j][m2.start() :]
                            if nxt.startswith("._data") or nxt.startswith(".data_ptr"):
                                continue
                            used_later = True
                            break
                        if used_later:
                            break
                    if not used_later:
                        danger.append((i + 1, var, line.strip()))

            if danger:
                total_danger += len(danger)
                print(f"\n=== {f.relative_to(ROOT)} ({len(danger)} DANGEROUS) ===")
                for lineno, var, line in danger:
                    print(f"  {lineno:5d}  {var:24s} {line[:110]}")

    print(f"\nTOTAL dangerous (owned-local, never-used-after) sites: {total_danger}")


if __name__ == "__main__":
    main()
