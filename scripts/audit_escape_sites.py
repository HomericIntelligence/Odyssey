#!/usr/bin/env python3
"""Refined escape audit: flag only sites where the owner is NOT used after the escape.

Dangerous (#6963): `var X = <expr>` then `X._data`/`X.data_ptr` escapes into a local,
and X is never referenced again in the function -> the compiler hoists X.__deinit__
to the escape line, so subsequent reads through the pointer hit freed memory.

Safe: X is referenced again later (or X is a borrowed param).
Note: `data_ptr` sites are origin-tied (the #6963 WAR) and safe; only raw
`._data` escapes can free early. Use `--raw-only` to flag ONLY raw `._data`
escapes (the remaining dangerous class) and exit 1 if any are found —
suitable for CI gating.

Usage: python3 scripts/audit_escape_sites.py [--raw-only] <file|dir>...
Exit: 0 if no dangerous sites (raw-only mode: no raw sites), 1 otherwise.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ESCAPE_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)(\._data|\.data_ptr)")
RAW_ESCAPE_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)(\._data)")
RAWHOLD_RE = re.compile(r"\bvar\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*[A-Za-z_][A-Za-z0-9_]*\._data")
VAR_DECL_RE = re.compile(r"\bvar\s+([A-Za-z_][A-Za-z0-9_]*)\s*=")
FN_START_RE = re.compile(r"^(?:fn|def)\s+[A-Za-z_][A-Za-z0-9_]*")

RAW_ONLY = "--raw-only" in sys.argv
if RAW_ONLY:
    sys.argv.remove("--raw-only")
    ESCAPE_RE = RAW_ESCAPE_RE


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

                    if RAW_ONLY:
                        # Only pointer-HELD escapes are dangerous: the raw
                        # pointer is stored in a variable (`var ptr = X._data...`)
                        # and dereferenced in a LATER statement. Single-line
                        # expression reads (`X._data...()[unsafe_offset=i]`,
                        # `bytes_to_hex(X._data,...)`, `X._data == Y._data`)
                        # consume the pointer in the same statement — safe.
                        if not RAWHOLD_RE.match(line.lstrip()):
                            continue
                        # Skip escapes consumed within the SAME statement
                        # (value reads like `var v = X._data.unsafe_bitcast[
                        # Float32]()[unsafe_offset=i]`): track bracket depth
                        # until the statement ends; a deref inside it means
                        # the pointer never outlives the statement. A pointer
                        # held in a var (depth closes with no deref) is the
                        # dangerous shape and falls through to the used-later
                        # check.
                        depth = 0
                        consumed = False
                        for j in range(i, min(i + 6, len(lines))):
                            code = lines[j].split("#")[0]
                            # In-statement deref of the escaped pointer.
                            if (
                                "unsafe_offset=" in code
                                or "[idx]" in code
                                or "unsafe_load" in code
                                or "unsafe_store" in code
                            ):
                                consumed = True
                            for ch in code:
                                if ch in "([{":
                                    depth += 1
                                elif ch in ")]}":
                                    depth = max(depth - 1, 0)
                            if depth == 0 and j > i:
                                break
                        if consumed:
                            continue

                    owned = is_owned_local(lines, fr, var)
                    if not owned:
                        continue  # borrowed param -> Class C safe
                    # Is `var` used again later in the same fn (as a bare ident,
                    # not as `var._data` / `var.data_ptr`)? Comments are
                    # stripped so prose mentioning the name doesn't count.
                    used_later = False
                    for j in range(i + 1, fr[1]):
                        code = lines[j].split("#")[0]
                        for m2 in re.finditer(rf"\b{re.escape(var)}\b", code):
                            nxt = code[m2.start() :]
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
                try:
                    rel = f.relative_to(ROOT)
                except ValueError:
                    rel = f
                print(f"\n=== {rel} ({len(danger)} DANGEROUS) ===")
                for lineno, var, line in danger:
                    print(f"  {lineno:5d}  {var:24s} {line[:110]}")

    print(f"\nTOTAL dangerous (owned-local, never-used-after) sites: {total_danger}")

    # CI gate: raw-only mode fails when any raw `._data` escape remains.
    if RAW_ONLY and total_danger > 0:
        print("FAIL: raw _data escapes found — migrate to origin-tied data_ptr (#6963 WAR)")
        sys.exit(1)


if __name__ == "__main__":
    main()
