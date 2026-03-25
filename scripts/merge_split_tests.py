#!/usr/bin/env python3
"""Re-merge ADR-009 split Mojo test files back into single files.

ADR-009 caused monolithic test files to be split into test_<base>_part1.mojo,
test_<base>_part2.mojo, etc. to work around a Mojo v0.26.1 heap corruption bug
in the JIT compiler. This script merges them back into single test_<base>.mojo
files now that the workaround is no longer needed.

Usage:
    python3 scripts/merge_split_tests.py                        # Process all directories
    python3 scripts/merge_split_tests.py --directory tests/shared/core/  # One dir
    python3 scripts/merge_split_tests.py --dry-run              # Show what would happen
    python3 scripts/merge_split_tests.py --dry-run --directory tests/shared/core/

Rules:
    - Script only CREATES merged files, never deletes anything.
    - If a base file (test_<base>.mojo) already exists alongside part files,
      its test functions are merged in too (base functions appear first).
    - Missing part1 anomalies (e.g. test_unsigned has part2/part3 only) are
      handled gracefully — parts are still merged in the order available.
    - Output files are ready for `mojo format`.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# ADR-009 header pattern — three-line comment block at top of split files
# ---------------------------------------------------------------------------
ADR009_HEADER_RE = re.compile(
    r"^# ADR-009:.*?$",
    re.MULTILINE,
)

# Matches the full three-line ADR-009 header block (may appear as lines 1-3)
ADR009_BLOCK_LINES = {
    "# ADR-009: This file is intentionally limited to",
    "# Mojo v0.26.1 heap corruption",
    "# high test load. Split from",
}


def _is_adr009_header_line(line: str) -> bool:
    """Return True if a line is part of the ADR-009 three-line header comment."""
    stripped = line.strip()
    if stripped.startswith("# ADR-009:"):
        return True
    if stripped.startswith("# Mojo v0.26.1 heap corruption"):
        return True
    if stripped.startswith("# high test load. Split from"):
        return True
    return False


# ---------------------------------------------------------------------------
# File parsing
# ---------------------------------------------------------------------------

class ParsedMojoFile:
    """Parsed representation of a split Mojo test file."""

    def __init__(self) -> None:
        self.docstring: str = ""          # Module-level triple-quoted docstring
        self.imports: List[str] = []      # Import lines (from … / import …)
        self.section_comments: List[str] = []  # ===…=== section comment blocks
        self.test_functions: List[str] = []    # Each fn test_* body as a block
        self.test_function_names: List[str] = []  # Corresponding fn names
        self.main_body: str = ""          # Content of fn main() (unused in output)
        self.helper_functions: List[str] = []   # Non-test, non-main fn bodies
        self.helper_names: List[str] = []


def _extract_function_name(fn_header: str) -> Optional[str]:
    """Extract the function name from a 'fn name(…' header line."""
    m = re.match(r"\s*fn\s+(\w+)\s*[\(\[]", fn_header)
    if m:
        return m.group(1)
    return None


def _collect_block(lines: List[str], start: int) -> Tuple[str, int]:
    """Collect a Mojo function/block starting at `start`.

    A block ends when we return to indent level 0 after having entered it.
    Returns (block_text, next_line_index).
    """
    block_lines: List[str] = []
    i = start
    # Consume the first (header) line
    block_lines.append(lines[i])
    i += 1
    # Now consume body lines until we hit an unindented non-blank line that
    # looks like a new top-level definition or end-of-file.
    while i < len(lines):
        line = lines[i]
        # A top-level fn/struct/var/alias at column 0 signals end of block,
        # but only after we've seen at least one body line.
        if len(block_lines) > 1 and line and not line[0].isspace() and line.strip():
            break
        block_lines.append(line)
        i += 1
    # Strip trailing blank lines from block, then add exactly one blank line
    text = "\n".join(block_lines).rstrip()
    return text, i


def parse_mojo_file(path: Path) -> ParsedMojoFile:
    """Parse a Mojo test file into its component parts."""
    result = ParsedMojoFile()
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()

    i = 0
    n = len(lines)

    # --- Skip ADR-009 header lines (lines 0-2 typically) ---
    while i < n and _is_adr009_header_line(lines[i]):
        i += 1

    # Skip blank lines after header
    while i < n and not lines[i].strip():
        i += 1

    # --- Module docstring ---
    if i < n and lines[i].strip().startswith('"""'):
        # Multi-line triple-quoted docstring
        docstring_lines = [lines[i]]
        if lines[i].strip() == '"""' or lines[i].strip().count('"""') == 1:
            # Opening triple quote on its own line or inline opening
            i += 1
            while i < n:
                docstring_lines.append(lines[i])
                if '"""' in lines[i] and lines[i].strip() != '"""' or (
                    lines[i].strip() == '"""' and len(docstring_lines) > 1
                ):
                    i += 1
                    break
                i += 1
        else:
            # Single-line docstring: """text"""
            i += 1
        result.docstring = "\n".join(docstring_lines)
    elif i < n and lines[i].strip().startswith("'''"):
        # Same but single-quoted
        docstring_lines = [lines[i]]
        i += 1
        while i < n:
            docstring_lines.append(lines[i])
            if "'''" in lines[i]:
                i += 1
                break
            i += 1
        result.docstring = "\n".join(docstring_lines)

    # Skip blank lines after docstring
    while i < n and not lines[i].strip():
        i += 1

    # --- Imports ---
    while i < n:
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("from ") or stripped.startswith("import "):
            # Collect multi-line imports (continuation with indent or opening paren)
            import_lines = [line]
            # Handle parenthesised multi-line imports
            if "(" in line and ")" not in line:
                i += 1
                while i < n:
                    import_lines.append(lines[i])
                    if ")" in lines[i]:
                        i += 1
                        break
                    i += 1
            else:
                i += 1
            result.imports.append("\n".join(import_lines))
        elif not stripped or stripped.startswith("#"):
            i += 1
        else:
            break

    # --- Top-level definitions (functions, section comments, etc.) ---
    in_main = False
    main_lines: List[str] = []
    current_section_comment: List[str] = []

    while i < n:
        line = lines[i]
        stripped = line.strip()

        # Section comment blocks (===…=== style)
        if stripped.startswith("# ===") or stripped == "#" or (
            stripped.startswith("# ") and "====" in stripped
        ):
            # Collect consecutive comment lines as a section block
            comment_block = []
            while i < n and (lines[i].strip().startswith("#") or not lines[i].strip()):
                comment_block.append(lines[i])
                i += 1
            # Only keep if it looks like a section divider (contains ===)
            block_text = "\n".join(comment_block).strip()
            if "===" in block_text:
                current_section_comment = comment_block
            continue

        # Function definitions
        if stripped.startswith("fn "):
            fn_name = _extract_function_name(stripped)
            block, i = _collect_block(lines, i)
            current_section_comment = []  # Reset after use

            if fn_name == "main":
                main_lines.append(block)
            elif fn_name and fn_name.startswith("test_"):
                result.test_functions.append(block)
                result.test_function_names.append(fn_name)
            elif fn_name in ("run_all_tests",):
                # Helper runner — skip it (we'll rebuild main)
                pass
            elif fn_name:
                result.helper_functions.append(block)
                result.helper_names.append(fn_name)
            continue

        # Struct definitions and other top-level items — preserve as helpers
        if stripped.startswith("struct ") or stripped.startswith("alias ") or stripped.startswith("var "):
            block, i = _collect_block(lines, i)
            result.helper_functions.append(block)
            result.helper_names.append("<top-level>")
            continue

        # Blank line or stray comment
        i += 1

    result.main_body = "\n".join(main_lines)
    return result


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------

def _deduplicate_imports(import_lists: List[List[str]]) -> List[str]:
    """Union all import lists, preserving first-seen order, deduplicating exact text."""
    seen = set()
    result: List[str] = []
    for imports in import_lists:
        for imp in imports:
            key = imp.strip()
            if key not in seen:
                seen.add(key)
                result.append(imp)
    return result


def _deduplicate_functions(
    names_and_bodies: List[Tuple[str, str]]
) -> List[Tuple[str, str]]:
    """Deduplicate functions by name, keeping the first occurrence."""
    seen: set = set()
    result: List[Tuple[str, str]] = []
    for name, body in names_and_bodies:
        if name not in seen:
            seen.add(name)
            result.append((name, body))
    return result


def _build_merged_docstring(base_name: str, part_count: int) -> str:
    """Build a module docstring for the merged file."""
    return f'"""Tests for {base_name} (merged from {part_count} ADR-009 split files)."""'


def _build_main(test_function_names: List[str], base_name: str) -> str:
    """Build the merged fn main() that calls every test function."""
    lines = [
        "fn main() raises:",
        f'    """Run all {base_name} tests."""',
        f'    print("Running {base_name} tests...")',
        "",
    ]
    for name in test_function_names:
        lines.append(f"    {name}()")
        lines.append(f'    print("\u2713 {name}")')
        lines.append("")
    lines.append(f'    print("\\nAll {base_name} tests passed!")')
    return "\n".join(lines)


def merge_group(
    base_name: str,
    directory: Path,
    part_files: List[Path],
    base_file: Optional[Path],
    dry_run: bool,
    repo_root: Optional[Path] = None,
) -> Tuple[int, int]:
    """Merge a group of part files (and optionally a base file) into one file.

    Returns (total_input_functions, output_functions).
    """
    # Sort part files numerically by part number
    def part_number(p: Path) -> int:
        m = re.search(r"_part(\d+)\.mojo$", p.name)
        return int(m.group(1)) if m else 0

    sorted_parts = sorted(part_files, key=part_number)

    # Parse all sources
    parsed_parts = [parse_mojo_file(p) for p in sorted_parts]
    parsed_base: Optional[ParsedMojoFile] = None
    if base_file is not None:
        parsed_base = parse_mojo_file(base_file)

    # Collect all imports
    all_import_lists = [p.imports for p in parsed_parts]
    if parsed_base:
        all_import_lists.insert(0, parsed_base.imports)
    merged_imports = _deduplicate_imports(all_import_lists)

    # Collect helpers (deduplicated by name)
    helper_pairs: List[Tuple[str, str]] = []
    if parsed_base:
        for name, body in zip(parsed_base.helper_names, parsed_base.helper_functions):
            helper_pairs.append((name, body))
    for p in parsed_parts:
        for name, body in zip(p.helper_names, p.helper_functions):
            helper_pairs.append((name, body))
    merged_helpers = _deduplicate_functions(helper_pairs)

    # Collect test functions: base functions first, then parts in order
    test_pairs: List[Tuple[str, str]] = []
    if parsed_base:
        for name, body in zip(parsed_base.test_function_names, parsed_base.test_functions):
            test_pairs.append((name, body))
    for p in parsed_parts:
        for name, body in zip(p.test_function_names, p.test_functions):
            test_pairs.append((name, body))
    merged_tests = _deduplicate_functions(test_pairs)

    total_input = sum(len(p.test_function_names) for p in parsed_parts)
    if parsed_base:
        total_input += len(parsed_base.test_function_names)
    total_output = len(merged_tests)

    if dry_run:
        sources = ([base_file] if base_file else []) + sorted_parts
        print(f"  [DRY-RUN] Would merge {len(sources)} files into {directory / (base_name + '.mojo')}")
        print(f"            Sources: {[p.name for p in sources]}")
        print(f"            Imports: {len(merged_imports)} unique")
        print(f"            Test functions: {total_input} input -> {total_output} output")
        return total_input, total_output

    # Build docstring
    # Use the part1 docstring if available, otherwise generate one
    docstring = ""
    if parsed_parts:
        docstring = parsed_parts[0].docstring
    if not docstring and parsed_base:
        docstring = parsed_base.docstring
    if not docstring:
        docstring = _build_merged_docstring(base_name, len(sorted_parts))
    # Clean up part-number references in the docstring
    docstring = re.sub(r"\s*\(Part \d+ of \d+\)\.?", "", docstring)
    docstring = re.sub(r" - Part \d+.*$", "", docstring, flags=re.MULTILINE)
    docstring = re.sub(r"\s*- Part \d+:.*?\n", "\n", docstring)
    # Remove "Part N." or "Part N of M" at end of first-line title
    docstring = re.sub(r"\s*\(Part \d+\)\.?", "", docstring)
    # Remove "Run with: mojo test <part-file>" lines
    docstring = re.sub(r"\nRun with:.*_part\d+\.mojo\n", "\n", docstring)
    # Remove ADR-009 lines that appear inside the docstring body
    docstring = re.sub(r"\nADR-009:.*?\n", "\n", docstring)
    docstring = re.sub(r"\nMojo v0\.26\.1 heap corruption.*?\n", "\n", docstring)
    docstring = re.sub(r"\nhigh test load\. Split from.*?\n", "\n", docstring)
    # Remove "Note: Split from monolithic..." boilerplate inside docstrings
    docstring = re.sub(r"\nNote: Split from.*?heap corruption\n.*?See Issue.*?\n", "\n", docstring, flags=re.DOTALL)
    docstring = re.sub(r"\n{3,}", "\n\n", docstring)

    # Assemble file content
    sections: List[str] = []

    # Docstring
    sections.append(docstring)

    # Imports
    if merged_imports:
        sections.append("\n".join(merged_imports))

    # Helpers (non-test, non-main functions)
    for _, body in merged_helpers:
        sections.append(body)

    # Test functions
    for _, body in merged_tests:
        sections.append(body)

    # Main
    test_names = [name for name, _ in merged_tests]
    sections.append(_build_main(test_names, base_name))

    # Join with double newlines between sections
    content = "\n\n\n".join(sections) + "\n"

    # Write the output file
    out_path = directory / (base_name + ".mojo")
    out_path.write_text(content, encoding="utf-8")
    display = str(out_path.relative_to(repo_root)) if repo_root else out_path.name
    print(f"  Wrote {display}")
    print(f"    {total_input} test functions in -> {total_output} test functions out")

    return total_input, total_output


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_groups(directory: Path) -> Dict[str, Dict]:
    """Discover all split test groups in a directory.

    Returns a dict mapping base_name -> {
        'parts': [Path, ...],
        'base': Path or None,
        'directory': Path,
    }
    """
    groups: Dict[str, Dict] = {}

    for path in sorted(directory.glob("test_*_part*.mojo")):
        m = re.match(r"^(test_.+)_part(\d+)\.mojo$", path.name)
        if not m:
            continue
        base_name = m.group(1)
        if base_name not in groups:
            groups[base_name] = {
                "parts": [],
                "base": None,
                "directory": directory,
            }
        groups[base_name]["parts"].append(path)

    # Check for co-existing base files
    for base_name, info in groups.items():
        base_path = directory / (base_name + ".mojo")
        if base_path.exists():
            info["base"] = base_path

    return groups


def process_directory(
    directory: Path,
    dry_run: bool,
    verbose: bool = True,
    repo_root: Optional[Path] = None,
) -> Tuple[int, int, int]:
    """Process all split test groups in a single directory.

    Returns (groups_found, files_merged, test_functions_merged).
    """
    groups = discover_groups(directory)
    if not groups:
        return 0, 0, 0

    dir_display = str(directory.relative_to(repo_root)) if repo_root else str(directory)
    if verbose:
        print(f"\nDirectory: {dir_display}")
        print(f"  Groups found: {len(groups)}")

    groups_found = len(groups)
    files_merged = 0
    total_test_fns = 0

    for base_name, info in sorted(groups.items()):
        parts = info["parts"]
        base = info["base"]

        # Anomaly: warn if part1 is missing
        part_numbers = []
        for p in parts:
            m = re.search(r"_part(\d+)\.mojo$", p.name)
            if m:
                part_numbers.append(int(m.group(1)))
        part_numbers.sort()
        if part_numbers and part_numbers[0] != 1:
            print(f"  WARNING: {base_name} missing part1 (has parts: {part_numbers})")

        source_count = len(parts) + (1 if base else 0)
        if verbose:
            status = "base+parts" if base else "parts only"
            print(f"\n  {base_name}: {source_count} sources ({status})")

        try:
            input_fns, output_fns = merge_group(
                base_name=base_name,
                directory=directory,
                part_files=parts,
                base_file=base,
                dry_run=dry_run,
                repo_root=repo_root,
            )
            files_merged += source_count
            total_test_fns += output_fns
        except Exception as exc:
            print(f"  ERROR merging {base_name}: {exc}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    return groups_found, files_merged, total_test_fns


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def find_all_test_directories(root: Path) -> List[Path]:
    """Find all directories under root that contain split test files."""
    dirs = set()
    for path in root.rglob("test_*_part*.mojo"):
        dirs.add(path.parent)
    return sorted(dirs)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-merge ADR-009 split Mojo test files into single files.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--directory",
        type=Path,
        default=None,
        help="Process only this directory (relative or absolute). "
             "Default: process all directories under tests/.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Show what would happen without writing any files.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress per-file output; only print summary.",
    )
    args = parser.parse_args()

    # Resolve the repository root (script lives in scripts/)
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    if args.dry_run:
        print("DRY-RUN mode: no files will be written.")

    total_groups = 0
    total_files_merged = 0
    total_test_fns = 0

    if args.directory:
        target = args.directory if args.directory.is_absolute() else repo_root / args.directory
        if not target.is_dir():
            print(f"ERROR: not a directory: {target}", file=sys.stderr)
            return 1
        directories = [target]
    else:
        tests_root = repo_root / "tests"
        if not tests_root.is_dir():
            print(f"ERROR: tests/ directory not found at {tests_root}", file=sys.stderr)
            return 1
        directories = find_all_test_directories(tests_root)

    print(f"Processing {len(directories)} director{'y' if len(directories) == 1 else 'ies'}...")

    for directory in directories:
        g, f, t = process_directory(
            directory=directory,
            dry_run=args.dry_run,
            verbose=not args.quiet,
            repo_root=repo_root,
        )
        total_groups += g
        total_files_merged += f
        total_test_fns += t

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Groups (base names) found  : {total_groups}")
    print(f"  Source files processed     : {total_files_merged}")
    print(f"  Test functions in output   : {total_test_fns}")
    if args.dry_run:
        print("  (no files written — dry-run mode)")
    else:
        print(f"  Merged files written       : {total_groups}")
    print("=" * 60)
    print("Next steps:")
    print("  1. Review the merged files (git diff --stat)")
    print("  2. Run: pixi run mojo format tests/")
    print("  3. Run tests to verify correctness")
    print("  4. Delete the _partN.mojo files once verified")

    return 0


if __name__ == "__main__":
    sys.exit(main())
