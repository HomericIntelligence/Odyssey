#!/usr/bin/env python3
"""
Fix Mojo 0.26.3 deprecation warnings in all .mojo files.

This script migrates:
1. 'fn ' → 'def ' (function definitions only, not type signatures)
2. 'from memory import' → 'from std.memory import'
3. 'from collections import' → 'from std.collections import'
4. 'from algorithm import' → 'from std.algorithm import'
5. 'from itertools import' → 'from std.itertools import'

Usage:
    python3 scripts/fix_mojo_0263_deprecations.py
"""

import re
import sys
from pathlib import Path


def fix_file(filepath: Path) -> tuple[bool, list[str]]:
    """
    Fix deprecations in a single .mojo file.

    Returns:
        Tuple of (was_modified, list of changes made)
    """
    try:
        content = filepath.read_text(encoding="utf-8")
    except Exception as e:
        print(f"ERROR: Failed to read {filepath}: {e}", file=sys.stderr)
        return False, []

    original = content
    changes = []

    # 1. Replace 'fn ' with 'def ' at function definitions
    # Pattern: optional whitespace + 'fn ' + word character
    # This avoids matching 'fn' in strings or type signatures like fn(...)
    new_content = re.sub(r"^(\s*)fn ([a-zA-Z_])", r"\1def \2", content, flags=re.MULTILINE)
    if new_content != content:
        changes.append("fn → def")
        content = new_content

    # 2. Qualify implicit stdlib imports
    old_len = len(content)

    # from memory import ... → from std.memory import ...
    content = content.replace("from memory import", "from std.memory import")
    if len(content) != old_len:
        changes.append("memory → std.memory")
        old_len = len(content)

    # from collections import ... → from std.collections import ...
    content = content.replace("from collections import", "from std.collections import")
    if len(content) != old_len:
        changes.append("collections → std.collections")
        old_len = len(content)

    # from algorithm import ... → from std.algorithm import ...
    content = content.replace("from algorithm import", "from std.algorithm import")
    if len(content) != old_len:
        changes.append("algorithm → std.algorithm")
        old_len = len(content)

    # from itertools import ... → from std.itertools import ...
    content = content.replace("from itertools import", "from std.itertools import")
    if len(content) != old_len:
        changes.append("itertools → std.itertools")
        old_len = len(content)

    # Write back if changed
    if content != original:
        try:
            filepath.write_text(content, encoding="utf-8")
            return True, changes
        except Exception as e:
            print(f"ERROR: Failed to write {filepath}: {e}", file=sys.stderr)
            return False, []

    return False, []


def main():
    """Process all .mojo files."""
    base = Path("/home/mvillmow/Odyssey")

    if not base.exists():
        print(f"ERROR: Directory {base} does not exist", file=sys.stderr)
        sys.exit(1)

    # Find all .mojo files
    mojo_files = sorted(base.rglob("*.mojo"))
    print(f"Found {len(mojo_files)} .mojo files")
    print()

    total_modified = 0
    total_changes: dict[str, int] = {}

    for filepath in mojo_files:
        # Skip hidden and cache directories
        if "__pycache__" in str(filepath) or "/.git" in str(filepath):
            continue

        was_modified, changes = fix_file(filepath)
        if was_modified:
            total_modified += 1
            rel_path = filepath.relative_to(base)
            print(f"✓ {rel_path}")
            for change in changes:
                print(f"    {change}")
                total_changes[change] = total_changes.get(change, 0) + 1

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Files modified: {total_modified}")
    print()
    if total_changes:
        print("Changes applied:")
        for change_type in sorted(total_changes.keys()):
            count = total_changes[change_type]
            print(f"  {change_type}: {count} file(s)")
    else:
        print("No changes needed")
    print()


if __name__ == "__main__":
    main()
