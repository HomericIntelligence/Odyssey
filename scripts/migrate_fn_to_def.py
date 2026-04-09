#!/usr/bin/env python3
"""
Migrate Mojo code from 'fn' to 'def' for Mojo 0.26.3 compatibility.

This script:
1. Replaces 'fn ' with 'def ' at function definitions (start of line + optional indent)
2. Qualifies implicit stdlib imports (from memory → from std.memory)
3. Avoids replacing 'fn' in type signatures (e.g., fn(...) capturing)

Usage:
    python3 scripts/migrate_fn_to_def.py
"""

import re
from pathlib import Path


def migrate_fn_to_def_in_file(filepath: Path) -> tuple[int, list[str]]:
    """
    Migrate 'fn' to 'def' in a single file.

    Returns:
        Tuple of (num_changes, list of descriptions of changes made)
    """
    try:
        content = filepath.read_text()
    except Exception as e:
        return 0, [f"Error reading {filepath}: {e}"]

    original_content = content
    changes = []

    # Replace 'fn ' at the start of a function definition
    # Pattern: start-of-line (or after indentation) + 'fn ' + identifier
    # Using raw regex to replace 'fn ' with 'def ' only for function definitions
    new_content = re.sub(
        r'^(\s*)fn (\w)',
        r'\1def \2',
        content,
        flags=re.MULTILINE
    )

    if new_content != content:
        changes.append(f"Replaced 'fn ' → 'def ' in function definitions")
        content = new_content

    # Fix implicit stdlib imports
    # from memory import ... → from std.memory import ...
    if 'from memory import' in content:
        old_mem = content
        content = content.replace('from memory import', 'from std.memory import')
        if content != old_mem:
            changes.append("Qualified 'from memory import' → 'from std.memory import'")

    # from collections import ... → from std.collections import ...
    if 'from collections import' in content:
        old_col = content
        content = content.replace('from collections import', 'from std.collections import')
        if content != old_col:
            changes.append("Qualified 'from collections import' → 'from std.collections import'")

    # from algorithm import ... → from std.algorithm import ...
    if 'from algorithm import' in content:
        old_algo = content
        content = content.replace('from algorithm import', 'from std.algorithm import')
        if content != old_algo:
            changes.append("Qualified 'from algorithm import' → 'from std.algorithm import'")

    # Write back if changed
    if content != original_content:
        filepath.write_text(content)
        return len(changes), changes

    return 0, []


def main():
    """Process all .mojo files in shared/, tests/, and scripts/."""
    mojo_files = list(Path('/home/mvillmow/ProjectOdyssey').glob('**/*.mojo'))

    if not mojo_files:
        print("No .mojo files found")
        return

    total_changes = 0
    files_modified = 0

    for filepath in sorted(mojo_files):
        # Skip generated files
        if '__pycache__' in str(filepath):
            continue

        num_changes, descriptions = migrate_fn_to_def_in_file(filepath)

        if num_changes > 0:
            files_modified += 1
            total_changes += num_changes
            rel_path = filepath.relative_to('/home/mvillmow/ProjectOdyssey')
            print(f"✓ {rel_path}")
            for desc in descriptions:
                print(f"  - {desc}")

    print(f"\nSummary:")
    print(f"  Files modified: {files_modified}")
    print(f"  Total changes: {total_changes}")


if __name__ == '__main__':
    main()
