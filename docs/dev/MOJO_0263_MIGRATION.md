# Mojo 0.26.3 Deprecation Migration Guide

## Overview

This document describes the migration from deprecated Mojo 0.26.1 syntax to Mojo 0.26.3 syntax, specifically addressing `--Werror` compilation failures.

## Deprecation Summary

Mojo 0.26.3 treats the following as errors under `--Werror`:

| Issue | Old Syntax | New Syntax | Count |
|-------|-----------|-----------|-------|
| Function keyword | `fn` | `def` | ~3,327 |
| Stdlib imports | `from memory import` | `from std.memory import` | 8+ |
| Stdlib imports | `from collections import` | `from std.collections import` | 8+ |
| Stdlib imports | `from algorithm import` | `from std.algorithm import` | 8+ |
| Stdlib imports | `from itertools import` | `from std.itertools import` | 8+ |

## Migration Scripts

Two migration scripts are provided:

### Python Script (Recommended)

**Location:** `scripts/fix_mojo_0263_deprecations_final.py`

**Usage:**

```bash
# Check what would be changed (dry-run)
python3 scripts/fix_mojo_0263_deprecations_final.py --check

# Apply changes
python3 scripts/fix_mojo_0263_deprecations_final.py
```

**Features:**

- Comprehensive regex-based migration
- Avoids false matches in strings, comments, type signatures
- Safe, idempotent, runs on all .mojo files
- Detailed output showing what was changed

### Bash Script (Alternative)

**Location:** `scripts/fix_deprecations.sh`

**Usage:**

```bash
chmod +x scripts/fix_deprecations.sh
./scripts/fix_deprecations.sh
```

## Key Implementation Details

### Function Definition Migration: `fn` → `def`

The migration uses the regex pattern:

```text
^(\s*)fn ([a-zA-Z_])
```

Replaced with:

```text
\1def \2
```

This pattern:

- Matches line start or indentation
- Captures leading whitespace
- Matches `fn ` followed by identifier start
- Preserves indentation
- **Avoids matching:**
  - `fn(...)` in type signatures
  - `fn` inside strings
  - `fn` in comments
  - `fn` in variable/module names

### Import Migration

Simple string replacement:

```python
'from memory import' → 'from std.memory import'
'from collections import' → 'from std.collections import'
'from algorithm import' → 'from std.algorithm import'
'from itertools import' → 'from std.itertools import'
```

## Verification

After migration, compile in strict mode:

```bash
# Package the shared library
pixi run mojo package --Werror -I . shared -o /tmp/shared.mojopkg

# Build utilities
pixi run mojo build --Werror -I . scripts/verify_installation.mojo

# Run tests
just test-mojo
```

Expected result: **Zero deprecation warnings**, clean compilation.

## Troubleshooting

### Issue: "fn is deprecated" still appears

**Cause:** Migration script may not have run on all files.

**Solution:**

```bash
# Re-run the migration script
python3 scripts/fix_mojo_0263_deprecations_final.py

# Verify changes
git diff --name-only
```

### Issue: Type signature errors after migration

**Cause:** Regex matched `fn` in type signatures (should not happen with provided scripts).

**Solution:**

Check affected file manually. Pattern should not match:

- `fn(arg: Type) -> ReturnType`
- `var callback: fn(...) capturing`

Safe pattern only matches `^(\s*)fn [a-zA-Z_]`, which requires `fn ` (with space) at line start.

### Issue: Import errors remain

**Cause:** Import uses different module name (not covered by script).

**Solution:**

Check import and qualify manually:

```mojo
# Before
from some_stdlib_module import Thing

# After
from std.some_stdlib_module import Thing
```

Refer to Mojo stdlib documentation for module names.

## Files Already Migrated

The following files have been manually migrated and verified:

- `tests/shared/core/test_hash.mojo` (37 fn→def + 1 import)
- `scripts/verify_installation.mojo` (1 fn→def)
- `shared/core/scalar_ops.mojo` (4 fn→def)
- `shared/core/activation_ops.mojo` (2 fn→def)
- `shared/version.mojo` (3 fn→def)

## References

- [Mojo 0.26.3 Release Notes](https://docs.modular.com/mojo/manual/)
- [Function Definitions in Mojo](https://docs.modular.com/mojo/manual/functions/)
- [Standard Library Imports](https://docs.modular.com/mojo/manual/stdlib/)
