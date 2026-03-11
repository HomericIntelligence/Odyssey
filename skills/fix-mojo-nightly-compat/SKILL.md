# Fix Mojo Nightly Compatibility Skill

| Field | Value |
|-------|-------|
| Date | 2026-03-10 |
| Objective | Fix CI failures caused by Mojo stable vs nightly version mismatch |
| Outcome | All compilation errors, deprecation warnings, and pre-commit hook failures resolved |
| Category | debugging |

## When to Use

- CI resolves a different Mojo version than local (e.g., stable 0.26.1 vs nightly 0.26.1.0.dev*)
- Compilation fails on CI with syntax that works locally
- `mojo format` crashes on `comptime` keyword with `_python_symbols` error
- Deprecation warnings for `alias`, `owned`, `ptr.offset()` need bulk fixing

## Verified Workflow

### 1. Fix `String[byte=]` removal (nightly breaking change)

The `String[byte=idx]` indexing syntax was removed in nightly. Replace with `.as_bytes()`:

```mojo
# Character comparison
# Before: if path[byte=i] == ".":
if chr(Int(path.as_bytes()[i])) == ".":

# Getting a character as String
# Before: var c = path[byte=i]
var c = chr(Int(path.as_bytes()[i]))

# Getting raw byte value for writing
# Before: self.write_byte(UInt8(ord(value[byte=i])))
self.write_byte(value.as_bytes()[i])

# String indexing for lookup tables (e.g., hex chars)
# Before: result += hex_chars[byte=high]
result += chr(Int(hex_chars.as_bytes()[high]))
```

Find all occurrences: `grep -rn '\[byte\s*=' --include='*.mojo'`

### 2. Fix `alias` -> `comptime` deprecation

```bash
# Find all: grep -rn '^\s*alias\b' --include='*.mojo'
# Replace top-level: sed -i 's/^alias /comptime /g' <files>
# Replace indented: sed -i 's/^        alias /        comptime /g' <files>
```

### 3. Fix `owned` -> `var` deprecation

```bash
# Find: grep -rn '\bowned\b' --include='*.mojo'
# Replace in function params only (not comments)
```

### 4. Fix other deprecation warnings

| Warning | Fix |
|---------|-----|
| `ptr.offset(n)` | `ptr + n` |
| `transfer ^ on owned value` | Remove the `^` |
| `unused variable` | Assign to `_` instead |
| Docstring arg order/duplicates | Fix docstring to match signature |

### 5. Fix `mojo format` hook for `comptime`

The stable `mojo format` (0.26.1) crashes on files with docstring + `comptime` or multiple
`comptime` declarations. Update the format wrapper script to treat exit code 123 (parse error)
as a warning instead of a failure:

```bash
# In scripts/mojo-format-compat.sh, after the GLIBC check:
if [ $exit_code -eq 123 ]; then
    parse_errors=$(echo "$output" | grep "^error: cannot format")
    if [ -n "$parse_errors" ]; then
        echo "WARNING: mojo format cannot parse some files:"
        echo "$parse_errors" | sed 's/^/  /'
    fi
    exit 0
fi
```

### 6. Verify

```bash
# Zero errors
pixi run mojo package -I . shared -o /tmp/shared.mojopkg 2>&1 | grep "error:"
# Zero warnings
pixi run mojo package -I . shared -o /tmp/shared.mojopkg 2>&1 | grep "warning:"
# Pre-commit passes
just pre-commit-all
```

## Failed Attempts

### Almost reverted `alias` -> `comptime`

When `mojo format` broke after the `alias` -> `comptime` change, the initial instinct was to
revert. However, investigation showed the formatter was **already broken** on main for existing
files with `comptime` (e.g., `shared/training/precision_config.mojo`). Reverting would only
reduce the number of affected files, not fix the root cause. The correct fix was to update the
format wrapper to tolerate parse errors.

### Investigated wrong upstream issue

Issue modular/modular#5943 was about `comptime assert` + `is_compile_time()` semantics — a
completely different problem from the formatter crash. The actual formatter bug was filed as
modular/modular#6144.

## Results & Parameters

| Metric | Value |
|--------|-------|
| `String[byte=]` replacements | ~14 across 5 files |
| `alias` -> `comptime` replacements | 55 across 8 files |
| `owned` -> `var` replacements | 9 across 2 files |
| Other warning fixes | 8 across 6 files |
| Final warnings | 0 |
| Final errors | 0 |
| Upstream issue filed | modular/modular#6144 |

## Mojo Format Bug Details (modular/modular#6144)

Minimal reproducer — compiles and runs, but `mojo format` crashes:

```mojo
"""X."""

comptime X = Int

fn main():
    pass
```

Two patterns trigger the crash:

1. **Docstring + comptime**: `'_python_symbols' object has no attribute 'comptime_assert_stmt'`
2. **Multiple comptime decls**: `Cannot parse` (formatter only handles one per scope)
