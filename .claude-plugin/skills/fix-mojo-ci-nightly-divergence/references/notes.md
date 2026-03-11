# fix-mojo-ci-nightly-divergence - Raw Notes

## Session Date: 2026-03-10

## Problem Statement

CI used `max-nightly` Mojo channel while local development used stable `max` channel.
This caused API differences where code compiled locally but failed in CI (and vice versa).

## Specific API Differences Encountered

### String Indexing

- **Stable (works)**: `string[byte=index]` - keyword-based byte indexing
- **Nightly (works)**: `string[index]` - positional indexing (deprecated in stable)
- **Neither works on both**: No single syntax works across both versions

### Python Object Conversion

- **Stable (works)**: `Int(py=python_obj)` - keyword-based conversion
- **Nightly (fails)**: `Int(py=python_obj)` - not recognized
- **Portable workaround**: `atol(String(python_obj))` - works on both but ugly

### Removed Functions

- `is_apple_silicon()` from `sys.info` - removed from stdlib entirely
- Workaround: hardcode `False` (our CI is Linux x86_64)

## Key Files Modified

### pixi.toml (root cause fix)

```toml
# Before (nightly - causes divergence)
channels = ["https://conda.modular.com/max-nightly", "conda-forge"]

# After (stable - matches local)
channels = ["https://conda.modular.com/max", "conda-forge"]
```

### Files with reverted workarounds

- `shared/utils/file_io.mojo` - String byte indexing
- `shared/utils/serialization.mojo` - String byte indexing
- `shared/utils/progress_bar.mojo` - String byte indexing
- `shared/utils/nvfp4.mojo` - String byte indexing
- `shared/data/_datasets_core.mojo` - String byte indexing
- `shared/utils/toml_loader.mojo` - Python object conversion

### .github/workflows/pre-commit.yml

```yaml
# grep exit code fix (line 47)
violations=$(grep -rn "\.__matmul__(" . --include="*.mojo" ... || true)
```

## Timeline

1. Ran `pre-commit run --all-files` - fixed 17 hooks
2. Fixed Mojo compilation errors (sequential.mojo, precision_config.mojo, elementwise.mojo)
3. Added docstring periods to eliminate compiler warnings
4. Fixed CI workflow YAML issues (duplicate keys, stale test references)
5. Added workarounds for nightly API differences (commit a50db177)
6. Realized root cause was channel mismatch → pinned to stable
7. Reverted workarounds (now unnecessary with stable channel)

## Decision: Pin to Stable vs Maintain Nightly Compatibility

**Decision**: Pin to stable.

**Rationale**:

- Nightly APIs change without notice
- Workarounds make code harder to read
- No benefit to tracking nightly for this project
- Stable releases are well-tested
- `pixi install` with stable channel resolves to same 0.26.1.0 version
