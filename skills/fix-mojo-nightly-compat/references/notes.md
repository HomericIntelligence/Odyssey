# Fix Mojo Nightly Compat — Raw Notes

## Session Date: 2026-03-10

## Files Modified

### String[byte=] fixes

- `shared/data/_datasets_core.mojo` (lines 237, 247)
- `shared/export/protobuf.mojo` (line 108)
- `shared/utils/file_io.mojo` (lines 431, 432, 459, 460, 750)
- `shared/utils/serialization.mojo` (lines 534, 535, 566, 567)
- `shared/core/types/nvfp4.mojo` (line 629)

### alias -> comptime fixes

- `shared/testing/gradient_checker.mojo` (2)
- `shared/export/protobuf.mojo` (4)
- `shared/export/onnx_proto.mojo` (26)
- `shared/__init__.mojo` (1)
- `shared/core/lazy_expression.mojo` (9)
- `shared/core/extensor.mojo` (3)
- `.templates/training_template.mojo` (4)
- `tests/shared/core/test_memory_leaks.mojo` (6)

### Other warning fixes

- `shared/core/sequential.mojo` — owned -> var (5 params)
- `shared/core/extensor.mojo` — ptr.offset() -> ptr+, remove ^
- `shared/training/dataset_loaders.mojo` — owned -> var (4 params)
- `shared/training/trainer_interface.mojo` — unused var
- `shared/training/__init__.mojo` — duplicate docstring arg
- `shared/utils/visualization.mojo` — unused vars (2)
- `shared/core/traits.mojo` — docstring formatting
- `shared/testing/gradient_checker.mojo` — missing arg doc, arg order

### Pre-commit hook fix

- `scripts/mojo-format-compat.sh` — handle exit code 123

## Mojo Format Minimization

Tested systematically to find minimal reproducer:

| Test | Result |
|------|--------|
| Single `comptime X: Int = 0` | PASS |
| `comptime X = Int` | PASS |
| Two `comptime` decls | FAIL (Cannot parse) |
| `comptime if True: pass` | FAIL (Cannot parse) — but also doesn't compile |
| Docstring + single `comptime` | FAIL (_python_symbols error) |

Final minimal: docstring + comptime decl (3 lines). Filed as modular/modular#6144.

## CI Workflow Failures (pre-fix)

| Workflow | Root Cause |
|----------|-----------|
| Comprehensive Tests | `String[byte=]` compilation error |
| Test Data Utilities | `String[byte=]` compilation error |
| Gradient Checking Tests | `String[byte=]` compilation error |
| Pre-commit Checks | mojo format crashes on comptime |
