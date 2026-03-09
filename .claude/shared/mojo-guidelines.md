# Mojo Guidelines

Shared Mojo language guidelines for all agents. Reference this file instead of duplicating.

**Mojo Version**: 0.26.1 (pinned in pixi.toml).
Official docs: <https://docs.modular.com/mojo/manual/>

## When to Use Mojo vs Python

| Use Case | Language | Reason |
|----------|----------|--------|
| ML/AI implementations | Mojo (required) | Performance, type safety |
| Performance-critical code | Mojo (required) | SIMD, optimization |
| Subprocess output capture | Python (allowed) | Mojo v0.26.1 limitation |
| Regex processing | Python (allowed) | No Mojo stdlib support |
| GitHub API interaction | Python (allowed) | Library availability |

**Default**: Mojo unless technical limitation documented.

## Current Syntax (v0.26.1+)

### Parameter Conventions

| Convention | Use For | Example |
|------------|---------|---------|
| `out self` | Constructors | `fn __init__(out self, value: Int)` |
| `mut self` | Mutating methods | `fn modify(mut self)` |
| `read` (default) | Read-only access | `fn get(self) -> Int` |
| `var` + `^` | Ownership transfer | `fn consume(var data: List[T])` |

### Deprecated Patterns

| Wrong | Correct | Notes |
|-------|---------|-------|
| `borrowed self` | `self` | Deprecated keyword |
| `inout self` | `mut self` | Deprecated keyword |
| `@value` | `@fieldwise_init` + traits | Add `(Copyable, Movable)` |
| `DynamicVector[T]` | `List[T]` | Use `.append()` not `.push_back()` |
| `-> (T1, T2)` | `-> Tuple[T1, T2]` | Explicit tuple type |

### Function Definitions

**Use `fn`** for:

- Performance-critical code (compile-time optimization)
- Functions with explicit type annotations
- SIMD/vectorized operations
- Production APIs

**Use `def`** for:

- Python-compatible functions
- Quick prototypes
- Dynamic typing needed

### Struct Patterns

```mojo
# With @fieldwise_init (recommended for simple structs)
@fieldwise_init
struct Point(Copyable, Movable):
    var x: Float32
    var y: Float32

# Manual constructor (for complex initialization)
struct Tensor(Copyable, Movable):
    var data: DTypePointer[DType.float32]
    var shape: List[Int]

    fn __init__(out self, shape: List[Int]):
        self.shape = shape
        # ... initialization logic
```

### List Initialization

**Per [Mojo Manual](https://docs.modular.com/mojo/manual/types#list)**: Use list literals.

```mojo
# CORRECT - List literal
var shape = [3, 4, 5]  # Type inferred as List[Int]
var shape: List[Int] = [3, 4, 5]  # Explicit type

# CORRECT - Empty list
var empty = List[Int]()

# ❌ WRONG - Variadic constructor does not exist
var shape = List[Int](3, 4, 5)  # Compiler error
```

## Mojo v0.26.1 Package Compilation Patterns

### Library Files vs. Executable Files

**Library Files** (with relative imports):

- ❌ CANNOT compile standalone with `mojo build file.mojo`
- ✅ CAN be imported by executables using `-I` flag
- Pattern: Files using `from ..module` or `from .submodule`
- Error (expected): "cannot import relative to a top-level package"

**Executable Files** (with main()):

- ✅ CAN compile standalone with `mojo build -I . file.mojo`
- Pattern: Files with `def main()` entry point
- Uses absolute imports: `from shared.core import ExTensor`

### Building Packages

**DO**:

```bash
mojo package shared                          # Build entire package
mojo build -I . examples/example.mojo       # Build executable
```

**DON'T**:

```bash
mojo build shared/__init__.mojo              # ❌ Fails with relative import error
mojo build shared/core/activation.mojo       # ❌ Fails with relative import error
```

### Expected "Errors" That Are Not Bugs

1. **"cannot import relative to a top-level package"**
   - When: Compiling library files with `mojo build`
   - Why: Library files use relative imports, not meant to compile standalone
   - Fix: Don't try to compile them standalone; use `mojo package` instead

2. **"module does not contain a 'main' function"**
   - When: Compiling library modules with `mojo build`
   - Why: Library modules aren't executables
   - Fix: Only compile files meant to be executables

3. **"unable to locate module 'shared'"**
   - When: Missing `-I .` flag or wrong working directory
   - Why: Mojo needs include path to find packages
   - Fix: Use `-I .` flag: `mojo build -I . file.mojo`

## Pre-Commit Checklist

- [ ] All `__init__` use `out self` (not `mut self`)
- [ ] No `inout` keyword (use `mut`)
- [ ] No `@value` decorator (use `@fieldwise_init` + traits)
- [ ] All List/Dict returns use `^` transfer operator
- [ ] Space after `var` keyword: `var a` not `vara`
- [ ] List initialization uses literals: `[1, 2, 3]` not `List[Int](1, 2, 3)`

See [mojo-anti-patterns.md](mojo-anti-patterns.md) for common mistakes.
See [CLAUDE.md](../../CLAUDE.md#mojo-syntax-standards-v0257) for complete reference.
