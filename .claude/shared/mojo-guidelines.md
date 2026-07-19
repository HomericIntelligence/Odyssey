# Mojo Guidelines

Shared Mojo language guidelines for all agents. Reference this file instead of duplicating.

**Mojo Version**: 1.0.0b2 (pinned in pixi.toml).
Official docs: <https://mojolang.org/docs/>

## ⚠️ Mojo 1.0 migration (in progress)

This repo is migrating from 0.26.3 to 1.0.0b2 on branch `mojo-bump-1.0.0b2-2026050805`.
Until that merges, the codebase is in a mixed state. **For agents writing NEW code, follow
the 1.0 conventions below.** For agents fixing 0.26 code that hasn't compiled yet, see
[`docs/dev/mojo-1.0-migration-recipe.md`](../../docs/dev/mojo-1.0-migration-recipe.md).

Key 1.0 changes from 0.26 (paraphrased from `modular/modular` v1.0.0b1 release notes):

- **`fn` is deprecated**, becoming a hard error in the next release. Default to `def`. The
  `fn`-vs-`def` guidance lower in this file (which says "Use fn for performance code") is
  wrong for 1.0 and is being rewritten.
- **`unified` keyword removed**. Use `{...}` capture lists alone, or add `capturing[_]`.
- **Parametric function values need `thin`**: `op: def[T: DType](Scalar[T]) thin -> Scalar[T]`.
- **`mojo test` subcommand removed**. Tests run via `mojo run` (existing repo tests have
  hand-rolled `def main()` blocks that work on 1.0 unchanged).
- **`UnsafePointer` non-null by design**. Use `Optional[UnsafePointer[T]]` for nullable.
- **`@register_passable` and `@doc_private` removed**. Use traits / `@doc_hidden`.
- **`__moveinit__` / `__copyinit__` removed**. Use `__init__(take/copy: Self)`.
- **`std.os.atomic` → `std.atomic`**, `Consistency` → `Ordering`, `MONOTONIC` → `RELAXED`.
- **`String.as_bytes_mut` → `unsafe_as_bytes_mut`**.
- **`String.__len__` deprecated** → `byte_length()` or `count_codepoints()`.
- **Negative indexing rejected** on List/Span/String/Dict at compile time.
- **`Boolable`/`Defaultable`/`Writable` no longer imply `ImplicitlyDestructible`** in
  generic bounds.

The guidance in the rest of this file is being audited for 1.0 alignment.

## When to Use Mojo vs Python

| Use Case | Language | Reason |
| --- | --- | --- |
| ML/AI implementations | Mojo (required) | Performance, type safety |
| Performance-critical code | Mojo (required) | SIMD, optimization |
| Subprocess output capture | Python (allowed) | Mojo v1.0 limitation |
| Regex processing | Python (allowed) | No Mojo stdlib support |
| GitHub API interaction | Python (allowed) | Library availability |

**Default**: Mojo unless technical limitation documented.

## Current Syntax (v1.0+)

### Parameter Conventions

| Convention | Use For | Example |
| --- | --- | --- |
| `out self` | Constructors | `fn __init__(out self, value: Int)` |
| `mut self` | Mutating methods | `fn modify(mut self)` |
| `read` (default) | Read-only access | `fn get(self) -> Int` |
| `var` + `^` | Ownership transfer | `fn consume(var data: List[T])` |

### Deprecated Patterns

| Wrong | Correct | Notes |
| --- | --- | --- |
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

## Mojo v1.0 Package Compilation Patterns

### Library Files vs. Executable Files

**Library Files** (with relative imports):

- ❌ CANNOT compile standalone with `mojo build file.mojo`
- ✅ CAN be imported by executables using `-I` flag
- Pattern: Files using `from ..module` or `from .submodule`
- Error (expected): "cannot import relative to a top-level package"

**Executable Files** (with main()):

- ✅ CAN compile standalone with `mojo build -I . file.mojo`
- Pattern: Files with `def main()` entry point
- Uses absolute imports: `from odyssey.core import AnyTensor`

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
See [AGENTS.md](../../AGENTS.md#mojo-syntax-standards-v0257) for complete reference.
