# Mojo Language Rules

Quick reference for agents working with Mojo code. This document consolidates the essential rules
from the shared reference files.

## Essential References

- [Mojo Guidelines](../.claude/shared/mojo-guidelines.md) - Syntax standards and conventions
- [Mojo Anti-Patterns](../.claude/shared/mojo-anti-patterns.md) - Common mistakes to avoid
- [AGENTS.md Mojo Section](../AGENTS.md#mojo-syntax-standards-v0257) - Complete reference

## Quick Rules (Must Know)

### 1. Constructor Signatures

Use `out self` for ALL constructors:

```mojo
fn __init__(out self, value: Int):
    self.value = value
```

**NEVER** use `mut self` in constructors.

### 2. Ownership Transfer

Always use `^` for List/Dict/String returns:

```mojo
fn get_weights(self) -> List[Float32]:
    return self.weights^
```

Create named variables for ownership transfer:

```mojo
# WRONG
var tensor = AnyTensor(List[Int](), DType.float32)

# CORRECT
var shape = List[Int]()
var tensor = AnyTensor(shape, DType.float32)
```

### 3. Deprecated Syntax

| Wrong | Correct |
| --- | --- |
| `inout self` | `mut self` |
| `@value` | `@fieldwise_init` + traits |
| `DynamicVector` | `List` |
| `-> (T1, T2)` | `-> Tuple[T1, T2]` |

### 4. Struct Declaration

```mojo
@fieldwise_init
struct MyType(Copyable, Movable):
    var value: Int
```

**NEVER** add `ImplicitlyCopyable` to structs with List/Dict/String fields.

### 5. List Operations

```mojo
# CORRECT - use append for new elements
var list = List[Int]()
list.append(42)

# WRONG - cannot assign to uninitialized index
var list = List[Int]()
list[0] = 42  # Runtime error!
```

## Pre-Commit Checklist

Before committing Mojo code:

- [ ] All `__init__` methods use `out self`
- [ ] All `__moveinit__` methods use `deinit existing` (not `owned existing`)
- [ ] No `inout` keyword (use `mut`)
- [ ] No `@value` decorator (use `@fieldwise_init` + traits)
- [ ] All List/Dict returns use `^` transfer operator
- [ ] Space after `var` keyword: `var a` not `vara`

## Compiler as Source of Truth

When in doubt, test with the Mojo compiler:

```bash
mojo build your_file.mojo
```

If it compiles, the syntax is valid. If it fails, trust the error message.
