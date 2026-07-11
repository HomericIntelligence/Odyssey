# Mojo UnsafePointer Memory Safety Guidance

Developer reference for correct `UnsafePointer` usage in Odyssey. Read
before writing any code that touches raw pointers.

---

## When UnsafePointer is Necessary

Use `UnsafePointer` only when a safe alternative does not exist:

- **C interop** — calling into native libraries (`alloc`, `free`, `memset_zero`,
  `bitcast`) where a typed Mojo value cannot cross the ABI boundary.
- **Type-erased storage** — `AnyTensor` stores `UnsafePointer[UInt8]` because the
  element dtype is a runtime value; a parametric `Tensor[dtype]` cannot fill that role.
- **Performance-critical SIMD kernels** — bulk `load` / `store` with width parameters
  require direct pointer arithmetic unavailable through higher-level APIs.

Do **not** use `UnsafePointer` where a `List`, `Span`, or typed `Tensor[dtype]` is
sufficient.

---

## Mojo 1.0 Changes (Breaking)

### Non-null by design

`UnsafePointer` no longer conforms to `Boolable` or `Defaultable`. The zero-overhead
design means null is the `None` niche of `Optional`, not a live pointer value.

**Recipe 3** in
[`mojo-1.0-migration-recipe.md`](mojo-1.0-migration-recipe.md#recipe-3-unsafepointer-non-null-currently-warning-will-be-error)
covers this pattern in full.

Short form:

```mojo
# BAD — was a warning in 1.0 beta, will become an error:
if self._refcount:          # Bool(UnsafePointer) no longer meaningful
    self._refcount.destroy_pointee()

# GOOD — Case A: field can legitimately be null
var _refcount: Optional[UnsafePointer[Int]]
if self._refcount is not None:
    self._refcount.value().destroy_pointee()

# GOOD — Case B: field is always non-null (set in every __init__ path)
self._refcount.destroy_pointee()   # just remove the guard
```

Prefer **Case A** (`Optional`) when you are unsure — it is zero-cost in 1.0 and
self-documents intent.

### No-arg constructor removed

`UnsafePointer[T]()` (null sentinel) no longer compiles. The escape hatch for
code that genuinely needs a null literal is:

```mojo
UnsafePointer[T](_unsafe_null=())   # escape hatch — prefer Optional instead
```

See [Recipe 6](mojo-1.0-migration-recipe.md#recipe-6-std_os_atomic-atomic-module-relocation-null-unsafepointer)
for the full `memory_pool.mojo` fix.

---

## Patterns to Avoid

| Anti-pattern | Why dangerous | Fix |
| --- | --- | --- |
| `if ptr:` guard | `Bool(UnsafePointer)` removed in 1.0 | Use `Optional`; see Recipe 3 |
| Manual `free()` without ownership tracking | Double-free or leak when copies exist | Track lifetime with a reference count (`_refcount` pattern) or `__del__` |
| Storing `UnsafePointer` in a struct without a `__del__` | Leak on scope exit | Implement `__del__` that frees the pointer |
| Implicit null via default constructor | Compile error in 1.0 | Wrap in `Optional` |
| `bitcast` across unrelated types without size check | Undefined behavior | Assert `sizeof[From]() == sizeof[To]()` before casting |

---

## Patterns to Use

### ASAP destruction via `__del__`

Mojo destroys values as soon as the last use is reached, not at scope end. Write
`__del__` to release resources at the right moment:

```mojo
struct Buffer:
    var _data: UnsafePointer[UInt8]

    fn __init__(out self, n: Int):
        self._data = alloc[UInt8](n)

    fn __del__(owned self):
        self._data.free()   # called as soon as the last `Buffer` use is past
```

### Explicit ownership transfer with `^`

Moving a pointer into a callee gives up ownership; do not free after the move:

```mojo
var buf = Buffer(1024)
consume(buf^)      # buf is moved; do not touch buf after this line
# buf.__del__ is NOT called here — the callee owns it
```

### Reference-count pattern (`AnyTensor`)

When multiple views of the same allocation must coexist, track lifetime with a
shared counter. See `src/projectodyssey/tensor/any_tensor.mojo` lines 129–487 for the full
implementation. Key invariants:

1. Every value-initialising constructor calls `alloc[Int](1)` and sets `_refcount[] = 1`.
2. The copy constructor increments `_refcount[]`.
3. `__del__` decrements `_refcount[]`; when it reaches zero it frees `_data` **and**
   `_refcount`.
4. The move constructor transfers the pointer without touching the count.

Because `_refcount` is set in every `__init__` path, the field is always non-null and
the `if self._refcount:` guard is unnecessary (Case B from Recipe 3).

### `Optional[UnsafePointer[T]]` for nullable pointers

```mojo
struct LazyBuffer:
    var _data: Optional[UnsafePointer[Float32]]

    fn __init__(out self):
        self._data = None          # not yet allocated

    fn allocate(mut self, n: Int):
        self._data = alloc[Float32](n)

    fn __del__(owned self):
        if self._data is not None:
            self._data.value().free()
```

---

## Relationship to Runtime Crashes

The `libKGENCompilerRTShared.so` class of JIT crashes (`modular/modular#6413`) can
be triggered or masked by incorrect pointer lifetimes. Specifically:

- A double-free caused by missing reference-count management may corrupt the
  allocator state in a way that surfaces as a later JIT segfault rather than an
  obvious crash at the free site.
- Always reproduce suspected UnsafePointer bugs inside the Podman container
  before closing an issue as a JIT flake.

---

## See Also

- [`mojo-1.0-migration-recipe.md`](mojo-1.0-migration-recipe.md) — Recipe 3
  (non-null UnsafePointer), Recipe 6 (null constructor removal)
- `src/projectodyssey/tensor/any_tensor.mojo` — canonical reference-count implementation
- [`mojo-anti-patterns.md`][anti-patterns] — broader Mojo failure patterns catalogue

[anti-patterns]: https://github.com/HomericIntelligence/Odyssey/blob/main/.claude/shared/mojo-anti-patterns.md
