# AnyTensor View Contract

**Status**: Active | **Tracked**: #3802 | **Last Updated**: 2026-03-14

This document describes the copy-vs-view semantics for `AnyTensor` operations.
Understanding when an operation returns a view (shared data pointer) vs a copy
(independent data buffer) is essential for correct gradient computation and
memory management.

## Background

`AnyTensor` uses reference-counted storage: each tensor holds an `UnsafePointer`
to a flat data buffer plus a reference count. When a tensor is "copied" in Mojo's
value-semantic sense, the `__copyinit__` method increments the reference count
rather than duplicating the buffer — this is what makes views possible.

A tensor is a **view** when:

- `is_view() == True` (the `_is_view` flag is set)
- It shares the same underlying data pointer as another tensor
- Modifying elements through the view modifies the original

A tensor is an **independent copy** when:

- `is_view() == False`
- It has its own data buffer
- Modifying elements does not affect the original

## Refcount Mechanics

`AnyTensor` uses a heap-allocated `Int` as a shared reference counter
(`_refcount: UnsafePointer[Int]`). Every constructor initialises the counter to 1.
The two key lifecycle methods are:

**`__copyinit__`** (copy constructor):

- Copies the raw data pointer — no buffer allocation.
- Copies the `_refcount` pointer — both tensors now point to the same counter.
- Increments `_refcount[]` by 1.
- This applies whether the source is a view (`_is_view = True`) or not.

**`__del__`** (destructor):

- Decrements `_refcount[]` by 1.
- If `_refcount[]` reaches 0 — i.e. the last live reference is destroyed — frees
  the data buffer and the refcount allocation itself.

Key subtlety: `_is_view` is a **semantic tag only**. It does not affect when memory
is freed. Both views and value-semantic copies participate equally in reference
counting. The buffer is freed when no reference — view or otherwise — is alive.

```mojo
var t = zeros([3, 4], DType.float32)  # refcount = 1
var v = t.reshape([12])               # refcount = 2, v._is_view = True
# t goes out of scope → refcount = 1
# v goes out of scope → refcount = 0 → buffer freed
```

## Operations That Return Views

These operations return a view — no data duplication occurs:

| Operation               | Location         | View? | Notes                                  |
|-------------------------|------------------|-------|----------------------------------------|
| `reshape(new_shape)`    | `any_tensor.mojo`  | Yes   | Zero-copy; shape metadata changes only |
| `transpose(dim0, dim1)` | `any_tensor.mojo`  | Yes   | Strides permuted; pointer shared       |
| `slice(...)`            | `any_tensor.mojo`  | Yes   | Offset pointer into same buffer        |
| `squeeze(dim)`          | `shape.mojo`     | Yes   | Removes size-1 dimensions              |
| `unsqueeze(dim)`        | `shape.mojo`     | Yes   | Inserts size-1 dimensions              |
| `broadcast_to(shape)`   | `shape.mojo`     | Yes   | Stride-based broadcast; no copy        |

All view operations set `_is_view = True` on the result. The refcount on the
underlying buffer is incremented by the `__copyinit__` in the view constructor.

## `view_with_strides()` — Not Available

`view_with_strides()` does **not exist** on `AnyTensor`. It was proposed during the
issue-3236 development cycle but never implemented; the prototype caused CI failures and
was dropped before merge.

If you need a view with custom strides, use the existing view-returning operations:

| Goal                              | Use instead                  |
|-----------------------------------|------------------------------|
| Change shape (no stride reorder)  | `reshape(new_shape)`         |
| Permute two axes                  | `transpose(dim0, dim1)`      |
| Select a sub-region               | `slice(start, end, axis)`    |
| Broadcast to a larger shape       | `broadcast_to(target_shape)` |

All four operations return a view (`_is_view = True`) and share the source buffer.
None allocate.

## Operations That Return Copies

These operations allocate a new data buffer:

| Operation                                       | Location         | Notes                                 |
|-------------------------------------------------|------------------|---------------------------------------|
| `as_contiguous()`                               | `any_tensor.mojo`  | Forces C-order layout into new buffer |
| `copy()`                                        | `any_tensor.mojo`  | Explicit deep copy                    |
| Element-wise ops (`__add__`, `__mul__`, etc.)   | `any_tensor.mojo`  | Output is always new tensor           |
| Reduction ops (`sum()`, `mean()`, etc.)         | `reduction.mojo` | Output is always new tensor           |
| `concatenate(tensors, axis)`                    | `shape.mojo`     | Allocates output; copies all inputs   |

## The `transpose_view()` Special Case

`transpose_view()` in `shared/core/matrix.mojo` is **not** a canonical view in the
sense above. It copies the raw bytes and then sets permuted strides — the data pointer
is independent. This makes it useful for testing `is_contiguous()` and `as_contiguous()`
in isolation but it is **not** the recommended API for transposing tensors in production
code. Prefer `transpose()` on `AnyTensor`.

## Detecting Views at Runtime

```mojo
var t = zeros([3, 4], DType.float32)
var v = t.reshape([12])

if v.is_view():
    print("v shares data with t")  # This will print

# Force an independent copy:
var c = v.as_contiguous()
if not c.is_view():
    print("c is independent")  # This will print
```

## Implications for Gradient Computation

When implementing backward passes:

1. **Views propagate gradients to the original tensor** — if a forward pass creates a
   view and the backward receives `grad_output`, the gradient must be accumulated into
   the original tensor's gradient, not just the view's metadata.

2. **`reshape` backward**: Return `grad_output.reshape(original_shape)` — this is
   another zero-copy view of the incoming gradient.

3. **`transpose` backward**: Return `grad_output.transpose(dim0, dim1)` — swapping
   the same dimensions undoes the forward permutation.

4. **`broadcast_to` backward**: Must sum over the broadcast dimensions. The gradient
   of a broadcast is a reduction (sum) over the dimensions that were expanded.

## Contiguity and Performance

Non-contiguous views (e.g. after `transpose`) are valid tensors but may be slower for
element-wise operations that assume C-order layout. Use `as_contiguous()` to materialize
a contiguous copy before passing to performance-critical kernels (e.g. SIMD matmul).

Check contiguity with `is_contiguous()`. A view is contiguous if and only if its strides
match the standard C-order strides for its shape.

### When to Call `as_contiguous()`

Call `as_contiguous()` before any kernel that assumes C-order (row-major) strides:

- SIMD matrix multiply / BLAS wrappers
- Custom SIMD loop kernels that index via `data_ptr + row * cols + col`
- Any operation that reads the flat buffer directly without stride arithmetic

**The guard pattern** — used in `matrix.mojo:604-611`:

```mojo
var a_cont: AnyTensor
if a.is_contiguous():
    a_cont = a               # zero-copy shared-ownership assignment
else:
    a_cont = as_contiguous(a)  # allocates and copies into C-order buffer
```

This avoids unnecessary allocations on tensors that are already contiguous
(the common case for freshly constructed or reshaped tensors).

**`as_contiguous()` copy semantics**:

- Always returns `_is_view = False`.
- Allocates a fresh buffer even if the input is already contiguous — use the
  guard pattern above when allocation must be avoided.
- The result is safe to mutate without affecting the original.

**Anti-patterns to avoid**:

```mojo
# Bad: unconditional as_contiguous() wastes memory when t is already contiguous.
var t2 = as_contiguous(t)

# Bad: checking is_view() instead of is_contiguous() — a view can still be
# contiguous (e.g. reshape returns a contiguous view).
if t.is_view():
    t = as_contiguous(t)  # Wrong guard — use is_contiguous() instead
```

## See Also

- `shared/core/any_tensor.mojo` — `reshape()`, `transpose()`, `slice()`
- `shared/core/shape.mojo` — `broadcast_to()`, `squeeze()`, `unsqueeze()`
- `shared/core/matrix.mojo` — `transpose_view()` (test utility)
- Issue #4082 — `transpose()` vs `transpose_view()` cross-reference
- Issue #4462 — `reshape()` zero-copy view semantics
- Issue #3862 — `broadcast_to()` dimension-reduction error documentation
