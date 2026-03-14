# ExTensor View Contract

**Status**: Active | **Tracked**: #3802 | **Last Updated**: 2026-03-14

This document describes the copy-vs-view semantics for `ExTensor` operations.
Understanding when an operation returns a view (shared data pointer) vs a copy
(independent data buffer) is essential for correct gradient computation and
memory management.

## Background

`ExTensor` uses reference-counted storage: each tensor holds an `UnsafePointer`
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

## Operations That Return Views

These operations return a view — no data duplication occurs:

| Operation | Location | View? | Notes |
|-----------|----------|-------|-------|
| `reshape(new_shape)` | `extensor.mojo` | Yes | Zero-copy; shape metadata changes only |
| `transpose(dim0, dim1)` | `extensor.mojo` | Yes | Strides permuted; pointer shared |
| `slice(...)` | `extensor.mojo` | Yes | Offset pointer into same buffer |
| `squeeze(dim)` | `shape.mojo` | Yes | Removes size-1 dimensions |
| `unsqueeze(dim)` | `shape.mojo` | Yes | Inserts size-1 dimensions |
| `broadcast_to(shape)` | `shape.mojo` | Yes | Stride-based broadcast; no copy |

All view operations set `_is_view = True` on the result. The refcount on the
underlying buffer is incremented by the `__copyinit__` in the view constructor.

## Operations That Return Copies

These operations allocate a new data buffer:

| Operation | Location | Notes |
|-----------|----------|-------|
| `as_contiguous()` | `extensor.mojo` | Forces C-order layout into new buffer |
| `copy()` | `extensor.mojo` | Explicit deep copy |
| Element-wise ops (`__add__`, `__mul__`, etc.) | `extensor.mojo` | Output is always new tensor |
| Reduction ops (`sum()`, `mean()`, etc.) | `reduction.mojo` | Output is always new tensor |
| `concatenate(tensors, axis)` | `shape.mojo` | Allocates output; copies all inputs |

## The `transpose_view()` Special Case

`transpose_view()` in `shared/core/matrix.mojo` is **not** a canonical view in the
sense above. It copies the raw bytes and then sets permuted strides — the data pointer
is independent. This makes it useful for testing `is_contiguous()` and `as_contiguous()`
in isolation but it is **not** the recommended API for transposing tensors in production
code. Prefer `transpose()` on `ExTensor`.

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

## See Also

- `shared/core/extensor.mojo` — `reshape()`, `transpose()`, `slice()`
- `shared/core/shape.mojo` — `broadcast_to()`, `squeeze()`, `unsqueeze()`
- `shared/core/matrix.mojo` — `transpose_view()` (test utility)
- Issue #4082 — `transpose()` vs `transpose_view()` cross-reference
- Issue #4462 — `reshape()` zero-copy view semantics
- Issue #3862 — `broadcast_to()` dimension-reduction error documentation
