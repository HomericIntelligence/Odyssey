"""Minimum reproducer: AnyTensor.slice() view creates offset _data pointer
that __del__ incorrectly tries to free.

Root cause: slice() at any_tensor.mojo:677 sets _is_view = True and offsets
_data pointer into the parent tensor's allocation, but __del__ at line 491
does NOT check _is_view before calling pooled_free(self._data, ...).
Freeing an offset pointer (not returned by malloc) is undefined behavior.

Without ASAN: silently corrupts heap metadata. After enough corruption
(~15-17 function calls with tensor allocations), the allocator's internal
state becomes inconsistent and triggers libKGENCompilerRTShared.so abort.
This is what made the crash appear "flaky" — it depends on heap layout.

With ASAN: crashes immediately on the first slice() call.

Reproduction:
    # With ASAN (immediate, 100% reproducible):
    pixi run mojo build --sanitize address -g -I "$(pwd)" -I . \
        -o /tmp/repro notes/blog/03-16-2026/artifacts/repro_slice_view_bad_free.mojo
    /tmp/repro
    # Expected: AddressSanitizer: bad-free

    # Without ASAN (may or may not crash, depends on heap layout):
    pixi run mojo -I "$(pwd)" -I . \
        notes/blog/03-16-2026/artifacts/repro_slice_view_bad_free.mojo

Bug location:
    src/odyssey/tensor/any_tensor.mojo:491  (__del__ does not check _is_view)
    src/odyssey/tensor/any_tensor.mojo:754  (slice() offsets _data pointer)

Fix: __del__ must skip pooled_free when _is_view == True (view tensors
share the parent's allocation, which the parent's destructor will free
via refcount decrement).
"""

from odyssey.tensor.any_tensor import AnyTensor, ones


def main() raises:
    # Create a 4D tensor: 8 samples x 2 channels x 4 height x 4 width
    # Total: 256 float32 elements = 1024 bytes
    var data = ones([8, 2, 4, 4], DType.float32)

    # slice(4, 8) returns a "view" — _data points 512 bytes into data's allocation
    # _is_view = True, but __del__ ignores this flag
    var batch = data.slice(4, 8)

    print("batch shape:", batch.shape()[0], batch.shape()[1], batch.shape()[2], batch.shape()[3])

    # When batch goes out of scope:
    #   __del__ calls pooled_free(self._data, self._allocated_size)
    #   self._data = data._data + 512 bytes (NOT from malloc)
    #   free() on this offset pointer → ASAN: "bad-free"
    #   Without ASAN: silent heap corruption
