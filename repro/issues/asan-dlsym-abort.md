# [BUG] ABORT in dlsym when loading Python FFI symbols under AddressSanitizer

## Environment

- Mojo version: 0.26.3 (dev2026040705)
- OS: Ubuntu 24.04
- GLIBC: 2.39
- ASAN: libasan.so.8

## Description

When running Mojo code that uses Python FFI (`from python import Python`) under
AddressSanitizer, the runtime aborts with:

```text
ABORT: oss/modular/mojo/stdlib/std/ffi/__init__.mojo:647:22: dlsym unexpectedly
returned non-NULL result when loading symbol: PyRun_SimpleString
```

ASAN intercepts `dlsym` calls and provides its own implementation. When Mojo's FFI
loader calls `dlsym(RTLD_DEFAULT, "PyRun_SimpleString")` to check whether CPython is
available, ASAN's interceptor returns a non-NULL result even when CPython should not be
accessible. The Mojo stdlib then aborts because it got an unexpected non-NULL result from
what it expected to be a "not found" check.

## Stack Trace

```text
ABORT: oss/modular/mojo/stdlib/std/ffi/__init__.mojo:647:22: dlsym unexpectedly
returned non-NULL result when loading symbol: PyRun_SimpleString
0  libasan.so.8               0x00007f6bbdd7a1e0
1  libKGENCompilerRTShared.so 0x00007f6bbdcbc4ab
2  libKGENCompilerRTShared.so 0x00007f6bbdcb9686
3  libKGENCompilerRTShared.so 0x00007f6bbdbd157
4  libc.so.6                  0x00007f6bbd9ae330
```

## Expected Behavior

Mojo FFI loader should handle ASAN-intercepted dlsym gracefully, or ASAN should not
intercept Mojo's internal FFI symbol lookups.

## Actual Behavior

Runtime aborts immediately when any Mojo code using Python FFI runs under ASAN.

## Minimal Reproducer

```mojo
from python import Python

def main() raises:
    var py = Python.import_module("os")
    print(py.getcwd())
```

Run with: `mojo build --sanitize address minimal_asan_ffi.mojo && ./minimal_asan_ffi`

## Impact

Prevents ASAN testing of any Mojo code that uses Python FFI, including serialization
utilities that call Python's os.makedirs() or pathlib functions.
