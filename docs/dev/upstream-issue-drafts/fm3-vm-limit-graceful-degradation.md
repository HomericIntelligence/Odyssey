---
title: "[BUG] mojo run aborts with opaque tcmalloc error when virtual memory is limited — request graceful degradation instead of crash"
labels: [bug, mojo]
---

### Bug description

#### Summary

When the process virtual-memory limit (`ulimit -v`) is set below Mojo's address-space
needs, `mojo run` **aborts** with an opaque, non-actionable tcmalloc error instead of a
clear compiler/runtime message. The documented minimum for Mojo development is 8 GiB RAM
(mojolang.org/docs/requirements/), so the crash itself is a below-minimum condition — but
the failure mode (hard abort, confusing output, no guidance) is a separate problem from the
address-space reservation issue tracked in #6433.

**This is NOT a duplicate of #6433.** #6433 (closed COMPLETED) was about the compiler
*reserving ~3.6 GB of virtual address space unconditionally*; it was fixed in nightly
(2026-05) and the reservation is now ~2.78 GB (measured below). This issue is about the
**behavior at the limit**: an unhandled `abort()` with a raw tcmalloc message, with no
graceful error telling the user what to do.

Reproduces on **1.0.0 stable (ed45d567)** and **1.0.0b2 (2cf4d08a)** — both versions.

#### Actual behavior

The simplest possible Mojo program, run under a 2.5 GB virtual limit, aborts:

```bash
$ ulimit -v 2621440 && mojo run hello.mojo
[pid] external/tcmalloc+/tcmalloc/internal/system_allocator.h:585] MmapAligned() failed -
unable to allocate with tag (hint=0x..., size=1073741824, alignment=1073741824) - is
something limiting address placement?
[pid] external/tcmalloc+/tcmalloc/central_freelist.h:649] tcmalloc: allocation failed 8192
ABORT: oss/modular/mojo/stdlib/std/memory/alloc.mojo:602:14: alloc failed: returned a null pointer
#0 0x... libKGENCompilerRTShared.so+0xfbf8e
...
mojo: error: execution crashed
```

The process aborts before printing anything — the message references internal tcmalloc
source paths and suggests nothing actionable.

#### Expected behavior

Mojo should **degrade gracefully**: when the virtual-memory limit is below the minimum
required for the compiler/runtime, emit a clear, actionable error such as:

```text
error: Mojo requires more virtual address space than the current limit allows.
Set `ulimit -v unlimited` (or raise the limit above ~3 GB) before running Mojo.
The documented minimum for Mojo development is 8 GiB of RAM.
```

rather than an opaque `abort()` with tcmalloc internals. Optionally, the runtime could
probe the limit at startup and fail fast with this message.

#### Steps to reproduce

Save as `hello.mojo`:

```mojo
def main():
    print("hello")
```

Run under various virtual limits (Linux):

```bash
# Crashes (measured on 1.0.0 stable + b2, x86_64, glibc 2.39):
ulimit -v 2621440 && mojo run hello.mojo   # 2.5 GB -> abort
ulimit -v 2359296 && mojo run hello.mojo   # 2.25 GB -> abort
ulimit -v 1572864 && mojo run hello.mojo   # 1.5 GB -> abort

# Passes:
ulimit -v 2883584 && mojo run hello.mojo   # 2.75 GB -> hello
```

Note the crash pattern is **non-monotonic** in the limit (e.g. 2.0 GB passes while
2.5 GB crashes) because tcmalloc's 1 GB-aligned mmap placement interacts with ASLR; the
reliable pass threshold is ~2.75–2.88 GB virtual. Measured VmPeak during a trivial
compile: **2,779,404 kB (2.78 GB)** with VmRSS of only ~194 MB.

Also reproduced with a plain `mojo build` followed by running the built binary: the binary
aborts below ~1.0–1.5 GB virtual.

#### Impact

- Users on memory-constrained runners (e.g. GitHub Actions free tier, small containers)
  get a hard crash with a confusing tcmalloc message instead of a helpful error.
- The crash signature (`alloc.mojo:602 alloc failed`) can be mistaken for the KGEN JIT
  buffer-overflow class (#6445) or other JIT bugs during triage, wasting investigation
  time.
- A graceful error would make the documented 8 GiB minimum self-enforcing: users would
  immediately know the limit is the problem.

#### Environment

- Mojo version: 1.0.0 (ed45d567) — also reproduced with 1.0.0b2 (2cf4d08a)
- OS: Linux x86_64 (Ubuntu 24.04 container), glibc 2.39
- Installed via `pip install mojo==1.0.0` / `mojo==1.0.0b2` (PyPI)

#### Related issues

- #6433 (closed COMPLETED) — the root cause of the large reservation (~3.6 GB virtual,
  now ~2.78 GB). This issue is the separate *failure-mode* problem: the hard abort at the
  limit instead of a graceful error.
- #6584 (open) — different: 53–59 GB physical RSS OOM on a large GPU kernel file, not a
  virtual-limit abort.
- #6445 (closed) — KGEN compile-time `__fortify_fail_abort`; this crash is a runtime
  allocator abort, not the KGEN buffer overflow.
