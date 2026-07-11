# Mojo ThreadSanitizer / tcmalloc Incompatibility

## Status

`just test-mojo-tsan` compiles all tests cleanly under `--sanitize thread`, but
every resulting binary aborts at startup with a tcmalloc `MmapAligned()` failure.
This is a **known incompatibility between ThreadSanitizer's shadow-memory layout
and Google tcmalloc** (the allocator linked into Mojo's runtime).

**Current:** Not functional — every `--sanitize thread` binary aborts before any
user code runs.

**When resolved:** Re-run `just test-mojo-tsan`. If it passes 298/298 tests, the
incompatibility has been reconciled upstream. Remove the inline justfile caveat
at that time and update this doc's Status section.

## Tracking

**Odyssey Issues & PRs:**

- Issue #5391 — TSAN runtime fails universally: tcmalloc/ThreadSanitizer
  incompatibility
- PR #5389 — added `just test-mojo-tsan` and documented this finding in commit
  message

**Upstream tracking:** Upstream issue **not yet filed** as of 2026-05-29. This
abort should be reported at [modular/modular issues](https://github.com/modular/modular/issues)
with the title and reproduction steps below. Until filed, track locally via
Odyssey #5391.

## Root Cause

ThreadSanitizer allocates a large shadow-memory region in the virtual address
space, typically in the high-address range (0x400000000000+). Google tcmalloc
(Mojo's linked allocator) also assumes it can mmap large regions in high memory
for its internal arena structure. When both are linked together, tcmalloc's mmap
attempts collide with TSAN's shadow memory layout, causing tcmalloc to fail
during initialization before any user code runs.

The abort occurs in `external/tcmalloc+/tcmalloc/internal/system_allocator.h`
during `MmapAligned()`, before Mojo's JIT or test framework execute.

## Symptom

Every `--sanitize thread` binary aborts at startup with:

```text
FATAL: ThreadSanitizer: unexpected memory mapping 0x<addr>-0x<addr>
<pid> external/tcmalloc+/tcmalloc/internal/system_allocator.h:585]
  MmapAligned() failed - unable to allocate with tag
  (hint=0x<addr>, size=1073741824, alignment=1073741824)
  - is something limiting address placement?
<pid> external/tcmalloc+/tcmalloc/internal/system_allocator.h:592]
  Note: the allocation may have failed because TCMalloc assumes a 48-bit
  virtual address space size; you may need to rebuild TCMalloc with
  TCMALLOC_ADDRESS_BITS defined to your system's virtual address space size
<pid> external/tcmalloc+/tcmalloc/arena.cc:59] CHECK in Alloc:
  FATAL ERROR: Out of memory trying to allocate internal tcmalloc data
  (bytes=131072, object-size=16384);
  is something preventing mmap from succeeding (sandbox, VSS limitations)?
Aborted (core dumped)
```

This happens even with a minimal `print("hello")` program.

## Evidence

From the sweep in PR #5389:

- **Tests attempted:** 298
- **Tests compiling cleanly:** 298/298 (100%)
- **Tests aborting at startup:** 298/298 (100%)
- **Distinct ThreadSanitizer race reports:** 0 (no test code runs)
- **Root of abort:** `external/tcmalloc+/...` (direct evidence tcmalloc is
  linked)

The abort message itself names `external/tcmalloc+/...` in the stack trace —
this is direct evidence that tcmalloc is the allocator linked into Mojo's
runtime.

## Reproduction

```bash
just podman-up

# Minimal reproduction: build a Mojo program with --sanitize thread
podman compose exec -T projectodyssey-dev bash -c \
  "cd /workspace && pixi run mojo build --sanitize thread --Werror -j1 \
   -I /workspace -Xlinker -lm tests/configs/test_env_vars.mojo \
   -o /tmp/tsan_test && /tmp/tsan_test"

# Expected output: binary aborts with MmapAligned() FATAL error
# (not a test failure — the runtime itself aborts before test code runs)
```

## Why This Repo Cannot Fix It

- tcmalloc is linked by Modular's Mojo runtime, not built in this repo
- The abort precedes any user code execution (happens during runtime
  initialization)
- This repo has no control over Mojo's allocator selection or configuration
- ThreadSanitizer's shadow-memory layout is fixed by the TSAN runtime, not user
  code

## Upstream Fix Paths

Either:

1. **Rebuild Mojo's tcmalloc with `TCMALLOC_ADDRESS_BITS`** configured to match
   the runtime's actual virtual address space size (64-bit systems typically need
   48–52 bits, not the default 48), or

2. **Link Mojo's `--sanitize thread` builds against the system allocator** (or
   libtsan's wrapper) instead of tcmalloc, for that specific build mode.

Both require Modular-side changes to Mojo's build system and runtime
configuration.

## Filing the Upstream Issue

When filing at [modular/modular issues](https://github.com/modular/modular/issues),
use:

**Title:** ThreadSanitizer builds abort at startup: tcmalloc MmapAligned()
shadow-memory conflict

**Description:**

Every Mojo binary built with `--sanitize thread --Werror -j1` aborts at startup
with a tcmalloc MmapAligned() FATAL error, before any test or user code runs.
This is a known incompatibility: ThreadSanitizer's shadow-memory layout
(high-address regions) collides with tcmalloc's mmap attempts.

**Evidence:** All 298 tests compile cleanly under `--sanitize thread` (codegen is
correct), but 298/298 binaries abort identically with the MmapAligned() error.
Zero ThreadSanitizer race reports (no test code reaches execution). The abort
message itself cites external/tcmalloc+/...

**Reproduction:** Build and run any Mojo program under `--sanitize thread`:

```bash
mojo build --sanitize thread hello.mojo -o hello && ./hello
```

**Expected:** races detected or test passes.

**Actual:** FATAL: ThreadSanitizer: unexpected memory mapping ... MmapAligned()
failed ...

**Mojo version:** 1.0.0b2 (from Odyssey CI)

**Upstream tracking:** Odyssey #5391, PR #5389.
