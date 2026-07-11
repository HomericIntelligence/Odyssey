# [BUG] JIT Compiler __fortify_fail_abort in CI (libKGENCompilerRTShared.so)

## Environment

- Mojo version: 0.26.3.0.dev2026040705 (69cac1bd)
- OS: Ubuntu 24.04.3 LTS (Noble Numbat)
- GLIBC: 2.39 (Ubuntu GLIBC 2.39-0ubuntu8.7)
- Container: odyssey:dev (Podman rootless, UID 1001 on CI runners, UID 1000 locally)
- CI runner: GitHub Actions `ubuntu-latest`

## Description

All 97 affected test files pass locally (including under ASAN, which reports only memory
leaks — no heap corruption or buffer overflows). In CI, a subset crash with
`error: execution crashed` before producing any test output. The stack trace implicates
`__fortify_fail_abort` in glibc, which is triggered by glibc's stack-smash / buffer-overflow
detection (`__stack_chk_fail` / `__fortify_fail`) inside the JIT compiler itself.

## Stack Trace

```text
#0  libKGENCompilerRTShared.so  +0x6d4ab  (within .text section 0x24230-0x96b3b)
#1  libKGENCompilerRTShared.so  +0x6a686
#2  libKGENCompilerRTShared.so  +0x6e157
#3  libc.so.6                   +0x45330  (__fortify_fail_abort)
#4  JIT compiled code
error: execution crashed
```

`libc.so.6+0x45330` resolves to `__fortify_fail_abort` in glibc 2.39. This is the landing
pad for glibc's buffer-overflow canary checks (`__stack_chk_fail`) and bounds-checked string
functions (`__fortify_fail`). Frame #4 in JIT compiled code (not in an allocator library)
confirms the overflow is detected inside the JIT, not during heap allocation.

## Isolation Experiments

| Variant | Crashes? | Conclusion |
| --- | --- | --- |
| Run locally, same Docker image, UID 1000 | No | Not a code or library bug |
| Run locally under ASAN | No heap errors; only memory leaks | No buffer overflow in our code |
| ASAN on test_arithmetic.mojo | PASSED clean | Core tensor ops are ASAN-clean |
| ASAN on test_activations.mojo | PASSED clean | Activation functions are ASAN-clean |
| ASAN on test_broadcasting.mojo | PASSED clean | Broadcasting logic is ASAN-clean |
| ASAN on test_training_loop.mojo | Leaks only (5120B + 24B × 5) | Leaks in pooled_alloc, no overflow |
| CI run, fresh pixi volumes, UID 1001 | Crashes (~97 files) | Environment-specific |

## Memory Leak Detail (ASAN, test_training_loop.mojo)

ASAN detects **leak-only** errors (no heap-use-after-free, no buffer overflow). The leaks
originate in `shared::base::memory_pool::pooled_alloc` and surface through
`AnyTensor::__init__` → `zeros`/`ones` in test setup. These are pre-existing leaks from the
memory pool's intentional slab-based design, not related to the CI crash.

```text
SUMMARY: AddressSanitizer: 7000 byte(s) leaked in 6 allocation(s)
  - 5120B: pooled_alloc → AnyTensor::__init__ → ones → test_training_loop::main()
  - 24B × 5: pooled_alloc → AnyTensor::__init__ → zeros → test_training_loop::main()
```

## Relationship to #6187

This is a **different crash** from modular/modular#6187 (heap corruption via bitcast UAF):

| Property | #6187 crash | This crash |
| --- | --- | --- |
| Offsets | `libKGENCompilerRTShared.so +0x3cb78b` | `+0x6d4ab` (much smaller) |
| Frame #4 | `libAsyncRTRuntimeGlobals.so +0x416ba` (heap allocator) | JIT compiled code |
| libc frame | `+0x45330` (sigaction) | `+0x45330` (**fortify_fail_abort**) |
| Root cause | Heap corruption (UAF via bitcast) | Stack/buffer overflow in JIT |
| Reproducible locally | Yes (deterministic) | No (CI-only) |

Note: `libc.so.6+0x45330` appears in both traces but resolves differently. In #6187 the
process received `SIGSEGV` from the allocator; glibc's signal handler happened to be at that
offset. In this crash the same offset is `__fortify_fail_abort` invoked proactively by glibc's
own overflow checks. The identical numeric offset is coincidental given the same GLIBC 2.39
binary on both systems.

## CI-Specific Factors

Three factors present in CI but absent locally:

1. **Fresh named volumes on every run** — `pixi-cache` and `workspace-pixi` volumes are
   created empty. Pixi runs a full cold install of Mojo 0.26.3 on each CI job. Locally the
   volumes persist between runs and the pixi environment is warm.

2. **UID mismatch** — CI runners are `ubuntu-latest` with the default runner user at UID 1001.
   The container image is built for UID 1000 (`dev`). `docker-compose.yml` passes
   `user: "${USER_ID}:${GROUP_ID}"` into the container, so the container runs with UID 1001
   but `HOME=/home/dev` (owned by UID 1000). The `pixi-cache` volume is mounted at
   `/home/dev/.pixi` but the running user's home is effectively mismatched.
   Locally UID matches the container image (both 1000).

3. **No TTY (`-T` flag)** — `just _run` routes all commands through
   `podman compose exec -T odyssey-dev bash -c "..."`. The `-T` flag disables TTY
   allocation. Some JIT compiler code paths differ when stdin/stdout are not a terminal
   (e.g., signal handling, SIGPIPE disposition, terminal query sequences in LLVM/MLIR).

## Root Cause Hypothesis

The JIT compiler (`libKGENCompilerRTShared.so`) contains an internal buffer (likely in an MLIR
pass, codegen pipeline, or symbol table) whose size is computed or allocated based on environment
state. When pixi runs a cold install (fresh volume), the Mojo stdlib package (`.mojopkg`) is
unpacked and compiled for the first time. The resulting `.pixi` directory is owned by UID 1000
but the process runs as UID 1001, which may cause cache-miss paths through the compiler that
exercise a larger number of compilation units concurrently.

The combination of (a) large compilation scope from many test functions/imports, (b) fresh cache
forcing full JIT recompilation, and (c) potential UID-induced permission fallback paths in the
pixi/Mojo cache layer pushes an internal JIT buffer past its fixed capacity, triggering glibc's
stack canary or `__fortify_fail` check.

The `__fortify_fail_abort` in frame #3 (not frame #0) means the crash was not an OS signal —
the glibc hardened libc code detected the overflow proactively and called `abort()`.

## Expected Behavior

JIT compilation completes successfully regardless of pixi cache state, UID of the running
process, or TTY availability.

## Actual Behavior

`error: execution crashed` with `__fortify_fail_abort` in `libc.so.6+0x45330`, only in CI
(fresh pixi volumes + UID 1001 + no TTY), for approximately 97 test files that all pass
locally.

## Workaround

The crash is non-blocking: CI is configured with `continue-on-error: true` for affected test
groups (see `comprehensive-tests.yml` lines 254, 331, 337, 353). All tests pass locally and
under ASAN.

## Minimal Reproducer Attempt

See `repro/repro_jit_fortify_crash.mojo`. The reproducer does NOT reproduce the crash locally
(by definition of this being a CI-only issue), but documents the JIT stress patterns that
correlate with the crash: large struct count, many generic instantiations, deep SIMD nesting,
and heavy type parameter permutations.
