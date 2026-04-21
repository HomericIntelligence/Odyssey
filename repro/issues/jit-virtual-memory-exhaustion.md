# Mojo JIT maps ~3.7GB virtual address space per compiler invocation

## Environment

- Mojo 0.26.3.0.dev2026040705 (69cac1bd)
- GLIBC 2.39-0ubuntu8.7 (Ubuntu 24.04)
- Linux x86_64, 16GB RAM (confirmed passes), 7GB RAM (crashes)

## Summary

Every `mojo run` / `mojo build` invocation maps approximately **3.7GB of virtual
address space** regardless of the source file's complexity. This exceeds the
available virtual address space on GitHub Actions free-tier runners (7GB total)
when multiple `mojo` processes run concurrently, causing OOM crashes.

## Reproduction

```bash
# Crashes with <3.7GB virtual memory limit
ulimit -v 3500000  # 3.5GB
mojo run repro/repro_hello.mojo  # even "def main(): print('hello')" crashes

# Passes with ≥3.7GB virtual memory
ulimit -v 4000000  # 4GB
mojo run repro/repro_hello.mojo  # PASS
```

The crash occurs even for trivially simple files:

```mojo
# repro/repro_hello.mojo
def main() raises:
    print("hello")
```

```bash
$ ulimit -v 3500000 && mojo run repro/repro_hello.mojo
# Crash in libKGENCompilerRTShared.so — no output produced
```

## Measured Data

Peak physical RSS (measured via /proc/$PID/status sampling):
- Any `mojo run` invocation: ~330MB RSS

Virtual address space (measured via ulimit -v threshold):
- Crash threshold: < 3.7GB virtual
- Safe threshold: ≥ 3.7GB virtual

Source file characteristics that do NOT affect the threshold:
- Line count (10 vs 1000 functions): same threshold
- Monomorphization count (1 vs 300 DType variants): same threshold
- Module-level vs per-function imports: same threshold
- Presence or absence of heavy cross-module import chains: same threshold

## Impact

On GitHub Actions free-tier runners (7GB total RAM, 2 cores):
- OS + runner + pixi env ≈ 2-3GB
- Available for `mojo` processes: 4-5GB
- If 2+ concurrent `mojo` processes overlap (each needing 3.7GB virtual):
  combined demand: 7.4GB+ → OOM crash

This affects any workflow that runs Mojo tests in a matrix with parallel jobs.

## Stack Trace (from crashing CI runs)

```
#0  0x00007f9999f4d4ab in libKGENCompilerRTShared.so (+0x6d4ab)
#1  0x00007f9999f4a686 in libKGENCompilerRTShared.so (+0x6a686)
#2  0x00007f9999f4e157 in libKGENCompilerRTShared.so (+0x6e157)
#3  0x00007f999b145330 in libc.so.6 (sigaction wrapper)
```

## Workarounds

1. **Limit concurrent Mojo test jobs** in CI matrix: `max-parallel: 2`
2. **Use larger runners** (16GB) for Mojo test matrix
3. **Reduce compilation time per process** via per-function imports in library modules —
   shorter compile = shorter window when multiple mojo processes overlap

## Notes

- Physical RSS is only ~330MB even on complex files — the issue is **virtual** address
  space reservation by the JIT compiler, not actual physical memory usage
- The crash is non-deterministic across CI runs depending on runner load and job overlap
- Related to #6187 (runtime allocator crash) but distinct cause: this is JIT initialization
  mapping virtual pages, not a use-after-free
