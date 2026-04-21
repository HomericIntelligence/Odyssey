# Mojo JIT maps ~3.5GB virtual address space per compiler invocation

## Environment

- Mojo 0.26.3.0.dev2026040705 (69cac1bd)
- GLIBC 2.39-0ubuntu8.7 (Ubuntu 24.04)
- Linux x86_64, 16GB RAM (confirmed passes), 7GB RAM (crashes)

## Summary

Every `mojo run` / `mojo build` invocation maps approximately **3.5GB of virtual
address space** regardless of the source file's complexity. This exceeds the
available virtual address space on GitHub Actions free-tier runners (7GB total)
when multiple `mojo` processes run concurrently, causing OOM crashes.

## Reproduction

Requires Docker or Podman. The commands below use Podman — replace `podman` with
`docker` if preferred.

```bash
# 1. Clone the repo (contains the minimal reproducer and Dockerfile)
git clone https://github.com/HomericIntelligence/ProjectOdyssey.git
cd ProjectOdyssey
REPO_ROOT="$(pwd)"

# 2. Install the Mojo toolchain into the pixi environment
pixi install

# 3. Build the container image from the repo's Dockerfile
podman build -t projectodyssey:repro .

# 4. Crash case: virtual limit below mojo's reservation (~3.5GB)
podman run --rm \
  -v "$REPO_ROOT:$REPO_ROOT:z" \
  --user "$(id -u):$(id -g)" \
  projectodyssey:repro \
  bash -c "ulimit -v 3500000 && MODULAR_HOME=$REPO_ROOT/.pixi/envs/default/share/max $REPO_ROOT/.pixi/envs/default/bin/mojo run $REPO_ROOT/repro/repro_hello.mojo"

# Expected output:
# JIT session error: Cannot allocate memory
# mojo: error: Failed to materialize symbols: { (exec, { main }) }

# 5. Pass case: virtual limit above mojo's reservation
podman run --rm \
  -v "$REPO_ROOT:$REPO_ROOT:z" \
  --user "$(id -u):$(id -g)" \
  projectodyssey:repro \
  bash -c "ulimit -v 4000000 && MODULAR_HOME=$REPO_ROOT/.pixi/envs/default/share/max $REPO_ROOT/.pixi/envs/default/bin/mojo run $REPO_ROOT/repro/repro_hello.mojo"

# Expected output:
# hello
```

The minimal reproducer file (`repro/repro_hello.mojo`) contains only:

```mojo
def main() raises:
    print("hello")
```

## Measured Data

VmPeak / VmRSS measured via `/proc/$PID/status` polling during a normal (no ulimit) run:

| Metric | Value |
|--------|-------|
| VmPeak (virtual peak) | 3,705,232 kB (~3.53 GB) |
| VmRSS (physical RAM) | ~321,000 kB (~314 MB) |

Virtual address space crash threshold (5 runs each):

| `ulimit -v` (kB) | Pass rate | Notes |
|------------------|-----------|-------|
| 3,500,000 | 0/5 (0%) | Reliable crash — use for repro |
| 3,700,000 | 1/5 (20%) | Non-deterministic (ASLR variance) |
| 3,750,000 | 4/5 (80%) | Non-deterministic |
| 3,800,000 | 5/5 (100%) | Reliable pass |
| 4,000,000 | 5/5 (100%) | Reliable pass — use for repro |

Source file characteristics that do NOT affect the threshold:

- Line count (10 vs 1000 functions): same threshold
- Monomorphization count (1 vs 300 DType variants): same threshold
- Module-level vs per-function imports: same threshold
- Presence or absence of heavy cross-module import chains: same threshold

## Impact

On GitHub Actions free-tier runners (7GB total RAM, 2 cores):

- OS + runner + pixi env ≈ 2-3GB
- Available for `mojo` processes: 4-5GB
- If 2+ concurrent `mojo` processes overlap (each needing ~3.5GB virtual):
  combined demand: 7GB+ → OOM crash

This affects any workflow that runs Mojo tests in a matrix with parallel jobs.

## Error Message

When the crash is triggered by `ulimit -v`, the output is:

```text
JIT session error: Cannot allocate memory
mojo: error: Failed to materialize symbols: { (exec, { main }) }
```

In CI without `ulimit`, the process is OOM-killed by the kernel and the signal
is caught by `libKGENCompilerRTShared.so`, producing a stack trace like:

```text
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

- Physical RSS is only ~314MB even on complex files — the issue is **virtual** address
  space reservation by the JIT compiler, not actual physical memory usage
- The non-determinism in the 3.7–3.75GB range is caused by ASLR shifting the base
  addresses of mmap regions between runs
- The crash is non-deterministic across CI runs depending on runner load and job overlap
- Related to #6187 (runtime allocator crash) but distinct cause: this is JIT initialization
  mapping virtual pages, not a use-after-free
