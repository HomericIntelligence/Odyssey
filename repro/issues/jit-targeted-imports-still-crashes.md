# [BUG] JIT crash persists after targeted import conversion (ADR-015)

## Environment

- Mojo version: 0.26.3 (dev2026040705)
- OS: Ubuntu 24.04 (GitHub Actions runner)
- GLIBC: 2.39
- Runner UID: 1001

## Description

After converting all test files from package-level imports (`from shared.core import …`)
to targeted submodule imports (`from shared.core.layers.linear import Linear`), JIT
compilation crashes are still observed intermittently on CI for the Core Layers test group.

This is significant because ADR-015's hypothesis was that import volume caused the crash.
If crashes persist with targeted imports, the root cause is elsewhere.

## Crash Signature

```text
/path/to/mojo: error: execution crashed
```

Stack (no symbols):

```text
#0 libKGENCompilerRTShared.so+0x6d4ab
#1 libKGENCompilerRTShared.so+0x6a686
#2 libKGENCompilerRTShared.so+0x6e157
#3 libc.so.6+0x45330
```

## Minimal Reproducer

See `repro/repro_jit_targeted_imports_crash.mojo`.

```bash
mojo repro/repro_jit_targeted_imports_crash.mojo
```

## Observed Pattern

- Occurs on GitHub Actions Ubuntu 24.04 runners
- Non-deterministic: same file may pass one run and crash the next
- ASAN tests (AOT compilation path) pass on the same SHA — confirms JIT-specific
- Crash occurs before any test output (during JIT compilation startup)

## Relationship

- Successor to modular/modular#6187 (bitcast UAF JIT crash)
- ADR-015 targeted-import conversion did not eliminate the crash
- May be UID-mismatch HOME directory permission issue (runner UID 1001, image built for UID 1000)
