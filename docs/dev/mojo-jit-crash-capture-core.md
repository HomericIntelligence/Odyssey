# Capturing a Real ELF Core Dump for Mojo JIT Crashes

**Upstream issue**: [modular/modular#6413](https://github.com/modular/modular/issues/6413)
**Related doc**: [mojo-jit-crash-workaround.md](mojo-jit-crash-workaround.md)

## What This Solves

The Mojo runtime registers an **in-process signal handler in `libKGENCompilerRTShared.so`**
that intercepts `SIGABRT` / `SIGILL` from JIT-emitted code. The handler prints a 3-frame
post-handler trace ("backtrace inside the handler") and exits cleanly, so:

- The kernel never generates a core (the process handled the signal itself).
- The pre-signal stack frame — the actual crash site — is gone by the time the handler
  prints anything.

To get a usable core we have to **stop the process in gdb before its own handler runs**,
then `generate-core-file` from inside the debugger.

The wrapper at [`scripts/mojo-under-gdb.sh`](../../scripts/mojo-under-gdb.sh) does exactly
that. CI runs it automatically on the test jobs listed in
[`.github/workflows/comprehensive-tests.yml`](../../.github/workflows/comprehensive-tests.yml)
whenever `MOJO_TEST_UNDER_GDB=1` is set (which it is, by default, for those jobs).

## CI Capture (Default Path)

Every CI run on a branch that touches Mojo tests produces a `crash-bundle-<job>` artifact
per test job. The artifact contains:

```text
crash-bundle/
├── cores/
│   ├── core.gdb.<timestamp>.mojo   ← ELF core (multi-MB to GB)
│   └── gdb-<timestamp>.log         ← symbolic backtrace + threads + shlibs
├── metadata.txt                    ← container / mojo version / host info
└── symbols/                        ← any debug-info bundles uploaded by setup-container
```

If a test job fails because of a JIT crash, the artifact's `gdb-*.log` shows the symbolic
frame at signal time (e.g. `assert_almost_equal () at shared/testing/assertions.mojo:170`).

### Downloading a Crash Bundle

```bash
# List artifacts for a run
gh api repos/HomericIntelligence/ProjectOdyssey/actions/runs/<RUN_ID>/artifacts \
  --jq '.artifacts[] | "\(.name)\t\(.size_in_bytes)"'

# Download a specific bundle
gh run download <RUN_ID> -n crash-bundle-<job> -D ./crash-bundle
```

Artifacts expire after **14 days** by default (configured in
`.github/actions/coredump-capture/action.yml`). If you need a permanent copy, compress
and re-upload to a gist or release before the expiry date:

```bash
xz -k -T 0 -3 crash-bundle/cores/core.gdb.<timestamp>.mojo

gh gist create --public --desc "Mojo JIT crash core (modular#6413)" \
  crash-bundle/cores/core.gdb.<timestamp>.mojo.xz \
  crash-bundle/cores/gdb-<timestamp>.log
```

Gist binary files are limited to 100 MB; cores typically compress 8–10× with `xz -3`
so a 700 MB ELF lands around 80 MB.

## Local Capture (Reproducing a Single Test)

If you have a test that reproduces a crash locally (rare — most are CI-runner specific,
see [mojo-jit-crash-workaround.md](mojo-jit-crash-workaround.md)):

### 1. Container with `gdb` available

`gdb` is required and is **not** in the base development container. Install it inside
the container:

```bash
just podman-up
USER_ID=$(id -u) GROUP_ID=$(id -g) podman compose exec -T -u root projectodyssey-dev \
  bash -c 'apt-get update && apt-get install -y gdb'
```

### 2. Run the test under the wrapper

The wrapper takes a core directory and the rest of the `mojo` argv:

```bash
USER_ID=$(id -u) GROUP_ID=$(id -g) podman compose exec -T projectodyssey-dev bash -lc '
  cd /workspace
  ulimit -c unlimited
  mkdir -p /tmp/cores
  bash scripts/mojo-under-gdb.sh /tmp/cores \
    --Werror -debug-level=line-tables -I /workspace -I . \
    tests/path/to/crashing_test.mojo
  echo "wrapper exit=$?"
  ls -la /tmp/cores/
'
```

Expected outcomes:

| Test result    | wrapper exit                                                    | Files in `/tmp/cores/`                |
| -------------- | --------------------------------------------------------------- | ------------------------------------- |
| Pass           | `0`                                                             | `gdb-<ts>.log` only                   |
| Fail (non-0)   | The test's exit code                                            | `gdb-<ts>.log` only                   |
| Crash (signal) | `128 + signo` (e.g. 132 = SIGILL, 134 = SIGABRT, 139 = SIGSEGV) | `gdb-<ts>.log` + `core.gdb.<ts>.mojo` |

### 3. Inspect the gdb log first

`gdb-<ts>.log` has the symbolic backtrace, threads, and shared library map. That alone is
often enough to identify the crash site without opening the core:

```text
Thread 1 "mojo" received signal SIGILL, Illegal instruction.
assert_almost_equal () at /workspace/shared/testing/assertions.mojo:170
170     var diff = abs(a - b)
[mojo-under-gdb] caught SIGILL; dumping /tmp/cores/core.gdb.<ts>.mojo
Saved corefile /tmp/cores/core.gdb.<ts>.mojo
#0  assert_almost_equal () at /workspace/shared/testing/assertions.mojo:170
#1  test_tensor_dataset_negative_indexing () at tests/.../test_tensor_dataset.mojo:166
#2  main () at tests/.../test_tensor_dataset.mojo:308
...
```

### 4. Open the core in gdb for deeper analysis

```bash
USER_ID=$(id -u) GROUP_ID=$(id -g) podman compose exec -T projectodyssey-dev bash -lc '
  MOJO_BIN=$(pixi run which mojo | tail -1)
  gdb -batch -ex "thread apply all bt full" "$MOJO_BIN" /tmp/cores/core.gdb.<ts>.mojo
'
```

To send the core to Modular without sharing the host's bind-mounted source paths, use
the symbol bundle uploaded as `crash-bundle/symbols/` so they can match the runtime
shared libraries (`libKGENCompilerRTShared.so` etc.) to the captured frames.

## How the Wrapper Works

The wrapper at `scripts/mojo-under-gdb.sh`:

1. **Resolves `mojo` via `pixi run which mojo`**, then runs `pixi run -- gdb ...` so the
   inferior inherits the pixi environment (`MODULAR_HOME`, `PATH`, stdlib search paths).
   Running gdb outside `pixi run` strips that activation and mojo fails to find `std`
   before it can crash.
2. **Uses `gdb` Python events** (not `hook-stop`) to distinguish a real signal from a
   normal exit:
   - `gdb.events.stop` → writes `128 + signo` to an exit-code file, dumps the core,
     captures `bt full` / `info threads` / `info sharedlibrary`.
   - `gdb.events.exited` → writes the inferior's real exit code **unless** a signal was
     already recorded (post-signal kill also fires `Exited`).
3. **Sets `set +e` around the gdb call** so a non-zero status doesn't abort the wrapper
   before it can read the exit-code file.
4. **Exits with the recorded code** so the calling shell sees:
   - `0` / `N` for normal exit code `N`
   - `128 + signo` for any caught signal

Signals handled: `SIGABRT`, `SIGSEGV`, `SIGBUS`, `SIGILL`, `SIGFPE`.
Mojo's `os.abort()` lowers to `llvm.trap` → `SIGILL` on Linux, and observed JIT crashes
in modular/modular#6413 also surface as `SIGILL`, not `SIGABRT`.

### Validating the Wrapper Locally

```bash
# Inside the container (with gdb installed):
ulimit -c unlimited
mkdir -p /tmp/cores

# Should produce wrapper exit=134, core.gdb.*.mojo:
bash scripts/mojo-under-gdb.sh /tmp/cores -c 'kill -ABRT $$'   # via /bin/bash

# Should produce wrapper exit=0, no core:
bash scripts/mojo-under-gdb.sh /tmp/cores -c 'echo ok'

# Should produce wrapper exit=132 (SIGILL), large core:
cat > /tmp/abort.mojo <<EOF
from os import abort
def main():
    abort()
EOF
bash scripts/mojo-under-gdb.sh /tmp/cores -I /tmp /tmp/abort.mojo
```

## Environment Variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `MOJO_TEST_UNDER_GDB` | `0` locally, `1` in CI | Switch in `justfile::_test-group-inner` between `pixi run mojo ...` and `bash scripts/mojo-under-gdb.sh ...` |
| `MOJO_UNDER_GDB` | `1` | Set to `0` to bypass gdb inside the wrapper itself (escape hatch for local dev when gdb is unavailable) |
| `CRASH_BUNDLE_DIR` | `<repo>/crash-bundle/cores` | Where the wrapper writes cores and gdb logs |

## Limitations

- **`gdb` must be available**: not present in the base container; add `gdb` to
  `Dockerfile.ci` before relying on `MOJO_TEST_UNDER_GDB=1` in any environment that
  builds the image from scratch.
- **Cores can be huge**: a Mojo crash dumps the full process address space, frequently
  ≥500 MB. Workflows must allocate disk and the `coredump-capture` action caps the
  bundle at 4 GB.
- **Stripped frames in `libKGENCompilerRTShared.so`**: shipped without debug info, so
  `#0` inside libKGEN shows as `?? ()`. The wrapper still records the surrounding Mojo
  source frames, which is usually enough for Modular to triage.
- **ASLR warning is benign**: `warning: Error disabling address space randomization:
  Function not implemented` comes from the container's seccomp profile blocking
  `personality(2)`. It does not affect signal capture.
