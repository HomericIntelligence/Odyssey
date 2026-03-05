# Mojo GLIBC Compatibility

## Problem

The Mojo binary requires `GLIBC_2.32`, `GLIBC_2.33`, and `GLIBC_2.34`, which were introduced
in glibc 2.32 (released 2020). Debian 10 (Buster) ships with glibc 2.28, which is too old.

When running `pixi run mojo format` on an incompatible host, the binary fails with:

```text
/home/user/.pixi/envs/default/bin/mojo: /lib/x86_64-linux-gnu/libc.so.6: version
`GLIBC_2.32' not found
```

This caused all `mojo-format` pre-commit hook invocations to fail, forcing contributors on
Debian 10 hosts to use `SKIP=mojo-format` as a persistent workaround.

**Tracking**: Issue #3170, follow-up from #3076.

## Affected Environments

| OS | glibc version | Status |
|----|--------------|--------|
| Debian 10 (Buster) | 2.28 | Incompatible - hook skips with warning |
| Debian 11 (Bullseye) | 2.31 | Incompatible - hook skips with warning |
| Debian 12 (Bookworm) | 2.36 | Compatible |
| Ubuntu 20.04 (Focal) | 2.31 | Incompatible - hook skips with warning |
| Ubuntu 22.04 (Jammy) | 2.35 | Compatible |
| Ubuntu 24.04 (Noble) | 2.39 | Compatible (CI environment) |

## Solution

A wrapper script `scripts/mojo-format-compat.sh` wraps `pixi run mojo format` and detects
the GLIBC error at runtime. On incompatible hosts it exits `0` (skip) with a clear warning
message instead of exiting `1` (failure). On compatible hosts it behaves identically to
calling `mojo format` directly.

The `.pre-commit-config.yaml` has been updated to use this wrapper with `language: script`
so pre-commit invokes it directly without needing `pixi run` as the entrypoint.

### Wrapper Behavior

- **Compatible host (glibc >= 2.32)**: Runs `mojo format`, propagates exit code and output
- **Incompatible host (glibc < 2.32)**: Exits `0` with a visible warning; files are NOT reformatted

### Why Exit 0 on Incompatible Hosts?

The GLIBC version is a host-environment constraint, not a code defect. Blocking commits on
contributors' machines because of an OS version mismatch creates unnecessary friction without
improving code quality. CI runs on Ubuntu 24.04 (glibc 2.39) and will always run the real
formatter, ensuring all merged code is properly formatted.

## Long-Term Resolution Options

1. **Upgrade host OS** to Debian 12+ or Ubuntu 22.04+ (preferred, eliminates the constraint)
2. **Use Docker for commits**: Run `just shell` to open a shell inside the Docker container,
   then commit from there — the container has a compatible glibc and runs full formatting
3. **Wait for Modular**: If Modular publishes a Mojo build targeting glibc 2.28, pinning that
   version in `pixi.toml` would allow direct compatibility

## Developer Workflow on Incompatible Hosts

Contributors on Debian 10 or other incompatible hosts no longer need `SKIP=mojo-format`.
The hook will run and automatically skip with a warning:

```text
WARNING: mojo-format skipped: host glibc is incompatible with Mojo binary.
         Mojo requires GLIBC_2.32+. Your system has an older glibc.
         Files were NOT reformatted. Run inside Docker for full formatting.
         See docs/dev/mojo-glibc-compatibility.md for details.
```

To format Mojo files on an incompatible host, run inside Docker:

```bash
just shell
# Inside container:
pixi run mojo format path/to/file.mojo
```
