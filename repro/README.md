# Crash Reproduction Files

This directory contains minimal reproducers for Mojo runtime crashes
discovered during ProjectOdyssey development. Each file documents a
specific crash with environment details, stack traces, and root cause
analysis.

## Affected Mojo Version

All reproducers target **Mojo 0.26.1** (build 156d3ac6) on Linux x86_64
with GLIBC 2.39 (WSL2). The crashes may or may not reproduce on other
Mojo versions or platforms.

## Files

### Crash Reproducers

- **`repro_crash_standalone.mojo`** -- Self-contained reproducer with no
  external dependencies. Demonstrates a deterministic crash in the Mojo
  runtime allocator triggered by heavy alloc/free churn followed by a
  `UnsafePointer.bitcast` write. Root cause: use-after-free when the
  compiler destroys the source tensor before the bitcast pointer is used.

- **`repro_libkgen_crash.mojo`** -- Same crash pattern as
  `repro_crash_standalone.mojo` but uses ProjectOdyssey's `shared.core`
  library (AnyTensor, conv2d, relu). Requires `just build` first.

- **`repro_libasyncrt_crash.mojo`** -- Variant that reproduces the crash
  through cumulative VGG16-style forward passes rather than a simple
  bitcast write. Demonstrates the same heap corruption through a more
  realistic deep learning workflow.

### Bug Reproduction Archives

- **`bug_repro_lenet5_layers_monolithic.mojo.bug`** -- The original
  24-test monolithic LeNet-5 test file from December 2025 (Issue #2942).
  Crashes after approximately 15 cumulative tests due to the same heap
  corruption bug. This file was split into smaller files as a workaround
  (see ADR-009).

- **`bug_repro_vgg16_e2e_part1_pre_fix.mojo.bug`** -- Pre-workaround
  VGG16 E2E test file that demonstrates the crash during training steps
  involving bitcast writes for target label creation.

### Validation Script

- **`run_all_experiments.sh`** -- Validates all crash claims from the
  Day 53 blog post (`notes/blog/03-16-2026/README.md`). Runs each
  reproducer and checks expected outcomes (crash, pass, ASAN detection).

## Root Cause

All crashes share the same root cause: a use-after-free in Mojo 0.26.1
where `UnsafePointer.bitcast` pointers become dangling after the
compiler applies ASAP destruction to the source tensor. This contradicts
Mojo's documentation on `MutAnyOrigin` wildcard origins which should
prevent early destruction.

Filed upstream: [modular/modular#6187](https://github.com/modular/modular/issues/6187)

## Workaround

Replace `tensor._data.bitcast[T]()[i] = val` with `tensor[i] = val`
(use `__setitem__` instead of direct pointer writes). See the Day 53
blog post for the full investigation.

## Related Issues

- Issue #2942 -- Original LeNet-5 heap corruption (December 2025)
- Issue #2702 -- FC layer gradient checking crashes
- ADR-009 -- Test file splitting workaround
