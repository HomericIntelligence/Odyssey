# ADR-014: 2026-03-28 Strict-Mode Audit Remediation

## Status

Accepted — 2026-06-20

## Context

A comprehensive strict-mode repository audit was performed on 2026-03-28 (issue #5191).
The audit covered 45+ files including 20+ source files, 10 test files, 22 CI workflows,
and all config files. The overall result was B+/87% with no critical blockers.

Twelve actionable findings were identified and tracked as child issues. All twelve have
been resolved as of 2026-06-20.

## Decisions

### Architecture (DRY / SRP)

- **#5181**: Extracted 3 parametric helpers (`_convert_to_fp8_family`, `_convert_to_int_dtype`,
  `_convert_to_block_quant`) to eliminate ~267 lines of duplicated dtype dispatch across 12
  `to_*` methods in `AnyTensor`. (PR #5502)

- **#5182**: Split `any_tensor.mojo` (4,373 lines) into 6 focused sibling modules to reach
  ≤3,000 lines: `tensor_ops`, `tensor_printing`, `tensor_split`, `tensor_indexing`,
  `tensor_dtype_conv`, `tensor_views`. Cross-module private-field access is valid Mojo
  (package-scoped privacy); 22+ existing call sites in `tensor_io/creation/utils` establish
  the pattern. (PR #5503)

### Reliability

- **#5183**: Added training operation timeouts and graceful shutdown via `interruption.mojo`.
  (PR #5490)

- **#5184**: Extended `CheckpointManager` with atomic tmp+rename saves, per-epoch incremental
  writes, `--fresh`/`--resume` recovery modes, and exit codes 0=success/1=error/2=transient/
  130=SIGINT. (PR #5501)

### Source Code Quality

- **#5179**: Fixed latent BFloat16 bug in FP8/BF8 conversion (commit 9a1d508f).
- **#5180**: Fixed `contiguous()` to skip clone when already contiguous (commit 1ab0ebb3).
- **#5185**: Removed unreachable bounds checks in AnyTensor (commit a1acb9b1).
- **#5186**: Cleaned up duplicate imports in test files (commit f5465262).

### Security / Developer Experience

- **#5187**: Added non-root USER directive to Dockerfile.ci (commit 63f5f45d).
- **#5188**: Removed committed .env file from git history (commit b89ea542).
- **#5189**: Added .editorconfig and devcontainer configuration (commit c3ece724).
- **#5190**: Automated version sync across VERSION/pixi.toml/pyproject.toml (commit 45eab7de).

## Consequences

- `any_tensor.mojo` is now ≤3,000 lines with a clear module decomposition pattern
  that can be extended as the file grows.
- Training loops have checkpoint recovery, enabling resumable long-running experiments.
- The audit B+/87% baseline is documented here; future audits can diff against it.
- The 6-module tensor decomposition (`tensor_ops`, `tensor_printing`, etc.) establishes
  the convention for future extractions from `AnyTensor` when the file grows again.
