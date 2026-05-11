#!/usr/bin/env bash
# Pre-commit hook: ban `.__matmul__(` call sites in Mojo files (use
# `matmul(A, B)` instead). Ref #3215.
#
# Refactored out of an inline `entry:` (previously a single 270-char bash -c
# string with a `grep ... || true` chain that swallowed pipefail-rc=1 from
# "no matches"). The script form is auditable, testable, and lets us replace
# the suppression with explicit `awk` filters (always exit 0) per Odysseus PR
# #280's Bucket D guidance.

set -euo pipefail

# `grep -r` returns 1 when no matches are found; that's the *good* case for
# this guard. Disable -e for the search so we can branch on the empty result.
set +e
raw=$(grep -rn '\.__matmul__(' . \
        --include='*.mojo' --include='*.🔥' \
        --exclude-dir='.pixi' --exclude-dir='.git')
set -e

violations=$(printf '%s\n' "$raw" \
  | awk '!/fn __matmul__\(/ && !/# __matmul__/ && !/__matmul__.*deprecated/ && NF')

if [ -n "$violations" ]; then
    echo "Found .__matmul__() call sites (use matmul(A, B) instead):"
    echo "$violations"
    exit 1
fi
