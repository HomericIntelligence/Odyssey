#!/usr/bin/env bash
# mojo-format-compat.sh — GLIBC-aware wrapper for mojo format
#
# On hosts with glibc < 2.32 (e.g., Debian 10/Buster), the Mojo binary fails with
# "GLIBC_2.32 not found". This wrapper detects that error and exits 0 (skip)
# with a warning instead of failing the commit. CI runs on Ubuntu 24.04
# (glibc 2.39) where this wrapper behaves identically to calling mojo format directly.
#
# See docs/dev/mojo-glibc-compatibility.md for details.
# Closes #3170

set -uo pipefail

output=$(pixi run mojo format "$@" 2>&1)
exit_code=$?

if echo "$output" | grep -q "GLIBC_2\." && echo "$output" | grep -q "not found"; then
    echo "WARNING: mojo-format skipped: host glibc is incompatible with Mojo binary."
    echo "         Mojo requires GLIBC_2.32+. Your system has an older glibc."
    echo "         Files were NOT reformatted. Run inside Docker for full formatting."
    echo "         See docs/dev/mojo-glibc-compatibility.md for details."
    exit 0
fi

echo "$output"
exit $exit_code
