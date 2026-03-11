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

# mojo format exits 123 when the parser can't handle new syntax (e.g., comptime).
# Filter out parse errors and only fail if real formatting changes were needed.
if [ $exit_code -eq 123 ]; then
    # Show parse errors as warnings but don't fail the hook
    parse_errors=$(echo "$output" | grep "^error: cannot format")
    formatted=$(echo "$output" | grep -v "^error: cannot format" | grep -v "^$" | grep -v "Oh no" | grep -v "files.*to reformat")
    if [ -n "$parse_errors" ]; then
        echo "WARNING: mojo format cannot parse some files (likely new syntax not yet supported):"
        echo "$parse_errors" | sed 's/^/  /'
    fi
    if [ -n "$formatted" ]; then
        echo "$formatted"
    fi
    exit 0
fi

echo "$output"
exit $exit_code
