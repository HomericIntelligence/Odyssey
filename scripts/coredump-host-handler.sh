#!/bin/bash
# coredump-host-handler.sh — pipe-mode core_pattern handler
#
# Invoked by the Linux kernel (as PID 1 of the HOST namespace) when any
# process with `ulimit -c > 0` crashes.  The kernel pipes the full ELF
# core on stdin and passes positional args set via core_pattern tokens.
#
# Install:
#   sudo cp scripts/coredump-host-handler.sh /usr/local/bin/coredump-host-handler.sh
#   sudo chmod +x /usr/local/bin/coredump-host-handler.sh
#   echo "|/usr/local/bin/coredump-host-handler.sh %p %e %t %s %P" \
#     | sudo tee /proc/sys/kernel/core_pattern
#
# core_pattern tokens used:
#   %p  PID of crashing process (in container namespace)
#   %e  executable basename
#   %t  time of crash (seconds since epoch)
#   %s  signal number
#   %P  global PID (host namespace)
#
# Manual test (does NOT hang — stdin TTY guard prevents blocking):
#   ./scripts/coredump-host-handler.sh 1234 mojo 1700000000 11 5678
#   echo "fake core data" | ./scripts/coredump-host-handler.sh 1234 mojo 1700000000 11 5678
#
# Related: modular/modular#6413, PR #5380, PR that introduced this file

set -euo pipefail

PID="${1:-unknown}"
EXE="${2:-unknown}"
TIME="${3:-0}"
SIGNAL="${4:-0}"
# $5 = global (host) PID — captured but not used in filename

# ── TTY guard ────────────────────────────────────────────────────────────────
# When invoked by the kernel, stdin is a pipe carrying the core ELF — not a
# terminal.  When invoked manually for testing with no piped input, stdin IS a
# terminal and reading from it would hang forever.  Bail out early in that case.
if [ -t 0 ]; then
    echo "coredump-host-handler: stdin is a TTY — refusing to run (would block)." >&2
    echo "  This script is meant to be invoked by the kernel via core_pattern." >&2
    echo "  Manual test: echo somecore | $0 <pid> <exe> <time> <signal> [<gpid>]" >&2
    exit 1
fi

# ── Destination directory ─────────────────────────────────────────────────────
# We are running as PID 1 of the HOST.  Walk well-known host-side paths.
# The GitHub Actions runner workspace is at:
#   /home/runner/work/<repo>/<repo>/
# which is the actual host directory that the container bind-mounts as
# /workspace.  Writing here is guaranteed to survive the container teardown
# and appear in the artifact upload step.

TARGET=""
for candidate in \
    /home/runner/work/ProjectOdyssey/ProjectOdyssey/crash-bundle/cores \
    /workspace/crash-bundle/cores \
    /tmp/crash-bundle/cores; do
    if [ -d "$candidate" ]; then
        TARGET="$candidate"
        break
    fi
done

# If none of the above exist yet, fall back to /tmp (create it).
if [ -z "$TARGET" ]; then
    TARGET=/tmp/crash-bundle/cores
fi

mkdir -p "$TARGET"

# ── Write core ELF ───────────────────────────────────────────────────────────
# Cap at 4 GB to prevent filling the runner disk when a runaway process
# generates an enormous core.  Real libKGEN JIT crashes are typically 50-500 MB.
OUT="$TARGET/core.${PID}.${EXE}.${TIME}.sig${SIGNAL}"
# Kernel-invoked pipe handler: the kernel ignores our exit code, so errors here
# cannot propagate to the caller.  Log every failure so handler.log is diagnostic.
if ! head -c $((4 * 1024 * 1024 * 1024)) > "$OUT"; then
    echo "$(date -Iseconds) ERROR: failed to write core to $OUT" >> "$LOG_DIR/handler.log" 2>/dev/null
fi
if ! chmod 644 "$OUT" 2>/dev/null; then
    echo "$(date -Iseconds) WARNING: chmod 644 $OUT failed (file may be unreadable)" >> "$LOG_DIR/handler.log" 2>/dev/null
fi

# ── Log the capture ───────────────────────────────────────────────────────────
LOG_DIR="$(dirname "$TARGET")"
SIZE=$(stat -c %s "$OUT" 2>/dev/null || echo "?")
{
    printf '%s wrote %s (%s bytes) signal=%s exe=%s\n' \
        "$(date -Iseconds)" "$OUT" "$SIZE" "$SIGNAL" "$EXE"
} >> "$LOG_DIR/handler.log" 2>/dev/null
# Kernel-invoked: if the log append failed (e.g., LOG_DIR is unwritable), there is
# no safe way to surface the error — the kernel has no channel for our exit code.
# Failure here is silent by design; the crash-bundle artifact upload step will
# surface a missing handler.log as "(no handler.log — handler was not invoked)".
true
