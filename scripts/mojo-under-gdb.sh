#!/bin/bash
# Run `mojo` under gdb to catch the in-process SIGABRT handler in libKGEN
# (modular/modular#6413) and dump a real ELF core via generate-core-file.
#
# Without gdb interception, libKGEN's SIGABRT handler catches the abort,
# prints its own 3-frame post-handler trace, and exits cleanly — the
# kernel never generates a core because the process handles the signal itself.
# Running under gdb forces the signal to stop in the debugger BEFORE the
# user handler runs, so we get the real pre-signal frame frozen in the core.
#
# Usage:
#   mojo-under-gdb.sh <core-dir> <mojo-args...>
#
#   <core-dir>   Directory where cores and gdb logs are written (created if absent)
#   <mojo-args>  All remaining arguments are passed verbatim to `pixi run mojo`
#
# Environment variables:
#   MOJO_UNDER_GDB=0   Skip gdb and exec mojo directly (local dev escape hatch)
#
# Output files (on crash):
#   <core-dir>/core.gdb.<timestamp>.mojo  — ELF core (multi-MB)
#   <core-dir>/gdb-<timestamp>.log        — full backtrace + threads + shlibs

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "[mojo-under-gdb] usage: $0 <core-dir> <mojo-args...>" >&2
    exit 1
fi

CORE_DIR="$1"
shift

# Escape hatch: MOJO_UNDER_GDB=0 bypasses gdb for local dev where overhead matters.
if [ "${MOJO_UNDER_GDB:-1}" = "0" ]; then
    exec pixi run mojo "$@"
fi

mkdir -p "$CORE_DIR"
TS=$(date +%s)
GDB_LOG="${CORE_DIR}/gdb-${TS}.log"
CORE_FILE="${CORE_DIR}/core.gdb.${TS}.mojo"

# Resolve the real mojo binary so gdb has a concrete ELF file argument.
# `pixi run which mojo` prints any activation preamble on stderr; `tail -1`
# grabs the last line (the actual path) to handle noisy pixi output.
# pixi run which mojo may print activation preamble on stderr; tail -1 grabs the
# actual path from the last line.  If the command fails (pixi not set up), MOJO_BIN
# is empty and the check below falls back to direct exec.
MOJO_BIN=$(pixi run which mojo 2>/dev/null | tail -1) || MOJO_BIN=""
if [ -z "$MOJO_BIN" ] || [ ! -x "$MOJO_BIN" ]; then
    echo "[mojo-under-gdb] WARNING: could not resolve mojo binary; falling back to direct exec" >&2
    exec pixi run mojo "$@"
fi

# Write the gdb command script to a temp file to avoid multi-line -ex quoting issues.
# gdb multi-line strings via -ex are not portable across gdb versions; -x is reliable.
GDB_SCRIPT=$(mktemp /tmp/mojo-gdb-XXXXXX.gdb)
# shellcheck disable=SC2064
trap "rm -f '$GDB_SCRIPT'" EXIT

cat > "$GDB_SCRIPT" <<GDBEOF
set pagination off
set confirm off
set logging file ${GDB_LOG}
set logging overwrite on
set logging enabled on

# Intercept crash signals before user (libKGEN) handlers run.
# "nopass" prevents the signal from being delivered to the inferior.
handle SIGABRT stop nopass print
handle SIGSEGV stop nopass print
handle SIGBUS  stop nopass print

# Dump core + context on stop (signal caught). On normal exit, this hook
# is never invoked, so a clean test run produces no spurious gdb errors.
define hook-stop
  printf "[mojo-under-gdb] stop reason captured; dumping core to ${CORE_FILE}\\n"
  generate-core-file ${CORE_FILE}
  bt full
  info threads
  info sharedlibrary
end

run

set logging enabled off
quit
GDBEOF

echo "[mojo-under-gdb] gdb log  : ${GDB_LOG}" >&2
echo "[mojo-under-gdb] core file: ${CORE_FILE} (written on crash)" >&2
echo "[mojo-under-gdb] binary   : ${MOJO_BIN}" >&2
echo "[mojo-under-gdb] args     : $*" >&2

# --args passes everything after it verbatim as the inferior's argv.
# stdout/stderr from the mojo process flow through gdb normally so CI
# logs remain readable.
#
# Run gdb INSIDE `pixi run` so the inferior inherits the activated pixi
# env (MODULAR_HOME, PATH, MOJO stdlib search paths). Running gdb outside
# `pixi run` strips that activation and mojo fails with "unable to locate
# module 'std'" before it ever has a chance to crash.
#
# `--return-child-result` makes gdb exit with the inferior's exit code
# (or 128+signo on signal). Without this, gdb's own zero/non-zero status
# masks the test result and we either always pass or always fail.
exec pixi run -- gdb -batch -nx --return-child-result -x "$GDB_SCRIPT" --args "$MOJO_BIN" "$@"
