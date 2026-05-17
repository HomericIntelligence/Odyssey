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
EXIT_CODE_FILE="${CORE_DIR}/exit-${TS}.code"
# shellcheck disable=SC2064
trap "rm -f '$GDB_SCRIPT'" EXIT

# The gdb script uses Python event hooks because:
#  1. `handle SIGABRT stop nopass` + a `hook-stop` block fires on EVERY stop
#     event in gdb 15.1 — including normal exit, where `generate-core-file`
#     fails with "You can't do that without a process to debug" and gdb
#     prints "Error while running hook_stop". A bare gdb-script `if` on
#     $_siginfo also fails after normal exit (no stack).
#  2. `--return-child-result` is unreliable in gdb 15.1 -batch mode for
#     processes killed by handled signals: gdb often exits 0 even when the
#     inferior died on SIGABRT, masking the test failure entirely.
#
# Python events distinguish gdb.SignalEvent (real crash) from gdb.ExitedEvent
# (clean exit) and record the desired exit code to a file. The wrapper then
# reads that file and exits with the recorded code, so the caller sees:
#   - 0 / N for normal exit code N
#   - 128 + signo for any caught signal (134 for SIGABRT, 139 for SIGSEGV…)
cat > "$GDB_SCRIPT" <<GDBEOF
set pagination off
set confirm off
set logging file ${GDB_LOG}
set logging overwrite on
set logging enabled on

python
import gdb
EXIT_FILE = "${EXIT_CODE_FILE}"
CORE_FILE = "${CORE_FILE}"
# POSIX shell convention for signal-terminated processes.
SIG_MAP = {"SIGABRT": 6, "SIGSEGV": 11, "SIGBUS": 7, "SIGFPE": 8, "SIGILL": 4}
state = {"signaled": False}

def write_exit(code):
    with open(EXIT_FILE, "w") as f:
        f.write(str(code))

# Default = 1 covers the case where neither handler fires (e.g. gdb itself dies).
write_exit(1)

def on_stop(event):
    # Only act on signal stops (we never set breakpoints, so any other
    # stop event is unexpected and best left to gdb's defaults).
    if isinstance(event, gdb.SignalEvent):
        signo = event.stop_signal
        print("[mojo-under-gdb] caught " + signo + "; dumping " + CORE_FILE)
        gdb.execute("generate-core-file " + CORE_FILE)
        gdb.execute("bt full")
        gdb.execute("info threads")
        gdb.execute("info sharedlibrary")
        state["signaled"] = True
        write_exit(128 + SIG_MAP.get(signo, 1))

def on_exit(event):
    # gdb fires Exited after the post-signal kill too. If we already
    # captured a signal, do not overwrite that exit code.
    if state["signaled"]:
        return
    code = getattr(event, "exit_code", None)
    write_exit(code if code is not None else 0)

gdb.events.stop.connect(on_stop)
gdb.events.exited.connect(on_exit)
end

# Intercept crash signals before user (libKGEN) handlers run.
# "nopass" prevents the signal from being delivered to the inferior.
# SIGILL is included because Mojo's os.abort() raises llvm.trap → SIGILL,
# and observed JIT crashes in modular/modular#6413 also surface as SIGILL.
handle SIGABRT stop nopass print
handle SIGSEGV stop nopass print
handle SIGBUS  stop nopass print
handle SIGILL  stop nopass print
handle SIGFPE  stop nopass print

run

set logging enabled off
quit
GDBEOF

echo "[mojo-under-gdb] gdb log  : ${GDB_LOG}" >&2
echo "[mojo-under-gdb] core file: ${CORE_FILE} (written on crash)" >&2
echo "[mojo-under-gdb] binary   : ${MOJO_BIN}" >&2
echo "[mojo-under-gdb] args     : $*" >&2

# Run gdb INSIDE `pixi run` so the inferior inherits the activated pixi
# env (MODULAR_HOME, PATH, MOJO stdlib search paths). Running gdb outside
# `pixi run` strips that activation and mojo fails with "unable to locate
# module 'std'" before it ever has a chance to crash.
# `set -e` would abort here if gdb exits non-zero before we can read
# EXIT_CODE_FILE. Disable it just for this invocation.
set +e
pixi run -- gdb -batch -nx -x "$GDB_SCRIPT" --args "$MOJO_BIN" "$@"
gdb_status=$?
set -e

# Prefer the Python-recorded exit code; fall back to gdb's own status if
# the file is missing (gdb crashed before the python hook fired).
if [ -r "$EXIT_CODE_FILE" ]; then
    inferior_exit=$(cat "$EXIT_CODE_FILE")
    rm -f "$EXIT_CODE_FILE"
    exit "$inferior_exit"
fi
exit "$gdb_status"
