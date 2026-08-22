#!/usr/bin/env bash
# repro_6959.sh — Reproducer for modular/modular#6959
# Premature __deinit__ / use-after-free in Mojo 1.0.0 stable.
# A struct's heap storage (unsafe_alloc) is freed while raw pointers from
# _as_atomic() are still live, corrupting Atomic[DType.int64] counters.
# The same logic passes on 1.0.0b2.
#
# Intended flow (fresh checkout):
#
#   gh repo clone HomericIntelligence/Odyssey
#   cd Odyssey
#   git fetch origin 5800-mojo-1.0.0 && git checkout 5800-mojo-1.0.0
#   bash scripts/repro_6959.sh          # or ./scripts/repro_6959.sh
#
# The script mounts the repo directory it is run from into the container at
# /workspace (via the repo's docker-compose.yml), builds/starts the Podman dev
# container, installs both Mojo compilers (1.0.0 stable + 1.0.0b2), and runs
# the two self-contained repros:
#
#   1. b2:    repro/repro_6959_inline_b2.mojo on 1.0.0b2   → expect PASS
#   2. stable: repro/repro_6959_inline.mojo on 1.0.0        → expect FAIL
#
# Requirements:
#   - podman (with podman-compose or `just` available in the repo)
#   - git, internet access
#
# Exit codes:
#   0  = bug confirmed (b2 PASS, stable FAIL)
#   1  = setup/validation failed
#   2  = bug not reproduced (stable did not fail)
#   3  = b2 baseline failed (pre-existing issue unrelated to #6959)

set -euo pipefail

# ── Colors ──────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
pass()  { echo -e "${GREEN}[PASS]${NC}  $*"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
header(){ echo -e "\n${YELLOW}════════════════════════════════════════════════════════════${NC}"; echo -e "${YELLOW}  $*${NC}"; echo -e "${YELLOW}════════════════════════════════════════════════════════════${NC}"; }

# ── Config ──────────────────────────────────────────────────────────────
# Repo root = the directory this script is run from (repo root, or scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/docker-compose.yml" ]; then
    REPO_DIR="$SCRIPT_DIR"
elif [ -f "$(pwd)/docker-compose.yml" ]; then
    REPO_DIR="$(pwd)"
else
    fail "Run this script from the Odyssey repo root (or scripts/ inside it)."
    exit 1
fi

# Force a deterministic compose project name so the container is always
# odyssey_odyssey-dev_1 regardless of the clone directory name.
export COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-odyssey}"
CONTAINER="odyssey_odyssey-dev_1"

STABLE_BRANCH="${STABLE_BRANCH:-5800-mojo-1.0.0}"  # branch holding the repro files
MODULAR_INDEX="https://modular.gateway.scarf.sh/simple/"
REPRO_STABLE="repro/repro_6959_inline.mojo"
REPRO_B2="repro/repro_6959_inline_b2.mojo"
TIMEOUT_S="${TIMEOUT_S:-120}"
SKIP_BUILD="${SKIP_BUILD:-0}"   # 1 = reuse existing container + compilers

# ── Helper: run inside container ────────────────────────────────────────
run_in() {
    podman exec "$CONTAINER" bash -c "$1"
}

check_prereqs() {
    local ok=1
    for cmd in podman git; do
        if ! command -v "$cmd" &>/dev/null; then
            fail "$cmd not found. Please install it first."
            ok=0
        fi
    done
    if ! git -C "$REPO_DIR" rev-parse --git-dir &>/dev/null; then
        fail "$REPO_DIR is not a git repository."
        ok=0
    fi
    if [ "$ok" -eq 0 ]; then
        exit 1
    fi
    return 0
}

# ── Step 0: repo sanity ────────────────────────────────────────────────
header "Step 0: Validate repo ($REPO_DIR)"
check_prereqs
cd "$REPO_DIR"
ORIG_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if ! git fetch origin "$STABLE_BRANCH" 2>/dev/null; then
    info "Fetch failed (offline?); using local refs"
fi
# The repro files live on $STABLE_BRANCH; make sure it is checked out.
if [ ! -f "$REPRO_STABLE" ] || [ ! -f "$REPRO_B2" ]; then
    if git rev-parse --verify -q "$STABLE_BRANCH" >/dev/null 2>&1 || \
       git rev-parse --verify -q "origin/$STABLE_BRANCH" >/dev/null 2>&1; then
        info "Repro files not present on $ORIG_BRANCH — checking out $STABLE_BRANCH..."
        git checkout -q "$STABLE_BRANCH" 2>/dev/null || git checkout -q -b "$STABLE_BRANCH" "origin/$STABLE_BRANCH"
    else
        fail "Cannot find branch $STABLE_BRANCH (holds the repro files)."
        exit 1
    fi
fi
[ -f "$REPRO_STABLE" ] || { fail "Missing $REPRO_STABLE"; exit 1; }
[ -f "$REPRO_B2" ] || { fail "Missing $REPRO_B2"; exit 1; }
info "Repo: $REPO_DIR"
info "HEAD: $(git rev-parse --short HEAD) on $(git rev-parse --abbrev-ref HEAD)"

# ── Step 1: start container with THIS repo mounted ─────────────────────
if [ "$SKIP_BUILD" = "1" ]; then
    info "SKIP_BUILD=1: reusing existing container + compilers"
    STATUS=$(podman inspect --format '{{.State.Status}}' "$CONTAINER" 2>/dev/null || echo missing)
    [ "$STATUS" = "running" ] || { fail "Container not running"; exit 1; }
else
    header "Step 1: Build + start dev container (mounting $REPO_DIR)"
    RUNNING=0
    if podman ps --format '{{.Names}}' 2>/dev/null | grep -q "^${CONTAINER}$"; then
        RUNNING=1
        # Verify the running container mounts THIS repo
        MOUNTED=$(podman inspect --format '{{range .Mounts}}{{.Destination}}={{.Source}} {{end}}' "$CONTAINER" 2>/dev/null \
            | tr ' ' '\n' | grep '^/workspace=' | cut -d= -f2 || true)
        MOUNTED_REAL="$(realpath -m "${MOUNTED:-}" 2>/dev/null || echo "$MOUNTED")"
        REPO_REAL="$(realpath -m "$REPO_DIR")"
        if [ -n "$MOUNTED" ] && [ "$MOUNTED_REAL" != "$REPO_REAL" ]; then
            info "Container mounts $MOUNTED, not $REPO_DIR — recreating with correct mount..."
            if ! podman rm -f "$CONTAINER" >/dev/null 2>&1; then
                warn "Container removal reported a failure; continuing anyway"
            fi
            RUNNING=0
        else
            info "Container $CONTAINER already running with this repo mounted"
        fi
    fi
    if [ "$RUNNING" = "0" ]; then
        if podman ps -a --format '{{.Names}}' 2>/dev/null | grep -q "^${CONTAINER}$"; then
            info "Removing stale container (fresh state required)..."
            podman rm -f "$CONTAINER" >/dev/null 2>&1 || info "No stale container to remove"
        fi
        info "Building + starting container via compose..."
        if command -v just &>/dev/null; then
            just podman-up
        else
            export USER_ID="${USER_ID:-$(id -u)}"
            export GROUP_ID="${GROUP_ID:-$(id -g)}"
            podman compose up -d --build
        fi
        sleep 5
    fi

    STATUS=$(podman inspect --format '{{.State.Status}}' "$CONTAINER" 2>/dev/null || echo down)
    info "Container status: $STATUS"
    [ "$STATUS" = "running" ] || { fail "Container did not start"; exit 1; }

    # ── Step 2: install both compilers ─────────────────────────────────────
    header "Step 2: Install Mojo compilers"
    info "Installing Mojo 1.0.0 stable into container venv..."
    run_in "uv pip install 'mojo==1.0.0' --extra-index-url $MODULAR_INDEX 2>&1 | tail -2"

    info "Installing Mojo 1.0.0b2 into /tmp/mojob2 venv..."
    run_in "mkdir -p /tmp/mojob2 && cd /tmp/mojob2 && \
            uv venv --python python3 2>/dev/null; \
            uv pip install --python .venv/bin/python 'mojo==1.0.0b2' \
                --extra-index-url $MODULAR_INDEX 2>&1 | tail -2"
fi

info "Verifying compilers..."
STABLE_VER=$(run_in "mojo --version")
B2_VER=$(run_in "/tmp/mojob2/.venv/bin/mojo --version")
info "  stable: $STABLE_VER"
info "  b2:     $B2_VER"
case "$STABLE_VER" in
    *"1.0.0 ("*"ed45d567"*) ;;
    *)
        fail "Stable compiler mismatch (expected 1.0.0 ed45d567): $STABLE_VER"
        exit 1
        ;;
esac

# ── Step 3: b2 baseline (expect PASS) ──────────────────────────────────
header "Step 3: b2 (1.0.0b2, b2-flavored repro) — expect PASS"
out_b2=$(run_in "cd /workspace && timeout $TIMEOUT_S /tmp/mojob2/.venv/bin/mojo run \
    $REPRO_B2 2>&1" || true)
if echo "$out_b2" | grep -q "All assertions passed!"; then
    pass "b2 $REPRO_B2: PASS"
    B2_OK=1
else
    fail "b2 $REPRO_B2: FAIL"
    echo "$out_b2" | tail -8 | sed 's/^/    /'
    B2_OK=0
fi

if [ "$B2_OK" -ne 1 ]; then
    warn "b2 baseline incomplete — cannot validate regression. Exit 3."
    exit 3
fi

# ── Step 4: stable (expect FAIL / assertion) ───────────────────────────
header "Step 4: stable 1.0.0 (1.0.0 repro) — expect FAIL"
out_stable=$(run_in "cd /workspace && timeout $TIMEOUT_S mojo run \
    $REPRO_STABLE 2>&1" || true)

S_OK=0
if echo "$out_stable" | grep -q "AssertionError"; then
    fail "stable $REPRO_STABLE: FAIL (assertion hit — bug reproduced)"
    echo "$out_stable" | grep -E "after unlock|after backoff|AssertionError" | sed 's/^/    /'
    S_OK=1
elif echo "$out_stable" | grep -q "All assertions passed!"; then
    pass "stable $REPRO_STABLE: PASS (bug not reproduced)"
    S_OK=0
else
    fail "stable $REPRO_STABLE: UNEXPECTED (compile error or other failure)"
    echo "$out_stable" | tail -8 | sed 's/^/    /'
    S_OK=0
fi

# ── Restore the caller's checkout ──────────────────────────────────────
if [ "$ORIG_BRANCH" != "$(git rev-parse --abbrev-ref HEAD)" ]; then
    info "Restoring checkout to $ORIG_BRANCH..."
    if ! run_in "cd /workspace && git checkout -q $ORIG_BRANCH 2>/dev/null || git checkout -q main"; then
        warn "Could not restore checkout; repo left on $STABLE_BRANCH"
    fi
fi

# ── Summary ─────────────────────────────────────────────────────────────
header "SUMMARY"
echo -e "  ${CYAN}Case${NC}                             ${CYAN}b2 (1.0.0b2)${NC}   ${CYAN}stable (1.0.0)${NC}"
echo -e "  ─────────────────────────────── ─────────── ──────────────────"
if [ "$B2_OK" = "1" ]; then
    echo -e "  inlined SpinLock counter     ${GREEN}PASS${NC}       $([ "$S_OK" = "1" ] && echo -e "${RED}FAIL${NC} (bug reproduced)" || echo -e "${YELLOW}PASS${NC} (not reproduced)")"
else
    echo -e "  inlined SpinLock counter     ${RED}FAIL${NC}       —"
fi

echo ""
if [ "$B2_OK" = "1" ] && [ "$S_OK" = "1" ]; then
    echo -e "  ${GREEN}VERDICT: Bug confirmed (#6959).${NC}"
    echo "  b2 passes, stable fails with the assertion — use-after-free reproduced."
    exit 0
elif [ "$B2_OK" = "1" ]; then
    echo -e "  ${YELLOW}VERDICT: Not reproduced.${NC}"
    echo "  Stable did not fail. The free/op ordering may have landed differently."
    exit 2
else
    exit 3
fi
