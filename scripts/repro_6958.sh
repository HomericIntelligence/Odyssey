#!/usr/bin/env bash
# repro_6958.sh — Reproducer for modular/modular#6958
# KGEN JIT runtime crash in libKGENCompilerRTShared.so on Mojo 1.0.0 stable.
# Passes on 1.0.0b2, crashes on 1.0.0 stable.
#
# Intended flow (fresh checkout):
#
#   gh repo clone HomericIntelligence/Odyssey
#   cd Odyssey
#   bash scripts/repro_6958.sh          # or ./scripts/repro_6958.sh
#
# The script mounts the repo directory it is run from into the container at
# /workspace (via the repo's docker-compose.yml), builds/starts the Podman dev
# container, installs both Mojo compilers (1.0.0 stable + 1.0.0b2), and runs
# the two failing tests on each:
#
#   1. b2 baseline:   test_typed_batchnorm / test_mobilenetv1_e2e on 1.0.0b2
#                     (pre-migration commit)   → expect PASS
#   2. stable:        test_typed_batchnorm / test_mobilenetv1_e2e on 1.0.0
#                     (migrated branch)        → expect CRASH
#
# Requirements:
#   - podman (with podman-compose or `just` available in the repo)
#   - git, internet access
#
# Exit codes:
#   0  = bug confirmed (b2 PASS, stable CRASH)
#   1  = setup/validation failed
#   2  = bug not reproduced (stable did not crash)
#   3  = b2 baseline failed (pre-existing issue unrelated to #6958)

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

B2_COMMIT="${B2_COMMIT:-febfcd23}"             # last b2-compatible commit (on main)
STABLE_BRANCH="${STABLE_BRANCH:-5800-mojo-1.0.0}"  # migrated branch (on origin)
MODULAR_INDEX="https://modular.gateway.scarf.sh/simple/"
TEST1="tests/odyssey/tensor/test_typed_batchnorm.mojo"
TEST2="tests/models/test_mobilenetv1_e2e.mojo"
TIMEOUT_S="${TIMEOUT_S:-180}"
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
if ! git rev-parse --verify -q "origin/$STABLE_BRANCH" >/dev/null 2>&1 && \
   ! git rev-parse --verify -q "$STABLE_BRANCH" >/dev/null 2>&1; then
    fail "Cannot find branch $STABLE_BRANCH (needed for the stable side)."
    fail "Check the branch name or fetch from origin."
    exit 1
fi
info "Repo: $REPO_DIR"
info "HEAD: $(git rev-parse --short HEAD) on $ORIG_BRANCH"

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

# ── Step 3: b2 baseline ────────────────────────────────────────────────
header "Step 3: b2 baseline (pre-migration code, expect PASS)"
run_in "cd /workspace && git checkout -q $B2_COMMIT 2>/dev/null || git checkout -q $B2_COMMIT"
info "Checked out $B2_COMMIT for b2"

run_case_b2() {
    local name="$1" file="$2"
    info "Running $name with b2..."
    local out
    out=$(run_in "cd /workspace && timeout $TIMEOUT_S /tmp/mojob2/.venv/bin/mojo run \
        -I src -I . -Xlinker -lm '$file' 2>&1" || true)
    if echo "$out" | grep -qE "All .*passed|All [0-9]+ .*tests? passed"; then
        pass "b2 $name: PASS"
        return 0
    else
        fail "b2 $name: FAIL"
        echo "$out" | tail -6 | sed 's/^/    /'
        return 1
    fi
}

B2_OK=0
run_case_b2 "test_typed_batchnorm" "$TEST1" && B2_OK=$((B2_OK+1))
run_case_b2 "test_mobilenetv1_e2e"  "$TEST2" && B2_OK=$((B2_OK+1))

if [ "$B2_OK" -ne 2 ]; then
    warn "b2 baseline incomplete — cannot validate regression. Exit 3."
    exit 3
fi

# ── Step 4: stable (migrated code, expect CRASH) ──────────────────────
header "Step 4: stable 1.0.0 (migrated code, expect CRASH)"
run_in "cd /workspace && git checkout -q $STABLE_BRANCH"
info "Checked out $STABLE_BRANCH for stable"

info "Compile-check test file on stable..."
if ! run_in "cd /workspace && timeout 300 mojo build -I src -I . -Xlinker -lm \
        $TEST1 -o /dev/null 2>&1" >/dev/null; then
    warn "test_typed_batchnorm does not compile on stable branch (migration incomplete)."
    warn "This means the crash path is not reachable; check branch contents."
fi

run_case_stable() {
    local name="$1" file="$2"
    local status_file="$3"
    info "Running $name with stable..."
    local out
    out=$(run_in "cd /workspace && timeout $TIMEOUT_S mojo run \
        -I src -I . -Xlinker -lm '$file' 2>&1" || true)
    if echo "$out" | grep -q "execution crashed"; then
        local n
        n=$(echo "$out" | grep -c "^PASS:\|^✓" || true)
        fail "stable $name: CRASH after ~$n checks"
        echo "$out" | grep -E "^#|execution crashed" | sed 's/^/    /'
        echo 0 > "$status_file"   # crash is the expected outcome
    elif echo "$out" | grep -qE "All .*passed|All [0-9]+ .*tests? passed"; then
        pass "stable $name: PASS (no crash — bug not reproduced)"
        echo 2 > "$status_file"
    else
        fail "stable $name: UNEXPECTED (compile error or other failure)"
        echo "$out" | tail -8 | sed 's/^/    /'
        echo 1 > "$status_file"
    fi
}

S_OK=0
STATUS_DIR=$(mktemp -d)
run_case_stable "test_typed_batchnorm" "$TEST1" "$STATUS_DIR/bn"
run_case_stable "test_mobilenetv1_e2e"  "$TEST2" "$STATUS_DIR/mn"
S_BN=$(cat "$STATUS_DIR/bn")
S_MN=$(cat "$STATUS_DIR/mn")
[ "$S_BN" = "0" ] && S_OK=$((S_OK+1))
[ "$S_MN" = "0" ] && S_OK=$((S_OK+1))

# ── Restore the caller's checkout ──────────────────────────────────────
info "Restoring checkout to $ORIG_BRANCH..."
if ! run_in "cd /workspace && git checkout -q $ORIG_BRANCH 2>/dev/null || git checkout -q main"; then
    warn "Could not restore checkout; repo left on $STABLE_BRANCH"
fi

# ── Summary ─────────────────────────────────────────────────────────────
header "SUMMARY"
echo -e "  ${CYAN}Case${NC}                    ${CYAN}b2 (1.0.0b2)${NC}   ${CYAN}stable (1.0.0)${NC}"
echo -e "  ───────────────────────── ─────────── ──────────────────"
if [ "$S_BN" = "0" ]; then
    echo -e "  test_typed_batchnorm    ${GREEN}PASS${NC}       ${RED}CRASH${NC}"
else
    echo -e "  test_typed_batchnorm    ${GREEN}PASS${NC}       ${YELLOW}?${NC} (code=$S_BN)"
fi
if [ "$S_MN" = "0" ]; then
    echo -e "  test_mobilenetv1_e2e    ${GREEN}PASS${NC}       ${RED}CRASH${NC}"
else
    echo -e "  test_mobilenetv1_e2e    ${GREEN}PASS${NC}       ${YELLOW}?${NC} (code=$S_MN)"
fi

echo ""
if [ "$S_OK" -eq 2 ]; then
    echo -e "  ${GREEN}VERDICT: Bug confirmed (#6958).${NC}"
    echo "  Both tests PASS on 1.0.0b2 and CRASH on 1.0.0 stable."
    exit 0
elif [ "$S_OK" -eq 1 ]; then
    echo -e "  ${YELLOW}VERDICT: Partially reproduced.${NC}"
    echo "  One test crashed on stable; the other did not."
    exit 2
else
    echo -e "  ${YELLOW}VERDICT: Not reproduced.${NC}"
    echo "  Stable did not crash. May require the full-suite JIT workload."
    exit 2
fi
