#!/usr/bin/env bash
# repro_6958.sh — Reproducer for modular/modular#6958
# KGEN JIT runtime crash in libKGENCompilerRTShared.so on Mojo 1.0.0 stable.
# Passes on 1.0.0b2, crashes on 1.0.0 stable.
#
# This script:
#   1. Ensures the Odyssey repo is checked out
#   2. Builds + starts the Podman dev container
#   3. Installs both Mojo compilers (1.0.0 stable + 1.0.0b2)
#   4. Runs test_typed_batchnorm and test_mobilenetv1_e2e on b2  (expect PASS)
#   5. Checks out the migrated branch and runs both tests on stable (expect CRASH)
#
# Usage:
#   chmod +x repro_6958.sh
#   ./repro_6958.sh
#
# Requirements:
#   - podman (with podman-compose or `just` available in repo)
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
header(){ echo -e "\n${YELLOW}════════════════════════════════════════════════════════════${NC}"; echo -e "${YELLOW}  $*${NC}"; echo -e "${YELLOW}════════════════════════════════════════════════════════════${NC}"; }

# ── Config ──────────────────────────────────────────────────────────────
REPO_DIR="${REPO_DIR:-$(pwd)}"                 # default to cwd (assume inside repo)
B2_COMMIT="${B2_COMMIT:-febfcd23}"             # last b2-compatible commit
STABLE_BRANCH="${STABLE_BRANCH:-5800-mojo-1.0.0}"  # migrated branch (pushed to origin)
CONTAINER="odyssey_odyssey-dev_1"
MODULAR_INDEX="https://modular.gateway.scarf.sh/simple/"
TEST1="tests/odyssey/tensor/test_typed_batchnorm.mojo"
TEST2="tests/models/test_mobilenetv1_e2e.mojo"
TIMEOUT_S="${TIMEOUT_S:-180}"
SKIP_BUILD="${SKIP_BUILD:-0}"   # 1 = skip container build + compiler install (use existing)

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
        fail "$REPO_DIR is not a git repository. Set REPO_DIR=<path-to-Odyssey>."
        ok=0
    fi
    if [ "$ok" -eq 0 ]; then
        exit 1
    fi
    return 0
}

# ── Step 1: repo sanity ────────────────────────────────────────────────
header "Step 1: Validate repo"
check_prereqs
cd "$REPO_DIR"
if ! git fetch origin "$STABLE_BRANCH" 2>/dev/null; then
    info "Fetch failed (offline?); using local refs"
fi
info "Repo: $REPO_DIR"
info "HEAD: $(git rev-parse --short HEAD)"

# ── Step 2: build + start container ────────────────────────────────────
if [ "$SKIP_BUILD" = "1" ]; then
    info "SKIP_BUILD=1: using existing container + compilers"
else
    header "Step 2: Build and start Podman dev container"
    if podman ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
        info "Container $CONTAINER already running"
    else
        if podman ps -a --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
            info "Removing stale container and rebuilding (fresh state required)..."
            podman rm -f "$CONTAINER" >/dev/null 2>&1 || info "No stale container to remove"
        fi
        info "Building + starting container via compose..."
        if command -v just &>/dev/null; then
            just podman-up
        else
            podman compose up -d
        fi
        sleep 5
    fi
    STATUS=$(podman inspect --format '{{.State.Status}}' "$CONTAINER" 2>/dev/null || echo down)
    info "Container status: $STATUS"
    [ "$STATUS" = "running" ] || { fail "Container did not start"; exit 1; }

    # ── Step 3: install both compilers ─────────────────────────────────────
    header "Step 3: Install Mojo compilers"
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

# ── Step 4: b2 baseline ────────────────────────────────────────────────
header "Step 4: b2 baseline (pre-migration code, expect PASS)"
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

# ── Step 5: stable (migrated code, expect CRASH) ──────────────────────
header "Step 5: stable 1.0.0 (migrated code, expect CRASH)"
run_in "cd /workspace && git checkout -q $STABLE_BRANCH"
info "Checked out $STABLE_BRANCH for stable"

# The migrated branch compiles on stable; apply no workarounds — we want
# the crash as-is. (If the branch does not compile, the repro is invalid.)
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
