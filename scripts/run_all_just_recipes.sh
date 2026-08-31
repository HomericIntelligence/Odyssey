#!/usr/bin/env bash
set -uo pipefail

# Exercise public just recipes from the host in dependency-aware order and
# preserve every output stream. Destructive, interactive, publishing, and
# multi-hour recipes are intentionally not included in this audit. Every
# recipe is invoked from the host; recipes handle their own container use.

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_ROOT="$ROOT/logs/just-recipes/$STAMP"
mkdir -p "$LOG_ROOT"
SUMMARY="$LOG_ROOT/summary.tsv"
: > "$SUMMARY"

# Fixed order: inspect/configuration, local checks, container startup,
# package/build, tests, documentation, and finally audit checks. Keep build
# immediately after package so compile failures are reported together.
RECIPE_ORDER=(
    show-user
    check-glibc
    check-matmul-calls
    check-note-format
    check-version-sync
    list-models
    podman-preflight
    podman-up
    podman-status
    check
    package-debug
    build-debug
    test-python
    test-mojo
    test-example-backward
    training-smoke-all
    jupyter-validate
    docs
    pre-commit
    audit
)

# These recipes are excluded because they stop/start/remove containers, mutate
# external systems, require an interactive terminal, serve indefinitely, or
# perform long-running builds/training. Run them explicitly when needed.
EXCLUDED='^(default|help|_.*|podman-down|podman-clean|podman-logs|podman-push|podman-push-all|podman-release|podman-run-shell|publish|docs-serve|jupyter|jupyter-notebook|shell|clean-all|bootstrap|download-emnist|install-local|publish-dry-run|podman-build|podman-rebuild|podman-build-ci|podman-build-ci-all|ci-podman-build|ci-podman-validate|podman-test-image|podman-run-tests|infer|infer-image|train|jupyter-clear|build|build-release|build-asan|build-tsan|build-all|ci-build|package|package-release|wheel|validate|test|test-mojo-asan|test-mojo-tsan|bench-precommit|pre-commit-all|test-group|test-group-asan|training-smoke-one|bump-version|clean|clean-worktrees)$'

run_one() {
    local recipe="$1"
    local safe_name="${recipe//[^A-Za-z0-9_.-]/_}"
    local log="$LOG_ROOT/${safe_name}.log"
    local start end status
    start="$(date +%s)"
    {
        echo "===== RECIPE: just $recipe ====="
        echo "UTC start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "Host: $(hostname)"
        echo "Working directory: $ROOT"
        echo
        set +e
        just "$recipe"
        status=$?
        echo
        echo "UTC end: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "Exit status: $status"
        exit "$status"
    } 2>&1 | tee "$log"
    status="${PIPESTATUS[0]}"
    end="$(date +%s)"
    printf '%s\t%s\t%s\t%s\n' "$recipe" "$status" "$((end - start))" "$log" >> "$SUMMARY"
}

printf 'Logs: %s\n' "$LOG_ROOT"
printf 'Summary: %s\n\n' "$SUMMARY"

# Resolve the declared order against the current justfile. This makes removed
# recipes visible as skips rather than silently disappearing from the report.
for recipe in "${RECIPE_ORDER[@]}"; do
    if ! just --summary 2>/dev/null | awk '{for (i = 1; i <= NF; i++) if ($i == "'"$recipe"'") found=1} END {exit !found}'; then
        printf 'SKIP\t%s\t(not present in justfile)\n' "$recipe" | tee -a "$SUMMARY"
        continue
    fi
    if [[ "$recipe" =~ $EXCLUDED ]]; then
        printf 'SKIP\t%s\t(excluded by safety policy)\n' "$recipe" | tee -a "$SUMMARY"
        continue
    fi
    run_one "$recipe"
done

printf '\n===== SUMMARY =====\n'
awk -F '\t' 'BEGIN {printf "%-28s %-8s %-8s %s\n", "RECIPE", "STATUS", "SECONDS", "LOG"} {if ($1 == "SKIP") printf "%-28s %-8s %-8s %s\n", $2, "SKIP", "-", $3; else printf "%-28s %-8s %-8s %s\n", $1, $2, $3, $4}' "$SUMMARY"
printf '\nNon-zero recipe exits:\n'
awk -F '\t' '$1 != "SKIP" && $2 != 0 {print "  " $1 " (exit " $2 "): " $4; failed=1} END {if (!failed) print "  none"}' "$SUMMARY"

# The audit is designed to continue through every recipe; callers can inspect
# summary.tsv for failures without the script masking them as successful.
exit 0
