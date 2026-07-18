#!/usr/bin/env bash
# Run one optimizer's or layer's Mojo unit test by name — a single-command
# entry point for the individual primitives.
#
# Usage:
#   bash scripts/run_primitive_test.sh <name>     # run one primitive's test
#   bash scripts/run_primitive_test.sh all        # run every primitive test
#   bash scripts/run_primitive_test.sh --list     # list known primitives + paths
#
# Examples:
#   bash scripts/run_primitive_test.sh sophia
#   bash scripts/run_primitive_test.sh gru
#   bash scripts/run_primitive_test.sh all
#
# A primitive whose test file is not present on the current branch is reported
# as SKIPPED (not a failure) — so this runner works incrementally as each
# primitive PR merges. Runs inside the uv env, mirroring CI's include paths
# (`mojo -I src -I . <test>`; CI additionally passes --Werror).

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

# name -> test file path. Optimizers under tests/.../training/optimizers/,
# layers under tests/.../core/layers/. Keep alphabetical within each group.
declare -A TEST
# --- optimizers ---
TEST[adopt]="tests/odyssey/training/optimizers/test_adopt.mojo"
TEST[adan]="tests/odyssey/training/optimizers/test_adan.mojo"
TEST[ftrl]="tests/odyssey/training/optimizers/test_ftrl.mojo"
TEST[kl_shampoo]="tests/odyssey/training/optimizers/test_kl_shampoo.mojo"
TEST[lionmuon]="tests/odyssey/training/optimizers/test_lionmuon.mojo"
TEST[mgup_muon]="tests/odyssey/training/optimizers/test_mgup_muon.mojo"
TEST[muon_hyperball]="tests/odyssey/training/optimizers/test_muon_hyperball.mojo"
TEST[prodigy]="tests/odyssey/training/optimizers/test_prodigy.mojo"
TEST[schedule_free]="tests/odyssey/training/optimizers/test_schedule_free.mojo"
TEST[schedule_free_plus]="tests/odyssey/training/optimizers/test_schedule_free_plus.mojo"
TEST[sf_normuon]="tests/odyssey/training/optimizers/test_sf_normuon.mojo"
TEST[soap]="tests/odyssey/training/optimizers/test_soap.mojo"
TEST[splus]="tests/odyssey/training/optimizers/test_splus.mojo"
TEST[sophia]="tests/odyssey/training/optimizers/test_sophia.mojo"
# --- layers ---
TEST[attention]="tests/odyssey/core/layers/test_attention.mojo"
TEST[sparse_attention]="tests/odyssey/core/layers/test_sparse_attention.mojo"
TEST[deepsets]="tests/odyssey/core/layers/test_deepsets.mojo"
TEST[ffn]="tests/odyssey/core/layers/test_feedforward.mojo"
TEST[gru]="tests/odyssey/core/layers/test_gru.mojo"
TEST[kan]="tests/odyssey/core/layers/test_kan.mojo"
TEST[layernorm]="tests/odyssey/core/layers/test_layernorm.mojo"
TEST[linear_attention]="tests/odyssey/core/layers/test_linear_attention.mojo"
TEST[lstm]="tests/odyssey/core/layers/test_lstm.mojo"
TEST[ltc]="tests/odyssey/core/layers/test_ltc.mojo"
TEST[mamba]="tests/odyssey/core/layers/test_mamba.mojo"
TEST[mlp_mixer]="tests/odyssey/core/layers/test_mlp_mixer.mojo"
TEST[rnn]="tests/odyssey/core/layers/test_rnn.mojo"
TEST[ssm]="tests/odyssey/core/layers/test_ssm.mojo"
TEST[transformer]="tests/odyssey/core/layers/test_transformer.mojo"

run_one() {
    local name="$1"
    local path="${TEST[$name]:-}"
    if [[ -z "$path" ]]; then
        echo "❌ unknown primitive '$name' (see: bash scripts/run_primitive_test.sh --list)"
        return 2
    fi
    if [[ ! -f "$path" ]]; then
        echo "⏭  SKIP $name — $path not on this branch yet"
        return 3
    fi
    echo "▶  $name  ($path)"
    if uv run mojo -I src -I . "$path"; then
        echo "✅ $name PASSED"
        return 0
    else
        echo "❌ $name FAILED"
        return 1
    fi
}

case "${1:-}" in
    ""|-h|--help)
        sed -n '2,18p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
        exit 0
        ;;
    --list)
        echo "Known primitives (name -> test file):"
        for k in $(printf '%s\n' "${!TEST[@]}" | sort); do
            mark=" "; [[ -f "${TEST[$k]}" ]] || mark="⏭"
            printf "  %s %-16s %s\n" "$mark" "$k" "${TEST[$k]}"
        done
        exit 0
        ;;
    all)
        fail=0; ran=0; skipped=0
        for k in $(printf '%s\n' "${!TEST[@]}" | sort); do
            run_one "$k"; rc=$?
            case $rc in
                0) ran=$((ran+1)) ;;
                1) ran=$((ran+1)); fail=1 ;;
                3) skipped=$((skipped+1)) ;;
            esac
        done
        echo "--------------------------------------------------"
        echo "primitives run: $ran   skipped (not on branch): $skipped"
        if [[ $fail -eq 1 ]]; then echo "❌ some primitive tests failed"; exit 1; fi
        echo "✅ all present primitive tests passed"
        exit 0
        ;;
    *)
        run_one "$1"
        rc=$?
        # a clean skip (3) is not a hard failure for a single named run either,
        # but signal it distinctly from pass(0)/fail(1)/unknown(2).
        exit $rc
        ;;
esac
