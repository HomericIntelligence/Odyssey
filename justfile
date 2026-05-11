# ML Odyssey Build System
# Unified build/test/lint interface for Podman containers

# Default recipe - show help
default: help

# Podman compose service name
podman_service := "projectodyssey-dev"

# Repository root
repo_root := justfile_directory()

# Build output root — defaults to <repo>/build, can be overridden by a parent
# meta-repo via env var (e.g. Odysseus sets BUILD_ROOT=/path/to/Odysseus/build/ProjectOdyssey)
BUILD_ROOT := env_var_or_default("BUILD_ROOT", repo_root / "build")


# ==============================================================================
# Automatically detect host UID/GID if not set
# ==============================================================================
USER_ID := `sh -c 'echo ${USER_ID:-$(id -u)}'`
GROUP_ID := `sh -c 'echo ${GROUP_ID:-$(id -g)}'`

show-user:
	@echo "USER_ID = {{USER_ID}}"
	@echo "GROUP_ID = {{GROUP_ID}}"

# ==============================================================================
# Mojo Compiler Flags
# ==============================================================================

# Strict analysis flags (applied to ALL builds)
MOJO_STRICT := "--Werror"

# Mode-specific flags
MOJO_DEBUG := "-g2 --no-optimization"
MOJO_RELEASE := "-g0 -O3"
MOJO_TEST := "-g1"

# Sanitizer flags
MOJO_ASAN := "--sanitize address"
MOJO_TSAN := "--sanitize thread"

# ==============================================================================
# Internal Helpers
# ==============================================================================

# Run command in Podman compose container
# If BUILD_ROOT is outside the repo, mounts it into the container at /ext-build
# and rewrites paths in cmd accordingly.
[private]
_run cmd:
	#!/usr/bin/env bash
	set -e
	if command -v podman &>/dev/null && \
		podman ps -q --filter "name={{podman_service}}" --filter "status=running" 2>/dev/null \
		| grep -q .; then
		BUILD_ROOT="{{BUILD_ROOT}}"
		REPO_ROOT="{{repo_root}}"
		# If BUILD_ROOT is outside the workspace, mount it and rewrite paths in cmd
		if [[ "$BUILD_ROOT" != "$REPO_ROOT"* ]]; then
			mkdir -p "$BUILD_ROOT"
			ORIGINAL_CMD='{{cmd}}'
			REWRITTEN_CMD="${ORIGINAL_CMD//$BUILD_ROOT/\/ext-build}"
			podman compose run --rm \
				-e USER_ID={{USER_ID}} -e GROUP_ID={{GROUP_ID}} \
				-v "$BUILD_ROOT":/ext-build:Z \
				{{podman_service}} bash -c "$REWRITTEN_CMD"
		else
			# Prepend a HOME-fixup fragment so mojo/pixi can always write their
			# caches even when the container image was built with a different UID
			# than the process UID (rootless Podman UID-mapping on CI runners).
			# Also bump core ulimit unconditionally — compose-level `ulimits`
			# does not propagate through `podman compose exec` invocations,
			# and we need real cores when the libKGEN JIT crashes
			# (modular/modular#6413). Pairs with the host-side core_pattern in
			# .github/actions/coredump-capture/action.yml.
			HOME_FIXUP='if [ ! -w "$HOME" ]; then export HOME="/tmp/mojo-home-$(id -u)"; mkdir -p "$HOME/.modular" "$HOME/.pixi"; fi; ulimit -c unlimited 2>/dev/null || echo "warn: ulimit -c rejected" >&2;'
			# Forward CI/coredump env vars into the container. Without these
			# -e passthroughs, MOJO_TEST_UNDER_GDB / CRASH_BUNDLE_DIR set on
			# the GHA runner are NOT visible to the bash recipe inside the
			# container, so the gdb wrapper would be silently skipped.
			podman compose exec \
				-e USER_ID={{USER_ID}} -e GROUP_ID={{GROUP_ID}} \
				-e MOJO_TEST_UNDER_GDB \
				-e MOJO_UNDER_GDB \
				-e CRASH_BUNDLE_DIR \
				-T {{podman_service}} bash -c "$HOME_FIXUP {{cmd}}"
		fi
	else
		echo "Error: Podman compose container '{{podman_service}}' is not running."
		echo "  Start Podman:     just podman-up"
		exit 1
	fi

# Ensure build directory exists
[private]
_ensure_build_dir mode:
    @mkdir -p "{{BUILD_ROOT}}/{{mode}}"

# ==============================================================================
# Podman Management
# ==============================================================================

# Start Podman development environment
#
# If a container already exists but is bind-mounted to a different clone
# (e.g. `~/ProjectOdyssey` vs `~/Projects/ProjectOdyssey`), tear it down
# first so `compose up` creates a fresh container mounted on `$PWD`. This
# lets the same dev work seamlessly across multiple clones without
# manually running `podman compose down` from the old clone first.
#
# docker-compose substitutes ${USER_ID}/${GROUP_ID} in docker-compose.yml from
# its OWN environment, not from just's template vars. Without explicit
# `USER_ID=... GROUP_ID=...` exports on the same line, the variables expand to
# empty strings and the container would get `user: ":"` (invalid), causing
# the compose call to fail or create a stuck container in an invalid state.
podman-up:
    #!/usr/bin/env bash
    set -euo pipefail
    export USER_ID="{{USER_ID}}" GROUP_ID="{{GROUP_ID}}" USER_NAME="${USER_NAME:-dev}"
    EXPECTED="{{repo_root}}"
    EXISTING=$(podman inspect projectodyssey-{{podman_service}}-1 \
        --format '{{{{range .Mounts}}{{{{if eq .Destination "/workspace"}}{{{{.Source}}{{{{end}}{{{{end}}' 2>/dev/null || echo "")
    if [ -n "$EXISTING" ] && [ "$EXISTING" != "$EXPECTED" ]; then
        echo "⚠️  Container exists with workspace mount '$EXISTING'"
        echo "    but this invocation is from '$EXPECTED'."
        echo "    Recreating container against the current clone..."
        podman compose down
    fi
    podman compose up -d {{podman_service}}
    # Rootless Podman UID-maps the bind-mounted workspace so the container
    # user cannot write to host-owned files. Make the workspace world-writable
    # so `just build`, test fixtures, and pre-commit can create output files.
    # (Safe: ephemeral dev machine; matches what the CI setup-container action does.)
    # Log unexpected failures (e.g. socket inodes) to stderr; do not abort.
    if ! chmod -R a+rwX .; then
        echo "warn: 'chmod -R a+rwX .' encountered files it could not modify" >&2
    fi

# Stop Podman development environment
podman-down:
    @USER_ID={{USER_ID}} GROUP_ID={{GROUP_ID}} USER_NAME=${USER_NAME:-dev} \
        podman compose down

# Build Podman images
podman-build:
    @podman compose build \
        --build-arg USER_ID={{USER_ID}} \
        --build-arg GROUP_ID={{GROUP_ID}} \
        --build-arg USER_NAME=dev

# Rebuild Podman images (no cache)
podman-rebuild:
    @podman compose build --no-cache \
        --build-arg USER_ID={{USER_ID}} \
        --build-arg GROUP_ID={{GROUP_ID}} \
        --build-arg USER_NAME=dev

# View Podman container logs
podman-logs:
    @podman compose logs -f {{podman_service}}

# Clean Podman resources
podman-clean:
    @podman compose down -v --rmi local

# Show Podman container status
podman-status:
    @podman compose ps

# ==============================================================================
# Container Registry (GHCR)
# ==============================================================================

# Default registry
REGISTRY := "ghcr.io"
REPO_NAME := "homericintelligence/projectodyssey"

# Build CI-optimized container image
podman-build-ci target="runtime":
    @podman build --format docker -f Dockerfile.ci --target {{target}} \
        -t {{REGISTRY}}/{{REPO_NAME}}:{{target}} \
        -t {{REGISTRY}}/{{REPO_NAME}}:{{target}}-$(git rev-parse --short HEAD) \
        .

# Build all CI image targets
podman-build-ci-all:
    @just podman-build-ci runtime
    @just podman-build-ci ci
    @just podman-build-ci production

# Push container image to GHCR (requires `podman login ghcr.io`)
podman-push target="runtime":
    @podman push {{REGISTRY}}/{{REPO_NAME}}:{{target}}
    @podman push {{REGISTRY}}/{{REPO_NAME}}:{{target}}-$(git rev-parse --short HEAD)

# Push all images to GHCR
podman-push-all:
    @just podman-push runtime
    @just podman-push ci
    @just podman-push production

# Build and push container image
podman-release target="runtime":
    @just podman-build-ci {{target}}
    @just podman-push {{target}}

# Test container image locally
podman-test-image target="runtime":
    @echo "Testing {{target}} image..."
    @podman run --rm {{REGISTRY}}/{{REPO_NAME}}:{{target}} pixi run mojo --version
    @echo "✅ Image {{target}} is working"

# Run tests in container image
podman-run-tests target="runtime":
    @podman run --rm {{REGISTRY}}/{{REPO_NAME}}:{{target}} \
        pixi run mojo test -I . tests/

# Interactive shell in container image
podman-run-shell target="runtime":
    @podman run -it --rm {{REGISTRY}}/{{REPO_NAME}}:{{target}} bash

# ==============================================================================
# CI Container Build (matches GitHub Actions)
# ==============================================================================

# Build container image exactly as CI does
ci-podman-build:
    @podman build --format docker \
        --file Dockerfile.ci \
        --target runtime \
        --platform linux/amd64 \
        -t {{REGISTRY}}/{{REPO_NAME}}:local \
        .

# Validate container build without pushing
ci-podman-validate:
    @echo "Validating Dockerfile.ci..."
    @podman build -f Dockerfile.ci --target runtime . >/dev/null
    @echo "✅ Dockerfile.ci is valid"

# ==============================================================================
# Build Recipes (mojo build - compile/validate Mojo files)
# ==============================================================================

# Build/compile Mojo files with mode-specific flags
build mode="debug":
    @just _run "just _build-inner {{mode}}"

[private]
_build-inner mode="debug":
    #!/usr/bin/env bash
    set -euo pipefail

    # ------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------

    MODE="{{mode}}"             # debug | release | test | ci | asan | tsan
    REPO_ROOT="$(pwd)"
    STRICT="--Werror"

    # tsan needs single-job execution to work around modular/modular#6413
    # (libKGEN JIT crashes under thread sanitizer with parallel codegen).
    JOBS=""

    case "$MODE" in
        debug)
            FLAGS="-g --no-optimization $STRICT"
            ;;
        release)
            FLAGS="-g0 -O3 $STRICT"
            ;;
        test)
            FLAGS="-g1 $STRICT"
            ;;
        ci)
            FLAGS="-g1 $STRICT"
            ;;
        asan)
            FLAGS="-g1 --sanitize address $STRICT"
            ;;
        tsan)
            FLAGS="-g1 --sanitize thread $STRICT"
            JOBS="-j1"
            ;;
        *)
            echo "❌ Unknown mode: $MODE"
            echo "Valid modes: debug | release | test | ci | asan | tsan"
            exit 1
            ;;
    esac

    BUILD_DIR="{{BUILD_ROOT}}/$MODE"
    mkdir -p "$BUILD_DIR"

    echo "🔨 Building Mojo files"
    echo "Mode:  $MODE"
    echo "Flags: $FLAGS${JOBS:+ $JOBS}"
    echo "Out:   $BUILD_DIR"
    echo

    # ------------------------------------------------------------
    # Phase A: Package the shared library
    # ------------------------------------------------------------
    # shared/ is a library (no main() entry points) — must be compiled with
    # `mojo package`, not file-by-file `mojo build`. This is the same logic
    # as `_package-inner`; we inline it here so `just build` produces a
    # complete artifact set in one invocation.

    # `mojo package` only accepts a small subset of compiler flags
    # (no -g / --no-optimization / -O3 / --sanitize). Pass STRICT only.
    echo "→ Packaging shared library"
    pixi run mojo package $STRICT -I "$REPO_ROOT" shared \
        -o "$BUILD_DIR/ProjectOdyssey-shared.mojopkg"

    # ------------------------------------------------------------
    # Phase B: AOT-compile every executable (examples, benchmarks, papers)
    # ------------------------------------------------------------
    # Sanitizer modes skip Phase B: the Mojo compiler under ASAN/TSAN
    # allocates 3-6 GB per invocation (modular/modular#6433), so sweeping
    # 46 binaries OOMs hosts with <32 GB RAM. Sanitizers are valuable for
    # runtime race/leak detection in tests, not for compile-time validation
    # of example binaries. Use `just test-mojo-asan` / `just test-mojo-tsan`
    # to exercise sanitizers against the test suite.
    if [ "$MODE" = "asan" ] || [ "$MODE" = "tsan" ]; then
        echo
        echo "✅ Sanitizer build complete (shared package only)"
        echo "  shared package: $BUILD_DIR/ProjectOdyssey-shared.mojopkg"
        echo "  Run sanitized tests: just test-mojo-${MODE}"
        exit 0
    fi

    # `-Xlinker -lm` is required for files that transitively use libm
    # symbols (fmaxf, sincos, etc.). Mirrors `_test-group-asan-inner` at
    # justfile:791 which already uses this flag successfully.
    #
    # set -euo pipefail aborts on the first failed compile. No silent
    # FAIL_ON_ERROR=0 escape hatch — Werror means Werror.

    # Collect candidate files, then keep only those with a main() entry
    # point. `grep -l ... -- file1 file2 ...` (single invocation) is more
    # robust than `xargs -I{} grep` — xargs treats a no-match exit 1 as a
    # child failure (exit 123). `|| true` swallows the no-match exit code.
    CANDIDATES=$(
        {
            find examples -name "*.mojo" \
                -not -path "*/.*" \
                -not -name "__init__.mojo" \
                -not -name "model.mojo"
            find benchmarks -name "*.mojo" \
                -not -path "*/.*" \
                -not -name "__init__.mojo"
            find papers -path "*/examples/*.mojo" \
                -not -path "*/.*" \
                -not -name "__init__.mojo"
        } 2>/dev/null | sort -u
    )
    EXEC_FILES=""
    if [ -n "$CANDIDATES" ]; then
        EXEC_FILES=$(echo "$CANDIDATES" | tr '\n' '\0' \
            | xargs -0 grep -lE "^fn main|^def main" 2>/dev/null || true)
    fi

    # Disable -e for the per-file loop so a single failure doesn't abort
    # the whole sweep — collect failures, report at end, exit non-zero if any.
    set +e
    BUILT=0
    FAILED=0
    FAILED_FILES=""
    while IFS= read -r file; do
        [ -z "$file" ] && continue
        # Flatten path → unique binary name: examples/foo/bar.mojo → examples_foo_bar
        rel="${file#./}"
        out_name="${rel%.mojo}"
        out_name="${out_name//\//_}"
        out="$BUILD_DIR/$out_name"

        echo "→ Building: $file"
        if pixi run mojo build $FLAGS ${JOBS:-} -I "$REPO_ROOT" \
                -Xlinker -lm "$file" -o "$out" 2>&1; then
            BUILT=$((BUILT + 1))
        else
            FAILED=$((FAILED + 1))
            FAILED_FILES="${FAILED_FILES}\n  - $file"
        fi
    done <<< "$EXEC_FILES"
    set -e

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    echo
    echo "=================================================="
    echo "Build summary ($MODE)"
    echo "=================================================="
    echo "  shared package: $BUILD_DIR/ProjectOdyssey-shared.mojopkg"
    echo "  executables:    $BUILT built, $FAILED failed"
    if [ "$FAILED" -gt 0 ]; then
        echo
        echo "❌ Failed files:"
        echo -e "$FAILED_FILES"
        exit 1
    fi
    echo "✅ Build successful"

# Build debug version
build-debug: (build "debug")

# Build release version
build-release: (build "release")

# Build with AddressSanitizer
build-asan: (build "asan")

# Build with ThreadSanitizer (uses -j1, modular/modular#6413 workaround)
build-tsan: (build "tsan")

# Build all modes
build-all:
    @just build debug
    @just build release
    @just build test
    @just build asan
    @just build tsan

# CI: Build and validate all Mojo code (entry points + shared library)
ci-build:
    @echo "Running CI build validation..."
    @just build ci
    @just package ci
    @echo "✅ CI build validation complete"

# Type-check shared library (compile without producing artifacts)
check:
    @just _run "just _check-inner"

[private]
_check-inner:
    #!/usr/bin/env bash
    set -euo pipefail
    REPO_ROOT="$(pwd)"
    echo "🔍 Type-checking shared/ library..."

    # Use mojo package to validate compilation (no --check flag available)
    OUT=$(mktemp -d)
    trap "rm -rf $OUT" EXIT

    if pixi run mojo package --Werror -I "$REPO_ROOT" shared -o "$OUT/shared.mojopkg" 2>&1; then
        echo "✅ shared/ type-check passed"
    else
        echo "❌ shared/ has compilation errors"
        exit 1
    fi

# ==============================================================================
# Package Recipes (mojo package - create .mojopkg libraries)
# ==============================================================================

# Package shared library with mode-specific flags
package mode="debug":
    @just _run "just _package-inner {{mode}}"

[private]
_package-inner mode="debug":
    #!/usr/bin/env bash
    set -e
    REPO_ROOT="$(pwd)"
    STRICT="--Werror"

    MODE="{{mode}}"
    mkdir -p "build/$MODE"

    case "$MODE" in
        debug)   FLAGS="$STRICT" ;;
        release) FLAGS="$STRICT" ;;
        test)    FLAGS="$STRICT" ;;
        *)       FLAGS="$STRICT" ;;
    esac

    echo "Packaging shared library in $MODE mode..."
    echo "Flags: $FLAGS"
    pixi run mojo package $FLAGS -I "$REPO_ROOT" shared -o "build/$MODE/ProjectOdyssey-shared.mojopkg"

# Package debug version
package-debug: (package "debug")

# Package release version
package-release: (package "release")

# ==============================================================================
# Model Training and Inference
# ==============================================================================

# Train a model (default: LeNet-5 on EMNIST). Accepts lenet/lenet5 as aliases for lenet_emnist.
train model="lenet_emnist" precision="fp32" epochs="10" batch_size="32" lr="0.001":
    #!/usr/bin/env bash
    MODEL="{{model}}"
    if [ "$MODEL" = "lenet" ] || [ "$MODEL" = "lenet5" ]; then MODEL="lenet_emnist"; fi
    echo "Training $MODEL with precision={{precision}}, epochs={{epochs}}"
    just _run "pixi run mojo run -I . examples/$MODEL/run_train.mojo \
        --epochs {{epochs}} --batch-size {{batch_size}} \
        --lr {{lr}} --precision {{precision}}"

# Run inference on test set. Accepts lenet/lenet5 as aliases for lenet_emnist.
infer model="lenet_emnist" checkpoint="lenet5_weights":
    #!/usr/bin/env bash
    MODEL="{{model}}"
    if [ "$MODEL" = "lenet" ] || [ "$MODEL" = "lenet5" ]; then MODEL="lenet_emnist"; fi
    echo "Running inference for $MODEL with checkpoint={{checkpoint}}"
    just _run "pixi run mojo run -I . examples/$MODEL/run_infer.mojo \
        --checkpoint {{checkpoint}} --test-set"

# Run inference on single image. Accepts lenet/lenet5 as aliases for lenet_emnist.
infer-image checkpoint image_path model="lenet_emnist":
    #!/usr/bin/env bash
    MODEL="{{model}}"
    if [ "$MODEL" = "lenet" ] || [ "$MODEL" = "lenet5" ]; then MODEL="lenet_emnist"; fi
    echo "Running inference on {{image_path}}"
    just _run "pixi run mojo run -I . examples/$MODEL/run_infer.mojo \
        --checkpoint {{checkpoint}} --image {{image_path}}"

# Download EMNIST dataset (balanced split) and flatten to datasets/emnist/
download-emnist:
    #!/usr/bin/env bash
    set -euo pipefail
    just _run "pixi run python scripts/download_emnist.py emnist datasets/emnist --split balanced"
    just _run "bash -c 'cd datasets/emnist && for f in gzip/emnist-balanced-*.gz; do [ -f \"\$f\" ] && gunzip -c \"\$f\" > \"\$(basename \"\$f\" .gz)\"; done'"

# List available models
list-models:
    @echo "Available models:"
    @ls -d examples/*/ 2>/dev/null | xargs -I{} basename {} | sed 's/-.*//g' | sort -u || echo "No models found"

# ==============================================================================
# Jupyter Notebooks
# ==============================================================================

# Launch Jupyter Lab
jupyter:
    @pixi run -e notebook jupyter lab --no-browser --ip=127.0.0.1

# Launch Jupyter Notebook (classic)
jupyter-notebook:
    @pixi run -e notebook jupyter notebook --no-browser --ip=127.0.0.1

# Execute all notebooks (for CI validation)
jupyter-validate:
    #!/usr/bin/env bash
    set -e
    NOTEBOOK_COUNT=0
    PASSED=0
    FAILED=0
    for notebook in notebooks/*.ipynb; do
        if [ -f "$notebook" ]; then
            echo "Validating: $notebook"
            NOTEBOOK_COUNT=$((NOTEBOOK_COUNT + 1))
            if pixi run -e notebook jupyter nbconvert --execute --to notebook --inplace "$notebook" 2>&1; then
                PASSED=$((PASSED + 1))
            else
                echo "❌ Failed to validate: $notebook"
                FAILED=$((FAILED + 1))
            fi
        fi
    done
    echo ""
    echo "Validated $PASSED/$NOTEBOOK_COUNT notebooks"
    if [ $FAILED -gt 0 ]; then
        echo "❌ $FAILED notebooks failed validation"
        exit 1
    fi

# Clear all notebook outputs
jupyter-clear:
    #!/usr/bin/env bash
    set -e
    NOTEBOOK_COUNT=0
    for notebook in notebooks/*.ipynb; do
        if [ -f "$notebook" ]; then
            echo "Clearing: $notebook"
            NOTEBOOK_COUNT=$((NOTEBOOK_COUNT + 1))
            pixi run -e notebook jupyter nbconvert --clear-output --inplace "$notebook"
        fi
    done
    echo "Cleared outputs from $NOTEBOOK_COUNT notebooks"

# ==============================================================================
# Development
# ==============================================================================

# One-command development environment setup (run after cloning)
bootstrap:
    #!/usr/bin/env bash
    set -e
    echo "Setting up ML Odyssey development environment..."
    echo ""

    # Step 1: Install pixi dependencies
    echo "==> Installing pixi dependencies..."
    pixi install
    echo "    Done."
    echo ""

    # Step 2: Install pre-commit hooks
    echo "==> Installing pre-commit hooks..."
    pixi run pre-commit install
    echo "    Done."
    echo ""

    # Step 3: Validate the setup
    echo "==> Validating setup..."
    echo -n "    Mojo:    "; pixi run mojo --version 2>/dev/null || echo "not available (use Podman for Mojo)"
    echo -n "    Python:  "; python3 --version 2>/dev/null || echo "not found"
    echo -n "    Just:    "; just --version 2>/dev/null || echo "not found"
    echo -n "    Pixi:    "; pixi --version 2>/dev/null || echo "not found"
    echo ""

    # Step 4: GLIBC compatibility check (informational only — host may lack glibc)
    echo "==> Checking GLIBC compatibility..."
    if ! just check-glibc 2>/dev/null; then
        echo "    (glibc check skipped or reported incompatibility — see docs/dev/mojo-glibc-compatibility.md)"
    fi
    echo ""

    echo "Bootstrap complete! Next steps:"
    echo "  just help          Show available commands"
    echo "  just podman-up     Start Podman dev environment"
    echo "  just test          Run all tests"
    echo "  just build         Build the project"

# Open development shell. Auto-starts the container if it isn't running, so
# new developers don't need to remember to `just podman-up` first (#5329).
shell:
    #!/usr/bin/env bash
    set -euo pipefail
    if ! podman compose ps --services --filter "status=running" 2>/dev/null | grep -q "^{{podman_service}}$"; then
        echo "Container '{{podman_service}}' is not running — starting it now (just podman-up)..."
        just podman-up
    fi
    podman compose exec -it -e USER_ID={{USER_ID}} -e GROUP_ID={{GROUP_ID}} {{podman_service}} bash

# Serve documentation
docs-serve:
    @mkdocs serve || echo "Install mkdocs for documentation"

# Build documentation
docs:
    @mkdocs build || echo "Install mkdocs for documentation"

# ==============================================================================
# CI/CD
# ==============================================================================

# Check GLIBC compatibility with Mojo (Issue #3824)
check-glibc:
    #!/usr/bin/env bash
    echo "Checking GLIBC compatibility with Mojo..."
    required_glibc="2.32"
    current_glibc=$(ldd --version | head -1 | awk '{print $NF}')
    echo "Current GLIBC: $current_glibc"
    echo "Required GLIBC: $required_glibc+"

    # Compare versions (requires awk/shell arithmetic)
    if [ -z "$current_glibc" ]; then
        echo "❌ Could not determine GLIBC version"
        exit 1
    fi

    # Simple version comparison: convert X.Y to integer XY for comparison
    current=$(echo "$current_glibc" | sed 's/\.//g')
    required=$(echo "$required_glibc" | sed 's/\.//g')

    if [ "$current" -ge "$required" ]; then
        echo "✅ GLIBC version is compatible"
        exit 0
    else
        echo "⚠️  WARNING: GLIBC $current_glibc is older than required $required_glibc"
        echo "   Mojo tests will fail locally. Use Podman for testing:"
        echo "   just podman-up"
        echo "   just shell"
        echo "   just test-mojo"
        exit 0
    fi

# Run pre-commit hooks
pre-commit:
    @pre-commit run

pre-commit-all:
    @pre-commit run --all-files

# Enforce no .__matmul__() call sites in Mojo files (use matmul(A, B) instead). Ref #3215
check-matmul-calls:
    #!/usr/bin/env bash
    set -e
    # Bucket D: `grep -v` returns 1 with no matches. Disable -e around the
    # pipeline so "no .__matmul__() call sites at all" doesn't fail the recipe.
    set +e
    raw=$(grep -rn "\.__matmul__(" . --include="*.mojo" --include="*.🔥" --exclude-dir=".pixi" --exclude-dir=".git")
    set -e
    violations=$(printf '%s\n' "$raw" | awk '!/fn __matmul__\(/ && !/# __matmul__/ && !/__matmul__.*deprecated/ && NF')
    if [ -n "$violations" ]; then
        echo "Found .__matmul__() call sites (use matmul(A, B) instead):"
        echo "$violations"
        exit 1
    fi
    echo "No .__matmul__() call sites found."

# Check NOTE format in all Mojo files
check-note-format:
    @python3 scripts/check_note_format.py

# Time pre-commit hooks on all files (for local benchmarking)
bench-precommit:
    #!/usr/bin/env bash
    set -e
    FILE_COUNT=$(git ls-files | wc -l | tr -d ' ')
    START=$SECONDS
    pixi run pre-commit run --all-files
    ELAPSED=$((SECONDS - START))
    echo ""
    echo "Hook runtime: ${ELAPSED}s"
    python3 scripts/bench_precommit.py --elapsed "$ELAPSED" --files "$FILE_COUNT" --status passed

# CI: Full validation (build + package + test)
validate:
    @echo "Validating configuration..."
    @test -f pixi.toml && echo "✅ pixi.toml exists"
    @test -f docker-compose.yml && echo "✅ docker-compose.yml exists"
    @echo "Running full CI validation..."
    @just ci-build
    @just test-mojo
    @echo "✅ Validation complete"

# ==============================================================================
# Testing
# ==============================================================================

# Run all tests
test:
    @echo "Running all tests..."
    @just test-mojo
    @just test-python

# Run only Python tests
test-python:
    @just _run "pixi run pytest tests/ -v --timeout=300"

# Test group of Mojo files with -Werror enforcement
test-group path pattern:
    @just _run "just _test-group-inner '{{path}}' '{{pattern}}'"

# Run a test group under AddressSanitizer
test-group-asan path pattern:
    @just _run "just _test-group-asan-inner '{{path}}' '{{pattern}}'"

[private]
_test-group-inner path pattern:
    #!/usr/bin/env bash
    set -e
    REPO_ROOT="$(pwd)"
    TEST_PATH="{{path}}"
    test_count=0
    passed_count=0
    failed_count=0
    failed_tests=""

    # Enable core dumps so the libKGEN JIT crash (modular/modular#6413)
    # produces a real coredump we can attach to gdb. Sandboxed runners may
    # forbid raising ulimit — log and continue.
    if ! ulimit -c unlimited 2>/dev/null; then
        echo "warn: 'ulimit -c unlimited' rejected by environment; coredumps may be unavailable" >&2
    fi

    echo "=================================================="
    echo "Testing: {{path}}"
    echo "Pattern: {{pattern}}"
    echo "=================================================="

    # Expand pattern into actual files
    test_files=""
    for pattern in {{pattern}}; do
        if [[ "$pattern" == *"*"* ]]; then
            # Glob pattern - expand it
            for file in $TEST_PATH/$pattern; do
                if [ -f "$file" ]; then
                    test_files="$test_files $file"
                fi
            done
        else
            # Direct file or subdirectory pattern
            if [ -f "$TEST_PATH/$pattern" ]; then
                test_files="$test_files $TEST_PATH/$pattern"
            elif [[ "$pattern" == *"/"* ]]; then
                for file in $TEST_PATH/$pattern; do
                    if [ -f "$file" ]; then
                        test_files="$test_files $file"
                    fi
                done
            fi
        fi
    done

    if [ -z "$test_files" ]; then
        echo "❌ ERROR: No test files found in {{path}} matching {{pattern}}"
        echo "   This usually means the directory is empty or was renamed."
        echo "   Fix: update the test group path/pattern in comprehensive-tests.yml"
        exit 1
    fi

    # gdb wrapper path: set MOJO_TEST_UNDER_GDB=1 in CI to intercept the
    # in-process SIGABRT handler in libKGEN before it swallows the crash
    # (modular/modular#6413). Default 0 so local dev runs mojo directly.
    CORE_DIR="${CRASH_BUNDLE_DIR:-${REPO_ROOT}/crash-bundle/cores}"

    # Run each test file
    for test_file in $test_files; do
        if [ -f "$test_file" ]; then
            echo ""
            echo "Running: $test_file"
            test_count=$((test_count + 1))

            if ! ulimit -v unlimited 2>/dev/null; then
                echo "warn: 'ulimit -v unlimited' rejected by environment" >&2
            fi
            test_exit=0
            # gdb-wrapper branch (CI default): intercept libKGEN's in-process
            # SIGABRT handler before it swallows the crash (modular/modular#6413).
            # Local dev defaults to MOJO_TEST_UNDER_GDB=0 → direct mojo invocation.
            if [ "${MOJO_TEST_UNDER_GDB:-0}" = "1" ]; then
                if ! bash "$REPO_ROOT/scripts/mojo-under-gdb.sh" "$CORE_DIR" \
                        --Werror -debug-level=line-tables -I "$REPO_ROOT" -I . "$test_file"; then
                    test_exit=$?
                fi
            else
                if ! pixi run mojo --Werror -debug-level=line-tables -I "$REPO_ROOT" -I . "$test_file"; then
                    test_exit=$?
                fi
            fi
            if [ "${test_exit}" -eq 0 ]; then
                passed_count=$((passed_count + 1))
            else
                failed_count=$((failed_count + 1))
                failed_tests="$failed_tests\n  - $test_file"
            fi
        fi
    done

    echo ""
    echo "=================================================="
    echo "Summary"
    echo "=================================================="
    echo "Total: $test_count tests"
    echo "Passed: $passed_count tests"
    echo "Failed: $failed_count tests"

    # Guard against no tests being run (beyond initial empty check)
    if [ $test_count -eq 0 ]; then
        echo ""
        echo "❌ ERROR: No tests were executed"
        exit 1
    fi

    if [ $failed_count -gt 0 ]; then
        echo ""
        echo "Failed tests:"
        echo -e "$failed_tests"
        exit 1
    fi

[private]
_test-group-asan-inner path pattern:
    #!/usr/bin/env bash
    set -e
    REPO_ROOT="$(pwd)"
    TEST_PATH="{{path}}"
    test_count=0
    passed_count=0
    failed_count=0
    failed_tests=""

    echo "=================================================="
    echo "ASAN Testing: {{path}}"
    echo "Pattern: {{pattern}}"
    echo "=================================================="

    # Expand pattern into actual files
    test_files=""
    for pattern in {{pattern}}; do
        if [[ "$pattern" == *"*"* ]]; then
            # Glob pattern - expand it
            for file in $TEST_PATH/$pattern; do
                if [ -f "$file" ]; then
                    test_files="$test_files $file"
                fi
            done
        else
            # Direct file or subdirectory pattern
            if [ -f "$TEST_PATH/$pattern" ]; then
                test_files="$test_files $TEST_PATH/$pattern"
            elif [[ "$pattern" == *"/"* ]]; then
                for file in $TEST_PATH/$pattern; do
                    if [ -f "$file" ]; then
                        test_files="$test_files $file"
                    fi
                done
            fi
        fi
    done

    if [ -z "$test_files" ]; then
        echo "❌ ERROR: No test files found in {{path}} matching {{pattern}}"
        exit 1
    fi

    for test_file in $test_files; do
        if [ -f "$test_file" ]; then
            echo ""
            echo "Running (ASAN): $test_file"
            test_count=$((test_count + 1))

            BINARY=$(mktemp /tmp/mojo-asan-XXXXXX)
            output=""
            output2=""
            if output=$(pixi run mojo build {{MOJO_ASAN}} {{MOJO_STRICT}} -I "$REPO_ROOT" -I . -Xlinker -lm "$test_file" -o "$BINARY" 2>&1) && output2=$("$BINARY" 2>&1); then
                echo "$output2"
                echo "✅ PASSED: $test_file"
                passed_count=$((passed_count + 1))
            else
                echo "$output"
                echo "$output2"
                echo "❌ FAILED: $test_file"
                failed_count=$((failed_count + 1))
                failed_tests="$failed_tests\n  - $test_file"
            fi
            rm -f "$BINARY"
        fi
    done

    echo ""
    echo "=================================================="
    echo "ASAN Summary"
    echo "=================================================="
    echo "Total: $test_count tests"
    echo "Passed: $passed_count tests"
    echo "Failed: $failed_count tests"

    if [ $test_count -eq 0 ]; then
        echo "❌ ERROR: No tests were executed"
        exit 1
    fi

    if [ $failed_count -gt 0 ]; then
        echo ""
        echo "Failed tests:"
        echo -e "$failed_tests"
        exit 1
    fi

# CI: Run all Mojo tests
test-mojo:
    @just _run "just _test-mojo-inner"

# Run all Mojo tests under AddressSanitizer (memory errors, leaks, UAF)
test-mojo-asan:
    @just _run "just _test-mojo-sanitized-inner address"

# Run all Mojo tests under ThreadSanitizer (data races).
# Uses -j1 codegen (modular/modular#6413 workaround). Note: each compile is
# ~5min and holds 3-6GB; on memory-limited hosts, prefer `test-group-asan`
# on individual groups.
test-mojo-tsan:
    @just _run "just _test-mojo-sanitized-inner thread"

[private]
_test-mojo-sanitized-inner sanitizer:
    #!/usr/bin/env bash
    set -e
    REPO_ROOT="$(pwd)"
    SANITIZER="{{sanitizer}}"
    SAN_FLAGS="--sanitize $SANITIZER"
    # Match the JIT-crash workaround used in `_build-inner` for tsan
    JOBS=""
    if [ "$SANITIZER" = "thread" ]; then
        JOBS="-j1"
    fi
    echo "Running all Mojo tests with --sanitize $SANITIZER..."

    # Walk tree recursively via `find` — bash's `tests/**/*.mojo` without
    # `shopt -s globstar` only matches one level deep, silently skipping
    # ~85% of the test suite.
    mapfile -t test_files < <(
        find tests -name "test_*.mojo" \
            -not -path "*/__init__.mojo" \
            -not -path "tests/helpers/fixtures.mojo" \
            -not -path "tests/helpers/utils.mojo" \
            -not -path "tests/conftest.mojo" \
            | sort
    )

    test_count=${#test_files[@]}
    failed=0
    failed_tests=""
    for test_file in "${test_files[@]}"; do
        [ ! -f "$test_file" ] && continue

        BINARY=$(mktemp /tmp/mojo-san-XXXXXX)
        echo "Testing ($SANITIZER): $test_file"
        if pixi run mojo build $SAN_FLAGS --Werror $JOBS \
                -I "$REPO_ROOT" -I . -Xlinker -lm \
                "$test_file" -o "$BINARY" 2>&1 \
           && "$BINARY" 2>&1; then
            : # passed
        else
            failed=$((failed + 1))
            failed_tests="$failed_tests\n  - $test_file"
        fi
        rm -f "$BINARY"
    done

    if [ $test_count -eq 0 ]; then
        echo "❌ ERROR: No Mojo test files found in tests/"
        exit 1
    fi
    if [ $failed -gt 0 ]; then
        echo "❌ $failed tests failed under --sanitize $SANITIZER:"
        echo -e "$failed_tests"
        exit 1
    fi
    echo "✅ All $test_count tests passed under --sanitize $SANITIZER"

[private]
_test-mojo-inner:
    #!/usr/bin/env bash
    set -e
    REPO_ROOT="$(pwd)"
    echo "Running all Mojo tests..."

    # `find` walks the tree recursively; bash's `tests/**/*.mojo` glob
    # without `shopt -s globstar` only matches one level deep, which would
    # silently skip ~98% of the test suite. Use `find` for correctness.
    mapfile -t test_files < <(
        find tests -name "test_*.mojo" \
            -not -path "*/__init__.mojo" \
            -not -path "tests/helpers/fixtures.mojo" \
            -not -path "tests/helpers/utils.mojo" \
            -not -path "tests/conftest.mojo" \
            | sort
    )

    test_count=${#test_files[@]}
    if [ "$test_count" -eq 0 ]; then
        echo "❌ ERROR: No Mojo test files found in tests/"
        echo "   This usually means the directory is empty or test files were renamed."
        exit 1
    fi

    echo "Found $test_count test files"
    failed=0
    failed_tests=""
    for test_file in "${test_files[@]}"; do
        echo "Testing: $test_file"
        if ! pixi run mojo --Werror -I "$REPO_ROOT" -I . "$test_file"; then
            failed=$((failed + 1))
            failed_tests="$failed_tests\n  - $test_file"
        fi
    done

    if [ "$failed" -gt 0 ]; then
        echo "❌ $failed of $test_count tests failed:"
        echo -e "$failed_tests"
        exit 1
    fi
    echo "✅ All $test_count tests passed"

# ==============================================================================
# Utility
# ==============================================================================

# Show help
help:
    @echo "ML Odyssey Build System"
    @echo "======================="
    @echo ""
    @echo "Podman mode (default):  just <recipe>"
    @echo ""
    @echo "Training:  train [model] [precision] [epochs], infer [model] [checkpoint]"
    @echo "           list-models, infer-image [model] [checkpoint] [image_path]"
    @echo "           download-emnist"
    @echo "Build:     build [mode], build-debug, build-release, check, ci-build"
    @echo "Package:   package [mode], package-debug, package-release"
    @echo "Test:      test, test-python, test-group, test-group-asan, test-mojo"
    @echo "Jupyter:   jupyter, jupyter-notebook, jupyter-validate, jupyter-clear"
    @echo "Podman:    podman-up, podman-down, podman-build, podman-logs, podman-status"
    @echo "Dev:       bootstrap, shell, docs, docs-serve, pre-commit, validate"
    @echo "Utility:   help, status, clean, clean-all"
    @echo ""
    @echo "Examples:"
    @echo "  just download-emnist                # Download EMNIST dataset (run once)"
    @echo "  just train                          # Train LeNet-5 with defaults"
    @echo "  just train lenet fp16 20            # Train with FP16, 20 epochs (lenet/lenet5/lenet_emnist all work)"
    @echo "  just infer lenet lenet5_weights     # Evaluate on test set"
    @echo "  just jupyter                        # Launch Jupyter Lab"
    @echo "  just validate                       # Run validation locally"
    @echo ""
    @echo "For more: just --list"

# Show project status
status:
    @echo "================="
    @echo "ML Odyssey Status"
    @echo "================="
    @podman compose ps
    @ls -la build/ 2>/dev/null || echo "No build artifacts"
    @echo "Versions:"
    @python3 --version || echo "Python not found"
    @podman --version || echo "Podman not found"
    @just --version || echo "Just not found"
    @pixi --version
    @pixi --list

# Clean build artifacts
clean:
    @echo "Cleaning..."
    @rm -rf build/ dist/ htmlcov/ .pytest_cache/ .coverage
    @# Pruning __pycache__ via -prune avoids the "no such file" warnings that
    @# arise when find descends into directories it has just deleted.
    @find . -type d -name "__pycache__" -prune -exec rm -rf {} +
    @echo "Clean complete"

# Clean everything including Podman containers
clean-all: clean podman-clean

# Clean up stale git worktrees (dry-run by default, pass "apply" to remove)
clean-worktrees mode="dry-run":
    @if [ "{{mode}}" = "apply" ]; then \
        ./scripts/cleanup_stale_worktrees.sh; \
    elif [ "{{mode}}" = "apply-all" ]; then \
        ./scripts/cleanup_stale_worktrees.sh --include-unmerged; \
    else \
        ./scripts/cleanup_stale_worktrees.sh --dry-run; \
    fi

# ==============================================================================
# Dependency Audit
# ==============================================================================

# Run dependency audit locally
audit:
    @echo "Running dependency audit..."
    @echo ""
    @echo "=== Safety Scan ==="
    -@safety check --file requirements.txt
    -@safety check --file requirements-dev.txt
    @echo ""
    @echo "=== pip-audit Scan ==="
    -@pip-audit
    @echo ""
    @echo "=== License Check ==="
    @pip-licenses --format=markdown

# ==============================================================================
# Version Management
# ==============================================================================

# Check that version numbers are in sync across pyproject.toml, pixi.toml, mojo.toml, and VERSION
check-version-sync:
    @python3 scripts/check_version_sync.py

# Bump project version atomically across all version files and verify sync.
# Usage: just bump-version 0.2.0
bump-version new_version:
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ -z "{{new_version}}" ]]; then
        echo "Usage: just bump-version <new_version>"
        exit 1
    fi
    echo "Bumping version to {{new_version}} in all version files..."
    # VERSION (plain text)
    echo "{{new_version}}" > VERSION
    # pyproject.toml — update [project] version line
    sed -i 's/^\(version\s*=\s*\)"[^"]*"/\1"{{new_version}}"/' pyproject.toml
    # pixi.toml — update top-level version line
    sed -i 's/^\(version\s*=\s*\)"[^"]*"/\1"{{new_version}}"/' pixi.toml
    # mojo.toml — update version line
    sed -i 's/^\(version\s*=\s*\)"[^"]*"/\1"{{new_version}}"/' mojo.toml
    echo "All files updated. Verifying sync..."
    python3 scripts/check_version_sync.py
