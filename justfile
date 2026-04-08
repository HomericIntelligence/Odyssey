# ML Odyssey Build System
# Unified build/test/lint interface for Podman containers and native execution

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

# Run command in Podman compose container or natively based on NATIVE env var.
# If BUILD_ROOT is outside the repo, mounts it into the container at /ext-build
# and rewrites paths in cmd accordingly.
[private]
_run cmd:
	#!/usr/bin/env bash
	set -e
	if [[ "${NATIVE:-}" == "1" ]]; then
		eval "{{cmd}}"
	elif command -v podman &>/dev/null && \
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
			podman compose exec -e USER_ID={{USER_ID}} -e GROUP_ID={{GROUP_ID}} -T {{podman_service}} bash -c "{{cmd}}"
		fi
	else
		echo "Error: Podman compose container '{{podman_service}}' is not running."
		echo "  Start Podman:     just podman-up"
		echo "  Or run natively:  NATIVE=1 just <recipe>"
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
podman-up:
    @podman compose up -d {{podman_service}}

# Stop Podman development environment
podman-down:
    @podman compose down

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

native prefix:
    @NATIVE=1 just {{prefix}}

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

    MODE="{{mode}}"             # debug | release | test | ci
    REPO_ROOT="$(pwd)"
    STRICT="--Werror"

    FAIL_ON_ERROR=1

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
        *)
            echo "❌ Unknown mode: $MODE"
            echo "Valid modes: debug | release | test | ci"
            exit 1
            ;;
    esac

    BUILD_DIR="{{BUILD_ROOT}}/$MODE"
    mkdir -p "$BUILD_DIR"

    echo "🔨 Building Mojo files"
    echo "Mode:  $MODE"
    echo "Flags: $FLAGS"
    echo "Out:   $BUILD_DIR"
    echo

    # ------------------------------------------------------------
    # Build
    # ------------------------------------------------------------

    FAILED=0

    find . -name "*.mojo" \
        -not -path "./.pixi/*" \
        -not -path "./worktrees/*" \
        -not -path "./.worktrees/*" \
        -not -path "./.claude/*" \
        -not -path "./tests/*" \
        -not -path "./shared/*" \
        -not -path "./examples/*" \
        -not -path "./benchmarks/*" \
        -not -path "./.templates/*" \
        -not -path "./papers/_template/*" \
        -not -path "./notes/*" \
        -not -name "test_*.mojo" \
        -not -name "model.mojo" \
        -not -name "__init__.mojo" \
        -not -name "repro_*.mojo" \
        | while read -r file; do
            out="$BUILD_DIR/$(basename "$file" .mojo)"
            echo "→ Building: $file"

            if ! pixi run mojo build $FLAGS -I "$REPO_ROOT" "$file" -o "$out" 2>&1; then
                echo "❌ Failed: $file"
                FAILED=1
                if [[ "$FAIL_ON_ERROR" -eq 1 ]]; then
                    exit 1
                fi
            fi
        done

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    if [[ "$FAILED" -eq 1 ]]; then
        echo
        echo "⚠️  Build completed with errors"
        echo "Outputs in $BUILD_DIR"
        [[ "$FAIL_ON_ERROR" -eq 1 ]] && exit 1
    else
        echo
        echo "✅ Build successful"
        echo "Outputs in $BUILD_DIR"
    fi

# Build debug version
build-debug: (build "debug")

# Build release version
build-release: (build "release")

# Build all modes
build-all:
    @just build debug
    @just build release
    @just build test

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

# Train a model (default: LeNet-5 on EMNIST)
train model="lenet_emnist" precision="fp32" epochs="10" batch_size="32" lr="0.001":
    @echo "Training {{model}} with precision={{precision}}, epochs={{epochs}}"
    @NATIVE=1 pixi run mojo run -I . examples/{{model}}/run_train.mojo \
        --epochs {{epochs}} \
        --batch-size {{batch_size}} \
        --lr {{lr}} \
        --precision {{precision}}

# Run inference on test set
infer model="lenet_emnist" checkpoint="lenet5_weights":
    @echo "Running inference for {{model}} with checkpoint={{checkpoint}}"
    @NATIVE=1 pixi run mojo run -I . examples/{{model}}/run_infer.mojo \
        --checkpoint {{checkpoint}} \
        --test-set

# Run inference on single image
infer-image checkpoint image_path model="lenet_emnist":
    @echo "Running inference on {{image_path}}"
    @NATIVE=1 pixi run mojo run -I . examples/{{model}}/run_infer.mojo \
        --checkpoint {{checkpoint}} \
        --image {{image_path}}

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

    # Step 4: GLIBC compatibility check
    echo "==> Checking GLIBC compatibility..."
    just check-glibc 2>/dev/null || true
    echo ""

    echo "Bootstrap complete! Next steps:"
    echo "  just help          Show available commands"
    echo "  just podman-up     Start Podman dev environment"
    echo "  just test          Run all tests"
    echo "  just build         Build the project"

# Open development shell
shell:
    @podman compose exec -it -e USER_ID={{USER_ID}} -e GROUP_ID={{GROUP_ID}} {{podman_service}} bash

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
    violations=$(grep -rn "\.__matmul__(" . --include="*.mojo" --include="*.🔥" --exclude-dir=".pixi" --exclude-dir=".git" | grep -v "fn __matmul__(" | grep -v "# __matmul__" | grep -v "__matmul__.*deprecated" || true)
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
    jit_crash_count=0
    failed_tests=""

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

    # Run each test file (with JIT crash retry — see #5104)
    for test_file in $test_files; do
        if [ -f "$test_file" ]; then
            echo ""
            echo "Running: $test_file"
            test_count=$((test_count + 1))

            retry_result=0
            bash "$REPO_ROOT/scripts/test-with-retry.sh" "$REPO_ROOT" "$test_file" || retry_result=$?
            if [ $retry_result -eq 0 ]; then
                passed_count=$((passed_count + 1))
            elif [ $retry_result -eq 2 ]; then
                failed_count=$((failed_count + 1))
                jit_crash_count=$((jit_crash_count + 1))
                failed_tests="$failed_tests\n  - $test_file (JIT crash)"
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
    if [ $jit_crash_count -gt 0 ]; then
        echo "JIT crash retries: $jit_crash_count (persistent crashes after retry)"
    fi

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
            if output=$(pixi run mojo build {{MOJO_ASAN}} {{MOJO_STRICT}} -I "$REPO_ROOT" -I . "$test_file" -o "$BINARY" 2>&1) && output2=$("$BINARY" 2>&1); then
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

[private]
_test-mojo-inner:
    #!/usr/bin/env bash
    set -e
    REPO_ROOT="$(pwd)"
    echo "Running all Mojo tests..."

    # Count test files
    test_count=0
    for test_file in tests/**/*.mojo; do
        [[ "$(basename "$test_file")" == "__init__.mojo" ]] && continue
        [[ "$(basename "$test_file")" == "conftest.mojo" ]] && continue
        [[ "$test_file" == "tests/helpers/fixtures.mojo" ]] && continue
        [[ "$test_file" == "tests/helpers/utils.mojo" ]] && continue
        [ -f "$test_file" ] && test_count=$((test_count + 1))
    done

    if [ $test_count -eq 0 ]; then
        echo "❌ ERROR: No Mojo test files found in tests/"
        echo "   This usually means the directory is empty or test files were renamed."
        exit 1
    fi

    failed=0
    for test_file in tests/**/*.mojo; do
        # Skip non-test library files (no main function)
        if [[ "$(basename "$test_file")" == "__init__.mojo" ]] || \
           [[ "$(basename "$test_file")" == "conftest.mojo" ]] || \
           [[ "$test_file" == "tests/helpers/fixtures.mojo" ]] || \
           [[ "$test_file" == "tests/helpers/utils.mojo" ]]; then
            continue
        fi
        if [ -f "$test_file" ]; then
            echo "Testing: $test_file"
            retry_result=0
            bash "$REPO_ROOT/scripts/test-with-retry.sh" "$REPO_ROOT" "$test_file" || retry_result=$?
            if [ $retry_result -ne 0 ]; then
                failed=1
            fi
        fi
    done

    if [ $failed -eq 1 ]; then
        echo "❌ Some tests failed"
        exit 1
    fi
    echo "✅ All tests passed"

# ==============================================================================
# Utility
# ==============================================================================

# Show help
help:
    @echo "ML Odyssey Build System"
    @echo "======================="
    @echo ""
    @echo "Podman mode (default):  just <recipe>"
    @echo "Native mode:            just native <recipe>"
    @echo ""
    @echo "Training:  train [model] [precision] [epochs], infer [model] [checkpoint]"
    @echo "           list-models, infer-image [model] [checkpoint] [image_path]"
    @echo "Build:     build [mode], build-debug, build-release, check, ci-build"
    @echo "Package:   package [mode], package-debug, package-release"
    @echo "Test:      test, test-python, test-group, test-group-asan, test-mojo"
    @echo "Jupyter:   jupyter, jupyter-notebook, jupyter-validate, jupyter-clear"
    @echo "Podman:    podman-up, podman-down, podman-build, podman-logs, podman-status"
    @echo "Dev:       bootstrap, shell, docs, docs-serve, pre-commit, validate"
    @echo "Utility:   help, status, clean, clean-all"
    @echo ""
    @echo "Examples:"
    @echo "  just train                          # Train LeNet-5 with defaults"
    @echo "  just train lenet5 fp16 20           # Train with FP16, 20 epochs"
    @echo "  just infer lenet5 ./weights         # Evaluate on test set"
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
    @find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
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
    @pip install safety pip-audit pip-licenses 2>/dev/null || true
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
