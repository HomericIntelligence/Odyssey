# ML Odyssey Build System
# Unified build/test/lint interface for Docker and native execution

# Default recipe - show help
default: help

# Docker service name
docker_service := "projectodyssey-dev"

# Repository root
repo_root := justfile_directory()

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
MOJO_STRICT := "--validate-doc-strings"

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

# Run command in Docker or natively based on NATIVE env var
[private]
_run cmd:
	#!/usr/bin/env bash
	set -e
	if [[ "${NATIVE:-}" == "1" ]]; then
		eval "{{cmd}}"
	else
		# Check if container is running
		if ! docker compose ps -q {{docker_service}} | xargs docker inspect -f '{{"{{"}}.State.Running{{"}}"}}' | grep -q true; then
			echo "Container {{docker_service}} not running. Starting..."
			docker compose up -d {{docker_service}}
		fi
		docker compose exec -e USER_ID={{USER_ID}} -e GROUP_ID={{GROUP_ID}} -T {{docker_service}} bash -c "{{cmd}}"
	fi

# Ensure build directory exists
[private]
_ensure_build_dir mode:
    @mkdir -p build/{{mode}}

# ==============================================================================
# Docker Management
# ==============================================================================

# Start Docker development environment
docker-up:
    @docker compose up -d {{docker_service}}

# Stop Docker development environment
docker-down:
    @docker compose down

# Build Docker images
docker-build:
    @docker compose build \
        --build-arg USER_ID={{USER_ID}} \
        --build-arg GROUP_ID={{GROUP_ID}} \
        --build-arg USER_NAME=dev

# Rebuild Docker images (no cache)
docker-rebuild:
    @docker compose build --no-cache \
        --build-arg USER_ID={{USER_ID}} \
        --build-arg GROUP_ID={{GROUP_ID}} \
        --build-arg USER_NAME=dev

# View Docker logs
docker-logs:
    @docker compose logs -f {{docker_service}}

# Clean Docker resources
docker-clean:
    @docker compose down -v --rmi local

# Show Docker status
docker-status:
    @docker compose ps

native prefix:
    @NATIVE=1 just {{prefix}}

# ==============================================================================
# Docker Registry (GHCR)
# ==============================================================================

# Default registry
REGISTRY := "ghcr.io"
REPO_NAME := "homericintelligence/projectodyssey"

# Build CI-optimized Docker image
docker-build-ci target="runtime":
    @docker build -f Dockerfile.ci --target {{target}} \
        -t {{REGISTRY}}/{{REPO_NAME}}:{{target}} \
        -t {{REGISTRY}}/{{REPO_NAME}}:{{target}}-$(git rev-parse --short HEAD) \
        .

# Build all CI image targets
docker-build-ci-all:
    @just docker-build-ci runtime
    @just docker-build-ci ci
    @just docker-build-ci production

# Push Docker image to GHCR (requires `docker login ghcr.io`)
docker-push target="runtime":
    @docker push {{REGISTRY}}/{{REPO_NAME}}:{{target}}
    @docker push {{REGISTRY}}/{{REPO_NAME}}:{{target}}-$(git rev-parse --short HEAD)

# Push all images to GHCR
docker-push-all:
    @just docker-push runtime
    @just docker-push ci
    @just docker-push production

# Build and push Docker image
docker-release target="runtime":
    @just docker-build-ci {{target}}
    @just docker-push {{target}}

# Test Docker image locally
docker-test-image target="runtime":
    @echo "Testing {{target}} image..."
    @docker run --rm {{REGISTRY}}/{{REPO_NAME}}:{{target}} pixi run mojo --version
    @echo "✅ Image {{target}} is working"

# Run tests in Docker image
docker-run-tests target="runtime":
    @docker run --rm {{REGISTRY}}/{{REPO_NAME}}:{{target}} \
        pixi run mojo test -I . tests/

# Interactive shell in Docker image
docker-run-shell target="runtime":
    @docker run -it --rm {{REGISTRY}}/{{REPO_NAME}}:{{target}} bash

# ==============================================================================
# CI Docker Build (matches GitHub Actions)
# ==============================================================================

# Build Docker image exactly as CI does
ci-docker-build:
    @docker buildx build \
        --file Dockerfile.ci \
        --target runtime \
        --platform linux/amd64 \
        --cache-from type=local,src=/tmp/.buildx-cache \
        --cache-to type=local,dest=/tmp/.buildx-cache-new,mode=max \
        -t {{REGISTRY}}/{{REPO_NAME}}:local \
        .
    @rm -rf /tmp/.buildx-cache
    @mv /tmp/.buildx-cache-new /tmp/.buildx-cache 2>/dev/null || true

# Validate Docker build without pushing
ci-docker-validate:
    @echo "Validating Dockerfile.ci..."
    @docker build -f Dockerfile.ci --target runtime . >/dev/null
    @echo "✅ Dockerfile.ci is valid"

# ==============================================================================
# Build Recipes (mojo build - compile/validate Mojo files)
# ==============================================================================

# Build/compile Mojo files with mode-specific flags
build mode="debug": (_ensure_build_dir mode)
    #!/usr/bin/env bash
    set -euo pipefail

    # ------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------

    MODE="${1:-debug}"          # debug | release | test | ci
    REPO_ROOT="$(pwd)"
    STRICT="--validate-doc-strings"

    # CI mode should continue despite linker errors (Mojo limitation: cannot pass -lm flag)
    FAIL_ON_ERROR=0

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
            # Don't fail on linker errors - Mojo doesn't support -lm flag yet
            FAIL_ON_ERROR=0
            ;;
        *)
            echo "❌ Unknown mode: $MODE"
            echo "Valid modes: debug | release | test | ci"
            exit 1
            ;;
    esac

    BUILD_DIR="build/$MODE"
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
        -not -path "./.claude/*" \
        -not -path "./tests/*" \
        -not -path "./shared/*" \
        -not -path "./benchmarks/*" \
        -not -name "test_*.mojo" \
        -not -name "model.mojo" \
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

# CI: Build Mojo files with strict analysis (mojo build)
ci-build:


# ==============================================================================
# Package Recipes (mojo package - create .mojopkg libraries)
# ==============================================================================

# Package shared library with mode-specific flags
package mode="debug": (_ensure_build_dir mode)
    #!/usr/bin/env bash
    set -e
    REPO_ROOT="$(pwd)"
    STRICT="--validate-doc-strings"

    case "{{mode}}" in
        debug)   FLAGS="$STRICT" ;;
        release) FLAGS="$STRICT" ;;
        test)    FLAGS="$STRICT" ;;
        *)       FLAGS="$STRICT" ;;
    esac

    echo "Packaging shared library in {{mode}} mode..."
    echo "Flags: $FLAGS"
    pixi run mojo package $FLAGS -I "$REPO_ROOT" shared -o build/{{mode}}/ProjectOdyssey-shared.mojopkg

# Package debug version
package-debug: (package "debug")

# Package release version
package-release: (package "release")

# ==============================================================================
# Model Training and Inference
# ==============================================================================

# Train a model (default: LeNet-5 on EMNIST)
train model="lenet-emnist" precision="fp32" epochs="10" batch_size="32" lr="0.001":
    @echo "Training {{model}} with precision={{precision}}, epochs={{epochs}}"
    @NATIVE=1 pixi run mojo run -I . examples/{{model}}/run_train.mojo \
        --epochs {{epochs}} \
        --batch-size {{batch_size}} \
        --lr {{lr}} \
        --precision {{precision}}

# Run inference on test set
infer model="lenet-emnist" checkpoint="lenet5_weights":
    @echo "Running inference for {{model}} with checkpoint={{checkpoint}}"
    @NATIVE=1 pixi run mojo run -I . examples/{{model}}/run_infer.mojo \
        --checkpoint {{checkpoint}} \
        --test-set

# Run inference on single image
infer-image checkpoint image_path model="lenet-emnist":
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
    @pixi run jupyter lab --no-browser --ip=127.0.0.1

# Launch Jupyter Notebook (classic)
jupyter-notebook:
    @pixi run jupyter notebook --no-browser --ip=127.0.0.1

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
            if pixi run jupyter nbconvert --execute --to notebook --inplace "$notebook" 2>&1; then
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
            pixi run jupyter nbconvert --clear-output --inplace "$notebook"
        fi
    done
    echo "Cleared outputs from $NOTEBOOK_COUNT notebooks"

# ==============================================================================
# Development
# ==============================================================================

# Open development shell
shell:
    @docker compose exec -it -e USER_ID={{USER_ID}} -e GROUP_ID={{GROUP_ID}} {{docker_service}} bash

# Serve documentation
docs-serve:
    @mkdocs serve || echo "Install mkdocs for documentation"

# Build documentation
docs:
    @mkdocs build || echo "Install mkdocs for documentation"

# ==============================================================================
# CI/CD
# ==============================================================================

# Run pre-commit hooks
pre-commit:
    @pre-commit run

pre-commit-all:
    @pre-commit run --all-files

# CI: Full validation (build + package + test)
validate:
    @echo "Validating configuration..."
    @test -f pixi.toml && echo "✅ pixi.toml exists"
    @test -f docker-compose.yml && echo "✅ docker-compose.yml exists"
    @echo "Running full CI validation..."
    @just build
    @just package
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
    #!/usr/bin/env bash
    set -e
    REPO_ROOT="$(pwd)"
    TEST_PATH="{{path}}"
    test_count=0
    passed_count=0
    failed_count=0
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

    # Run each test file
    for test_file in $test_files; do
        if [ -f "$test_file" ]; then
            echo ""
            echo "Running: $test_file"
            test_count=$((test_count + 1))

            attempt=0
            max_attempts=3
            delay=1
            test_passed=0
            while [ $attempt -lt $max_attempts ]; do
                attempt=$((attempt + 1))
                if pixi run mojo -I "$REPO_ROOT" -I . "$test_file"; then
                    test_passed=1
                    break
                fi
                if [ $attempt -lt $max_attempts ]; then
                    echo "⚠️  FAILED (attempt $attempt), retrying in ${delay}s: $test_file"
                    sleep $delay
                    delay=$((delay * 2))
                fi
            done
            if [ $test_passed -eq 1 ]; then
                if [ $attempt -eq 1 ]; then
                    echo "✅ PASSED: $test_file"
                else
                    echo "✅ PASSED on attempt $attempt: $test_file"
                fi
                passed_count=$((passed_count + 1))
            else
                echo "❌ FAILED: $test_file"
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

    if [ $failed_count -gt 0 ]; then
        echo ""
        echo "Failed tests:"
        echo -e "$failed_tests"
        exit 1
    fi

# CI: Run all Mojo tests
test-mojo:
    #!/usr/bin/env bash
    set -e
    REPO_ROOT="$(pwd)"
    echo "Running all Mojo tests..."

    failed=0
    for test_file in tests/**/*.mojo; do
        if [ -f "$test_file" ]; then
            echo "Testing: $test_file"
            if ! pixi run mojo -I "$REPO_ROOT" -I . "$test_file"; then
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
    @echo "Docker mode (default):  just <recipe>"
    @echo "Native mode:            just native <recipe>"
    @echo ""
    @echo "Training:  train [model] [precision] [epochs], infer [model] [checkpoint]"
    @echo "           list-models, infer-image [model] [checkpoint] [image_path]"
    @echo "Build:     build [mode], build-debug, build-release"
    @echo "Package:   package [mode], package-debug, package-release"
    @echo "Test:      test, test-python, test-group, test-mojo"
    @echo "Jupyter:   jupyter, jupyter-notebook, jupyter-validate, jupyter-clear"
    @echo "Docker:    docker-up, docker-down, docker-logs"
    @echo "Dev:       dev, shell, docs, docs-serve, pre-commit, validate"
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
    @docker compose ps
    @ls -la build/ 2>/dev/null || echo "No build artifacts"
    @echo "Versions:"
    @python3 --version || echo "Python not found"
    @docker --version || echo "Docker not found"
    @just --version || echo "Just not found"
    @pixi --version
    @pixi --list

# Clean build artifacts
clean:
    @echo "Cleaning..."
    @rm -rf build/ dist/ htmlcov/ .pytest_cache/ .coverage
    @find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    @echo "Clean complete"

# Clean everything including Docker
clean-all: clean docker-clean

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
