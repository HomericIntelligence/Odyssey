#!/usr/bin/env python3
"""Smoke tests for scripts/test-with-retry.sh.

Validates the JIT crash retry logic by simulating different failure modes
using mock test scripts instead of real Mojo tests:
  1. Passing test → exit 0, no retry
  2. Real test failure (no "execution crashed") → exit 1, no retry
  3. Persistent JIT crash → exit 2, retried once then fails
  4. Transient JIT crash (crash then pass) → exit 0, retried once then passes
"""

import os
import subprocess
import stat
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
RETRY_SCRIPT = REPO_ROOT / "scripts" / "test-with-retry.sh"


@pytest.fixture(scope="module")
def retry_script_exists() -> Path:
    """Ensure the retry script exists and is executable."""
    assert RETRY_SCRIPT.exists(), f"test-with-retry.sh not found at {RETRY_SCRIPT}"
    assert os.access(RETRY_SCRIPT, os.X_OK), "test-with-retry.sh is not executable"
    return RETRY_SCRIPT


def _make_mock_mojo(tmp_path: Path, name: str, script_content: str) -> Path:
    """Create a mock 'pixi' wrapper that simulates mojo test behavior.

    The retry script invokes `pixi run mojo --Werror -I ... -I . <file>`.
    We override PATH so that our mock `pixi` script runs instead.
    """
    mock_pixi = tmp_path / "pixi"
    mock_pixi.write_text(textwrap.dedent(script_content))
    mock_pixi.chmod(mock_pixi.stat().st_mode | stat.S_IEXEC)
    return mock_pixi


def _run_retry(
    tmp_path: Path,
    mock_script_content: str,
    test_file: str = "fake_test.mojo",
    max_retries: int = 1,
) -> subprocess.CompletedProcess:
    """Run test-with-retry.sh with a mock pixi on PATH."""
    _make_mock_mojo(tmp_path, "pixi", mock_script_content)

    env = os.environ.copy()
    # Prepend tmp_path so our mock pixi is found first
    env["PATH"] = str(tmp_path) + ":" + env.get("PATH", "")
    env["TEST_WITH_RETRY_MAX"] = str(max_retries)

    result = subprocess.run(
        ["bash", str(RETRY_SCRIPT), str(REPO_ROOT), test_file],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    return result


class TestRetryScriptExists:
    """Verify the retry script is present and well-formed."""

    def test_script_exists(self, retry_script_exists: Path) -> None:
        """test-with-retry.sh must exist at scripts/test-with-retry.sh."""
        assert retry_script_exists.exists()

    def test_script_has_shebang(self, retry_script_exists: Path) -> None:
        """Script must have a proper bash shebang line."""
        first_line = retry_script_exists.read_text().split("\n")[0]
        assert first_line.startswith("#!/"), "Missing shebang line"
        assert "bash" in first_line, "Shebang should reference bash"


class TestRetryBehavior:
    """Validate retry logic for different failure modes."""

    def test_passing_test_exits_zero(self, tmp_path: Path, retry_script_exists: Path) -> None:
        """A passing test should exit 0 with no retry."""
        mock = """\
        #!/usr/bin/env bash
        echo "test_example ... PASS"
        exit 0
        """
        result = _run_retry(tmp_path, mock)
        assert result.returncode == 0
        assert "PASSED" in result.stdout
        assert "retry" not in result.stdout.lower() or "PASSED on retry" not in result.stdout

    def test_real_failure_exits_one_no_retry(self, tmp_path: Path, retry_script_exists: Path) -> None:
        """A real test failure (no 'execution crashed') should exit 1, never retry."""
        mock = """\
        #!/usr/bin/env bash
        echo "test_example ... FAIL: assertion failed"
        exit 1
        """
        result = _run_retry(tmp_path, mock)
        assert result.returncode == 1
        assert "FAILED" in result.stdout
        # Should NOT contain retry messaging
        assert "retrying" not in result.stdout.lower()

    def test_persistent_jit_crash_exits_two(self, tmp_path: Path, retry_script_exists: Path) -> None:
        """Persistent JIT crash (both attempts crash) should exit 2."""
        mock = """\
        #!/usr/bin/env bash
        echo "execution crashed"
        exit 1
        """
        result = _run_retry(tmp_path, mock)
        assert result.returncode == 2
        assert "JIT crash" in result.stdout.lower() or "jit crash" in result.stdout.lower()

    def test_transient_jit_crash_passes_on_retry(self, tmp_path: Path, retry_script_exists: Path) -> None:
        """Transient JIT crash (crash first, pass on retry) should exit 0."""
        # Use a state file to track attempts
        state_file = tmp_path / "attempt_counter"
        state_file.write_text("0")
        mock = f"""\
        #!/usr/bin/env bash
        STATE="{state_file}"
        COUNT=$(cat "$STATE")
        COUNT=$((COUNT + 1))
        echo "$COUNT" > "$STATE"
        if [ "$COUNT" -eq 1 ]; then
            echo "execution crashed"
            exit 1
        else
            echo "test_example ... PASS"
            exit 0
        fi
        """
        result = _run_retry(tmp_path, mock)
        assert result.returncode == 0
        assert "retry" in result.stdout.lower()

    def test_jit_crash_then_real_failure(self, tmp_path: Path, retry_script_exists: Path) -> None:
        """JIT crash on first attempt, real failure on retry → exit 1."""
        state_file = tmp_path / "attempt_counter"
        state_file.write_text("0")
        mock = f"""\
        #!/usr/bin/env bash
        STATE="{state_file}"
        COUNT=$(cat "$STATE")
        COUNT=$((COUNT + 1))
        echo "$COUNT" > "$STATE"
        if [ "$COUNT" -eq 1 ]; then
            echo "execution crashed"
            exit 1
        else
            echo "test_example ... FAIL: assertion failed at line 42"
            exit 1
        fi
        """
        result = _run_retry(tmp_path, mock)
        assert result.returncode == 1
        assert "FAILED" in result.stdout
