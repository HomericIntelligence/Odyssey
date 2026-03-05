#!/usr/bin/env python3
"""Unit tests for fix-build-errors.py automation script.

Tests:
- Branch name sanitization
- Build validation
- Worktree management
- Metrics tracking
- Dependency validation
- Retry logic integration
"""

import importlib.util
import json
import pathlib
import shutil
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

# Add scripts directory to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "scripts"))

# Import module with dash in filename
spec = importlib.util.spec_from_file_location(
    "fix_build_errors", pathlib.Path(__file__).parent.parent.parent / "scripts" / "fix-build-errors.py"
)
fix_build_errors = importlib.util.module_from_spec(spec)
sys.modules["fix_build_errors"] = fix_build_errors  # Add to sys.modules for patching
spec.loader.exec_module(fix_build_errors)


class TestSanitizeBranchName(unittest.TestCase):
    """Test branch name sanitization."""

    def test_sanitize_simple_path(self):
        """Verify simple file path sanitization."""
        result = fix_build_errors.sanitize_branch_name("shared/core/tensor.mojo")
        self.assertEqual(result, "fix-shared-core-tensor")

    def test_sanitize_with_extension(self):
        """Verify extension is removed."""
        result = fix_build_errors.sanitize_branch_name("file.mojo")
        self.assertEqual(result, "fix-file")

    def test_sanitize_leading_dot_slash(self):
        """Verify leading ./ is removed."""
        result = fix_build_errors.sanitize_branch_name("./path/to/file.mojo")
        self.assertEqual(result, "fix-path-to-file")

    def test_sanitize_underscores(self):
        """Verify underscores replaced with hyphens."""
        result = fix_build_errors.sanitize_branch_name("test_utils.mojo")
        self.assertEqual(result, "fix-test-utils")

    def test_sanitize_numbers_start(self):
        """Verify numeric start gets 'fix-' prefix."""
        result = fix_build_errors.sanitize_branch_name("123_test.mojo")
        self.assertEqual(result, "fix-fix-123-test")  # Double fix- due to logic

    def test_sanitize_special_chars(self):
        """Verify special characters are removed."""
        result = fix_build_errors.sanitize_branch_name("path/to/@special#file.mojo")
        self.assertEqual(result, "fix-path-to-specialfile")

    def test_sanitize_multiple_slashes(self):
        """Verify multiple slashes handled correctly."""
        result = fix_build_errors.sanitize_branch_name("a/b/c/d/file.mojo")
        self.assertEqual(result, "fix-a-b-c-d-file")

    def test_sanitize_preserves_alphanumeric(self):
        """Verify alphanumeric characters preserved."""
        result = fix_build_errors.sanitize_branch_name("file123abc.mojo")
        self.assertEqual(result, "fix-file123abc")


class TestBuildValidation(unittest.TestCase):
    """Test build validation logic."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_path = pathlib.Path(self.temp_dir) / "test.log"

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("fix_build_errors.run")
    def test_build_ok_success_no_warnings(self, mock_run):
        """Verify successful build with no warnings."""
        mock_run.return_value = Mock(returncode=0, stderr="Build completed successfully", stdout="")

        result = fix_build_errors.build_ok(cwd=self.temp_dir, root="/root", file="test.mojo", log_path=self.log_path)

        self.assertTrue(result)
        mock_run.assert_called_once()

    @patch("fix_build_errors.run")
    def test_build_ok_has_warnings(self, mock_run):
        """Verify build with warnings returns False."""
        mock_run.return_value = Mock(
            returncode=0,
            stderr="warning: unused variable 'x'\nBuild completed",
            stdout="",
        )

        result = fix_build_errors.build_ok(cwd=self.temp_dir, root="/root", file="test.mojo", log_path=self.log_path)

        self.assertFalse(result)

    @patch("fix_build_errors.run")
    def test_build_ok_failure(self, mock_run):
        """Verify build failure returns False."""
        mock_run.return_value = Mock(returncode=1, stderr="error: syntax error at line 10", stdout="")

        result = fix_build_errors.build_ok(cwd=self.temp_dir, root="/root", file="test.mojo", log_path=self.log_path)

        self.assertFalse(result)

    @patch("fix_build_errors.run")
    def test_build_ok_case_insensitive_warning(self, mock_run):
        """Verify warning detection is case-insensitive."""
        mock_run.return_value = Mock(returncode=0, stderr="Warning: unused import", stdout="")

        result = fix_build_errors.build_ok(cwd=self.temp_dir, root="/root", file="test.mojo", log_path=self.log_path)

        self.assertFalse(result)

    @patch("fix_build_errors.run")
    def test_build_ok_logs_output(self, mock_run):
        """Verify build output is logged."""
        mock_run.return_value = Mock(returncode=0, stderr="Build output", stdout="")

        fix_build_errors.build_ok(cwd=self.temp_dir, root="/root", file="test.mojo", log_path=self.log_path)

        # Verify log file created and contains output
        self.assertTrue(self.log_path.exists())
        log_content = self.log_path.read_text()
        self.assertIn("Build output", log_content)


class TestWorktreeManagement(unittest.TestCase):
    """Test git worktree operations."""

    @patch("fix_build_errors.run")
    @patch("fix_build_errors.shutil.rmtree")
    @patch("pathlib.Path.exists")
    def test_create_worktree_cleanup_existing(self, mock_exists, mock_rmtree, mock_run):
        """Verify existing worktree is cleaned up before creation."""
        mock_exists.return_value = True
        mock_run.return_value = Mock(returncode=0, stderr="", stdout="")

        fix_build_errors.create_worktree("test-branch")

        # Verify cleanup was called
        mock_rmtree.assert_called()

    @patch("fix_build_errors.run")
    def test_create_worktree_fetches_main(self, mock_run):
        """Verify main branch is fetched before creating worktree."""
        mock_run.return_value = Mock(returncode=0, stderr="", stdout="")

        fix_build_errors.create_worktree("test-branch")

        # Verify fetch was called
        fetch_calls = [call for call in mock_run.call_args_list if "fetch" in str(call)]
        self.assertTrue(len(fetch_calls) > 0)

    @patch("fix_build_errors.run")
    def test_create_worktree_deletes_existing_branch(self, mock_run):
        """Verify existing branch is deleted before creating worktree."""
        mock_run.return_value = Mock(returncode=0, stderr="", stdout="")

        fix_build_errors.create_worktree("test-branch")

        # Verify branch delete was called
        delete_calls = [call for call in mock_run.call_args_list if "-D" in str(call)]
        self.assertTrue(len(delete_calls) > 0)

    @patch("fix_build_errors.run")
    def test_cleanup_worktree_removes_directory(self, mock_run):
        """Verify cleanup removes worktree directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = pathlib.Path(tmpdir) / "test-worktree"
            test_path.mkdir()

            fix_build_errors.cleanup_worktree(test_path)

            # Verify remove command was called
            mock_run.assert_called()
            # Directory should be removed (by rmtree in cleanup)
            self.assertFalse(test_path.exists())

    @patch("fix_build_errors.run")
    def test_cleanup_worktree_deletes_remote_branch(self, mock_run):
        """Verify cleanup deletes remote branch when specified."""
        mock_run.return_value = Mock(returncode=0, stderr="", stdout="")

        fix_build_errors.cleanup_worktree(pathlib.Path("/tmp/test"), branch="test-branch")  # nosec B108 - test code

        # Verify push --delete was called
        delete_calls = [call for call in mock_run.call_args_list if "--delete" in str(call)]
        self.assertTrue(len(delete_calls) > 0)


class TestMetricsTracking(unittest.TestCase):
    """Test metrics collection."""

    def setUp(self):
        """Reset metrics before each test."""
        fix_build_errors.metrics = {
            "start_time": None,
            "end_time": None,
            "files_processed": 0,
            "files_succeeded": 0,
            "files_failed": 0,
            "files_skipped": 0,
            "total_time_seconds": 0.0,
            "retries": {"claude_timeouts": 0, "git_operations": 0},
            "per_file": [],
        }

    def test_track_file_metrics_success(self):
        """Verify successful file metrics tracked."""
        fix_build_errors.track_file_metrics("test.mojo", success=True, elapsed_time=5.5)

        self.assertEqual(fix_build_errors.metrics["files_processed"], 1)
        self.assertEqual(fix_build_errors.metrics["files_succeeded"], 1)
        self.assertEqual(fix_build_errors.metrics["files_failed"], 0)
        self.assertEqual(len(fix_build_errors.metrics["per_file"]), 1)

    def test_track_file_metrics_failure(self):
        """Verify failed file metrics tracked."""
        fix_build_errors.track_file_metrics("test.mojo", success=False, elapsed_time=3.2)

        self.assertEqual(fix_build_errors.metrics["files_processed"], 1)
        self.assertEqual(fix_build_errors.metrics["files_succeeded"], 0)
        self.assertEqual(fix_build_errors.metrics["files_failed"], 1)

    def test_track_retry_claude_timeout(self):
        """Verify Claude timeout retry tracked."""
        fix_build_errors.track_retry("claude_timeouts")

        self.assertEqual(fix_build_errors.metrics["retries"]["claude_timeouts"], 1)

    def test_track_retry_git_operation(self):
        """Verify git operation retry tracked."""
        fix_build_errors.track_retry("git_operations")

        self.assertEqual(fix_build_errors.metrics["retries"]["git_operations"], 1)

    def test_track_multiple_files(self):
        """Verify multiple file metrics aggregated correctly."""
        fix_build_errors.track_file_metrics("file1.mojo", success=True, elapsed_time=2.0)
        fix_build_errors.track_file_metrics("file2.mojo", success=False, elapsed_time=3.0)
        fix_build_errors.track_file_metrics("file3.mojo", success=True, elapsed_time=4.0)

        self.assertEqual(fix_build_errors.metrics["files_processed"], 3)
        self.assertEqual(fix_build_errors.metrics["files_succeeded"], 2)
        self.assertEqual(fix_build_errors.metrics["files_failed"], 1)

    def test_save_metrics_calculates_derived(self):
        """Verify save_metrics calculates success rate and average time."""
        fix_build_errors.track_file_metrics("file1.mojo", success=True, elapsed_time=2.0)
        fix_build_errors.track_file_metrics("file2.mojo", success=False, elapsed_time=4.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            fix_build_errors.LOG_DIR = pathlib.Path(tmpdir)
            fix_build_errors.save_metrics()

            # Load saved metrics
            metrics_file = pathlib.Path(tmpdir) / "metrics.json"
            with open(metrics_file) as f:
                saved = json.load(f)

            # Verify derived metrics
            self.assertEqual(saved["success_rate"], 0.5)  # 1/2 succeeded
            self.assertEqual(saved["average_time_per_file"], 3.0)  # (2+4)/2


class TestDependencyValidation(unittest.TestCase):
    """Test dependency checking."""

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_check_dependencies_all_present(self, mock_run, mock_which):
        """Verify check passes when all dependencies available."""
        mock_which.return_value = "/usr/bin/cmd"
        mock_run.return_value = Mock(returncode=0)

        try:
            fix_build_errors.check_dependencies()
        except RuntimeError:
            self.fail("check_dependencies() raised RuntimeError unexpectedly")

    @patch("shutil.which")
    def test_check_dependencies_missing_command(self, mock_which):
        """Verify check fails when command missing."""

        def which_side_effect(cmd):
            return None if cmd == "mojo" else "/usr/bin/" + cmd

        mock_which.side_effect = which_side_effect

        with self.assertRaises(RuntimeError) as ctx:
            fix_build_errors.check_dependencies()

        self.assertIn("mojo", str(ctx.exception))
        self.assertIn("not found", str(ctx.exception).lower())

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_check_dependencies_gh_not_authenticated(self, mock_run, mock_which):
        """Verify check fails when gh not authenticated."""
        mock_which.return_value = "/usr/bin/cmd"
        mock_run.return_value = Mock(returncode=1)

        with self.assertRaises(RuntimeError) as ctx:
            fix_build_errors.check_dependencies()

        self.assertIn("not authenticated", str(ctx.exception))
        self.assertIn("gh auth login", str(ctx.exception))

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_health_check_all_ok(self, mock_run, mock_which):
        """Verify health check returns 0 when all OK."""
        mock_which.return_value = "/usr/bin/cmd"
        mock_run.return_value = Mock(returncode=0, stdout="version 1.0.0\n", stderr="Logged in as user\n")

        result = fix_build_errors.health_check()

        self.assertEqual(result, 0)

    @patch("shutil.which")
    def test_health_check_missing_dependency(self, mock_which):
        """Verify health check returns 1 when dependency missing."""

        def which_side_effect(cmd):
            return None if cmd == "claude" else "/usr/bin/" + cmd

        mock_which.side_effect = which_side_effect

        result = fix_build_errors.health_check()

        self.assertEqual(result, 1)


class TestRetryIntegration(unittest.TestCase):
    """Test retry logic integration with fix-build-errors."""

    @patch("fix_build_errors.run")
    def test_run_with_retry_success_no_retry(self, mock_run):
        """Verify successful command doesn't retry."""
        mock_run.return_value = Mock(returncode=0, stderr="", stdout="success")

        result = fix_build_errors.run_with_retry(["git", "status"])

        self.assertEqual(result.returncode, 0)
        mock_run.assert_called_once()

    @patch("fix_build_errors.run")
    @patch("time.sleep")
    def test_run_with_retry_network_error(self, mock_sleep, mock_run):
        """Verify network errors trigger retry."""
        mock_run.side_effect = [
            Mock(returncode=1, stderr="fatal: could not resolve hostname", stdout=""),  # Retry
            Mock(returncode=0, stderr="", stdout="success"),  # Success
        ]

        result = fix_build_errors.run_with_retry(["git", "fetch"])

        self.assertEqual(result.returncode, 0)
        self.assertEqual(mock_run.call_count, 2)

    @patch("fix_build_errors.run")
    def test_run_with_retry_non_network_error_no_retry(self, mock_run):
        """Verify non-network errors don't trigger retry."""
        mock_run.return_value = Mock(returncode=1, stderr="fatal: invalid reference", stdout="")

        result = fix_build_errors.run_with_retry(["git", "checkout", "invalid"])

        # Should not raise, just return error result
        self.assertEqual(result.returncode, 1)
        mock_run.assert_called_once()


class TestBuildCommand(unittest.TestCase):
    """Test build command construction."""

    def test_build_cmd_includes_all_flags(self):
        """Verify build command includes all required flags."""
        cmd = fix_build_errors.build_cmd(root="/root", file="test.mojo")

        self.assertIn("pixi", cmd)
        self.assertIn("run", cmd)
        self.assertIn("mojo", cmd)
        self.assertIn("build", cmd)
        self.assertIn("-g", cmd)
        self.assertIn("--no-optimization", cmd)
        self.assertIn("--validate-doc-strings", cmd)
        self.assertIn("-I", cmd)
        self.assertIn("/root", cmd)
        self.assertIn("test.mojo", cmd)

    def test_build_cmd_output_path(self):
        """Verify build command includes correct output path."""
        cmd = fix_build_errors.build_cmd(root="/root", file="path/to/test.mojo")

        self.assertIn("-o", cmd)
        self.assertIn("build/debug/test", cmd)


if __name__ == "__main__":
    unittest.main()
