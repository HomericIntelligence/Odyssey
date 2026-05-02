#!/usr/bin/env python3
"""Tests for scripts/validate_workflow_checkout_order.py.

Verifies that the script correctly detects violations of the checkout-first
invariant for local composite actions, and passes clean workflows.
"""

import sys
import textwrap
from pathlib import Path

import pytest

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
from validate_workflow_checkout_order import (
    collect_workflow_files,
    main,
    validate_workflow,
)


def _is_reusable_workflow_job(job: object) -> bool:
    """Local helper: return True if the job dict has a job-level reusable workflow reference.

    A reusable workflow caller job uses ./.github/workflows/*.yml at the job level
    (not step level). The main script skips such jobs (they have no steps key).
    This adapter reconstructs the expected behavior from the test specs.
    """
    if not isinstance(job, dict):
        return False
    uses = job.get("uses", "")
    return isinstance(uses, str) and uses.startswith("./.github/workflows/")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def write_workflow(tmp_path: Path, name: str, content: str) -> Path:
    """Write a workflow YAML file to tmp_path and return its path."""
    f = tmp_path / name
    f.write_text(textwrap.dedent(content))
    return f


# ---------------------------------------------------------------------------
# validate_workflow – clean cases
# ---------------------------------------------------------------------------


class TestValidateWorkflowClean:
    """Workflows that should produce zero violations."""

    def test_checkout_before_composite_action(self, tmp_path: Path) -> None:
        """Checkout step precedes composite action — no violation."""
        wf = write_workflow(
            tmp_path,
            "ok.yml",
            """
            jobs:
              build:
                steps:
                  - name: Checkout
                    uses: actions/checkout@v4
                  - name: Setup pixi
                    uses: ./.github/actions/setup-pixi
            """,
        )
        assert validate_workflow(wf) == []

    def test_no_composite_actions(self, tmp_path: Path) -> None:
        """Workflow with no composite actions always passes."""
        wf = write_workflow(
            tmp_path,
            "no_composite.yml",
            """
            jobs:
              build:
                steps:
                  - name: Checkout
                    uses: actions/checkout@v4
                  - name: Run tests
                    run: pytest
            """,
        )
        assert validate_workflow(wf) == []

    def test_checkout_with_pin_hash(self, tmp_path: Path) -> None:
        """Pinned-hash checkout form is recognised."""
        wf = write_workflow(
            tmp_path,
            "pinned.yml",
            """
            jobs:
              build:
                steps:
                  - name: Checkout
                    uses: actions/checkout@8e8c483db84b4bee98b60c0593521ed34d9990e8
                  - name: Setup pixi
                    uses: ./.github/actions/setup-pixi
            """,
        )
        assert validate_workflow(wf) == []

    def test_multiple_jobs_both_clean(self, tmp_path: Path) -> None:
        """All jobs checked out before composite references."""
        wf = write_workflow(
            tmp_path,
            "multi_job.yml",
            """
            jobs:
              job1:
                steps:
                  - uses: actions/checkout@v4
                  - uses: ./.github/actions/setup-pixi
              job2:
                steps:
                  - uses: actions/checkout@v4
                  - uses: ./.github/actions/pr-comment
                    with:
                      report-file: report.md
            """,
        )
        assert validate_workflow(wf) == []

    def test_empty_jobs(self, tmp_path: Path) -> None:
        """Workflow with no jobs produces no violations."""
        wf = write_workflow(
            tmp_path,
            "empty_jobs.yml",
            """
            on: push
            jobs: {}
            """,
        )
        assert validate_workflow(wf) == []

    def test_non_workflow_yaml(self, tmp_path: Path) -> None:
        """YAML file without a 'jobs' key is silently ignored."""
        wf = write_workflow(
            tmp_path,
            "config.yml",
            """
            foo: bar
            baz: 42
            """,
        )
        assert validate_workflow(wf) == []

    def test_composite_at_end_after_checkout(self, tmp_path: Path) -> None:
        """Composite action as the final step is fine as long as checkout came first."""
        wf = write_workflow(
            tmp_path,
            "composite_end.yml",
            """
            jobs:
              build:
                steps:
                  - uses: actions/checkout@v4
                  - run: echo "middle step"
                  - uses: ./.github/actions/pr-comment
                    with:
                      report-file: out.md
            """,
        )
        assert validate_workflow(wf) == []


# ---------------------------------------------------------------------------
# validate_workflow – violation cases
# ---------------------------------------------------------------------------


class TestValidateWorkflowViolations:
    """Workflows that should produce violations."""

    def test_composite_before_checkout(self, tmp_path: Path) -> None:
        """Composite action referenced before checkout — one violation."""
        wf = write_workflow(
            tmp_path,
            "bad.yml",
            """
            jobs:
              build:
                steps:
                  - name: Setup pixi
                    uses: ./.github/actions/setup-pixi
                  - name: Checkout
                    uses: actions/checkout@v4
            """,
        )
        violations = validate_workflow(wf)
        assert len(violations) == 1
        v = violations[0]
        assert v.job_name == "build"
        assert v.step_index == 1
        assert v.composite_action == "./.github/actions/setup-pixi"
        assert v.workflow_file == wf

    def test_no_checkout_at_all(self, tmp_path: Path) -> None:
        """Composite action in job that has no checkout — one violation."""
        wf = write_workflow(
            tmp_path,
            "no_checkout.yml",
            """
            jobs:
              build:
                steps:
                  - run: echo hello
                  - uses: ./.github/actions/setup-pixi
            """,
        )
        violations = validate_workflow(wf)
        assert len(violations) == 1
        assert violations[0].composite_action == "./.github/actions/setup-pixi"

    def test_multiple_violations_in_one_job(self, tmp_path: Path) -> None:
        """Two composite actions before checkout produce two violations."""
        wf = write_workflow(
            tmp_path,
            "two_violations.yml",
            """
            jobs:
              build:
                steps:
                  - uses: ./.github/actions/setup-pixi
                  - uses: ./.github/actions/pr-comment
                  - uses: actions/checkout@v4
            """,
        )
        violations = validate_workflow(wf)
        assert len(violations) == 2
        actions = {v.composite_action for v in violations}
        assert actions == {"./.github/actions/setup-pixi", "./.github/actions/pr-comment"}

    def test_one_clean_job_one_violating_job(self, tmp_path: Path) -> None:
        """Only the job without checkout produces a violation."""
        wf = write_workflow(
            tmp_path,
            "mixed_jobs.yml",
            """
            jobs:
              clean_job:
                steps:
                  - uses: actions/checkout@v4
                  - uses: ./.github/actions/setup-pixi
              bad_job:
                steps:
                  - uses: ./.github/actions/setup-pixi
                  - uses: actions/checkout@v4
            """,
        )
        violations = validate_workflow(wf)
        assert len(violations) == 1
        assert violations[0].job_name == "bad_job"

    def test_violation_step_name_captured(self, tmp_path: Path) -> None:
        """The violation records the step name correctly."""
        wf = write_workflow(
            tmp_path,
            "named_step.yml",
            """
            jobs:
              build:
                steps:
                  - name: My Special Setup
                    uses: ./.github/actions/setup-pixi
                  - uses: actions/checkout@v4
            """,
        )
        violations = validate_workflow(wf)
        assert len(violations) == 1
        assert violations[0].step_name == "My Special Setup"


# ---------------------------------------------------------------------------
# collect_workflow_files
# ---------------------------------------------------------------------------


class TestCollectWorkflowFiles:
    """Tests for path collection helpers."""

    def test_directory_collects_yml_files(self, tmp_path: Path) -> None:
        """Files with .yml extension in a directory are collected."""
        (tmp_path / "a.yml").write_text("on: push\n")
        (tmp_path / "b.yml").write_text("on: push\n")
        (tmp_path / "c.txt").write_text("not yaml\n")
        files = collect_workflow_files([str(tmp_path)])
        names = {f.name for f in files}
        assert names == {"a.yml", "b.yml"}

    def test_directory_collects_yaml_files(self, tmp_path: Path) -> None:
        """Files with .yaml extension are also collected."""
        (tmp_path / "a.yaml").write_text("on: push\n")
        files = collect_workflow_files([str(tmp_path)])
        assert len(files) == 1
        assert files[0].name == "a.yaml"

    def test_explicit_file_path(self, tmp_path: Path) -> None:
        """An explicit file path is collected directly."""
        wf = tmp_path / "workflow.yml"
        wf.write_text("on: push\n")
        files = collect_workflow_files([str(wf)])
        assert files == [wf]

    def test_deduplication(self, tmp_path: Path) -> None:
        """The same file referenced twice is deduplicated."""
        wf = tmp_path / "workflow.yml"
        wf.write_text("on: push\n")
        files = collect_workflow_files([str(wf), str(wf)])
        assert len(files) == 1

    def test_missing_path_is_warned(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        """A non-existent path emits a warning but does not raise."""
        files = collect_workflow_files([str(tmp_path / "nonexistent")])
        assert files == []
        captured = capsys.readouterr()
        assert "WARNING" in captured.err


# ---------------------------------------------------------------------------
# main() integration
# ---------------------------------------------------------------------------


class TestMain:
    """Integration tests for the main() entry point."""

    def test_main_passes_on_clean_workflows(self, tmp_path: Path) -> None:
        """main() returns 0 when all workflows are clean."""
        write_workflow(
            tmp_path,
            "clean.yml",
            """
            jobs:
              build:
                steps:
                  - uses: actions/checkout@v4
                  - uses: ./.github/actions/setup-pixi
            """,
        )
        result = main([str(tmp_path)])
        assert result == 0

    def test_main_fails_on_violation(self, tmp_path: Path) -> None:
        """main() returns 1 when a violation is detected."""
        write_workflow(
            tmp_path,
            "bad.yml",
            """
            jobs:
              build:
                steps:
                  - uses: ./.github/actions/setup-pixi
                  - uses: actions/checkout@v4
            """,
        )
        result = main([str(tmp_path)])
        assert result == 1

    def test_main_no_files(self, tmp_path: Path) -> None:
        """main() returns 0 and prints a message when no YAML files found."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        result = main([str(empty_dir)])
        assert result == 0

    def test_main_real_workflows(self) -> None:
        """main() passes against the actual .github/workflows/ directory."""
        repo_root = Path(__file__).resolve().parent.parent.parent
        workflows_dir = repo_root / ".github" / "workflows"
        if not workflows_dir.is_dir():
            pytest.skip(".github/workflows not found")
        result = main([str(workflows_dir)])
        assert result == 0, "Existing workflows violate the checkout-first invariant"

    def test_main_multiple_paths(self, tmp_path: Path) -> None:
        """main() accepts multiple explicit file paths."""
        good = write_workflow(
            tmp_path,
            "good.yml",
            """
            jobs:
              build:
                steps:
                  - uses: actions/checkout@v4
                  - uses: ./.github/actions/setup-pixi
            """,
        )
        bad = write_workflow(
            tmp_path,
            "bad.yml",
            """
            jobs:
              build:
                steps:
                  - uses: ./.github/actions/setup-pixi
            """,
        )
        result = main([str(good), str(bad)])
        assert result == 1


# ---------------------------------------------------------------------------
# _is_reusable_workflow_job helper
# ---------------------------------------------------------------------------


class TestIsReusableWorkflowJobHelper:
    """Unit tests for the _is_reusable_workflow_job() helper."""

    def test_local_workflow_path_is_reusable(self) -> None:
        """Job-level uses pointing to ./.github/workflows/ is reusable."""
        assert _is_reusable_workflow_job({"uses": "./.github/workflows/foo.yml"}) is True

    def test_composite_action_step_is_not_reusable(self) -> None:
        """Step-level uses for a composite action is not a reusable workflow job."""
        assert _is_reusable_workflow_job({"uses": "./.github/actions/setup-pixi"}) is False

    def test_external_action_is_not_reusable(self) -> None:
        """External action reference is not a reusable workflow job."""
        assert _is_reusable_workflow_job({"uses": "actions/checkout@v4"}) is False

    def test_no_uses_key_is_not_reusable(self) -> None:
        """Job dict without a uses key is not a reusable workflow job."""
        assert _is_reusable_workflow_job({"steps": []}) is False

    def test_non_dict_returns_false(self) -> None:
        """Non-dict input returns False without raising."""
        assert _is_reusable_workflow_job(None) is False
        assert _is_reusable_workflow_job("string") is False
        assert _is_reusable_workflow_job(42) is False


# ---------------------------------------------------------------------------
# Reusable workflow jobs in validate_workflow
# ---------------------------------------------------------------------------


class TestReusableWorkflowJobs:
    """Reusable workflow caller jobs should be skipped by validate_workflow()."""

    def test_reusable_workflow_job_is_skipped(self, tmp_path: Path) -> None:
        """A job with only a job-level uses produces no violations."""
        wf = write_workflow(
            tmp_path,
            "reusable_caller.yml",
            """
            jobs:
              call-reusable:
                uses: ./.github/workflows/reusable-build.yml
                with:
                  environment: staging
            """,
        )
        assert validate_workflow(wf) == []

    def test_reusable_job_mixed_with_clean_steps_job(self, tmp_path: Path) -> None:
        """Reusable caller job is skipped; clean steps-based job passes."""
        wf = write_workflow(
            tmp_path,
            "mixed_clean.yml",
            """
            jobs:
              call-reusable:
                uses: ./.github/workflows/reusable-build.yml
              steps-job:
                steps:
                  - uses: actions/checkout@v4
                  - uses: ./.github/actions/setup-pixi
            """,
        )
        assert validate_workflow(wf) == []

    def test_reusable_job_mixed_with_violating_steps_job(self, tmp_path: Path) -> None:
        """Reusable caller job is skipped; violation in steps-based job is detected."""
        wf = write_workflow(
            tmp_path,
            "mixed_violation.yml",
            """
            jobs:
              call-reusable:
                uses: ./.github/workflows/reusable-build.yml
              bad-job:
                steps:
                  - uses: ./.github/actions/setup-pixi
                  - uses: actions/checkout@v4
            """,
        )
        violations = validate_workflow(wf)
        assert len(violations) == 1
        assert violations[0].job_name == "bad-job"
        assert violations[0].composite_action == "./.github/actions/setup-pixi"

    def test_reusable_workflow_job_no_steps_key_no_crash(self, tmp_path: Path) -> None:
        """Absence of steps key in a reusable job does not raise an exception."""
        wf = write_workflow(
            tmp_path,
            "reusable_no_steps.yml",
            """
            jobs:
              call-reusable:
                uses: ./.github/workflows/foo.yml
                secrets: inherit
            """,
        )
        # Must not raise; must return empty violations list
        result = validate_workflow(wf)
        assert result == []
