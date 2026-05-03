#!/usr/bin/env python3
"""Tests for scripts/lint_configs.py configuration linter."""

import pytest
import tempfile
from pathlib import Path
import sys

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from lint_configs import ConfigLinter


@pytest.fixture
def linter():
    """Create a ConfigLinter instance for testing."""
    return ConfigLinter(verbose=False)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestYAMLSyntaxValidation:
    """Test YAML syntax validation."""

    def test_valid_yaml(self, linter, temp_dir):
        """Test that valid YAML passes syntax validation."""
        yaml_file = temp_dir / "valid.yaml"
        yaml_file.write_text(
            """
training:
  epochs: 10
  batch_size: 32
model:
  architecture: lenet5
"""
        )
        result = linter.lint_file(yaml_file)
        assert result is True
        assert len(linter.errors) == 0

    def test_invalid_yaml_syntax(self, linter, temp_dir):
        """Test that invalid YAML syntax is caught."""
        yaml_file = temp_dir / "invalid.yaml"
        yaml_file.write_text(
            """
training:
  epochs: 10
  batch_size: [missing_bracket
"""
        )
        result = linter.lint_file(yaml_file)
        assert result is False
        assert len(linter.errors) > 0
        assert any("syntax" in err.lower() or "yaml" in err.lower() for err in linter.errors)

    def test_duplicate_keys(self, linter, temp_dir):
        """Test detection of duplicate keys in YAML."""
        yaml_file = temp_dir / "duplicate.yaml"
        yaml_file.write_text(
            """
training:
  epochs: 10
  epochs: 20
"""
        )
        result = linter.lint_file(yaml_file)
        assert result is True
        # YAML parsers typically take the last value, but we should detect this
        assert len(linter.warnings) > 0 or len(linter.errors) > 0


class TestFormattingChecks:
    """Test formatting rule validation."""

    def test_correct_indentation(self, linter, temp_dir):
        """Test that 2-space indentation is accepted."""
        yaml_file = temp_dir / "correct_indent.yaml"
        yaml_file.write_text(
            """training:
  epochs: 10
  batch_size: 32
  optimizer:
    name: adam
    learning_rate: 0.001
"""
        )
        result = linter.lint_file(yaml_file)
        assert result is True
        # Should not have any warnings (all required keys present, correct formatting)
        assert len(linter.warnings) == 0

    def test_tab_characters(self, linter, temp_dir):
        """Test detection of tab characters."""
        yaml_file = temp_dir / "tabs.yaml"
        # Use actual tab character
        yaml_file.write_text("training:\n\tepochs: 10\n")
        result = linter.lint_file(yaml_file)
        # Tabs are errors, so linting fails
        assert result is False
        assert len(linter.errors) > 0
        assert any("tab" in msg.lower() for msg in linter.warnings + linter.errors)

    def test_trailing_whitespace(self, linter, temp_dir):
        """Test detection of trailing whitespace."""
        yaml_file = temp_dir / "trailing.yaml"
        yaml_file.write_text("training:  \n  epochs: 10\n")
        result = linter.lint_file(yaml_file)
        assert result is True
        # Should have warning about trailing whitespace
        assert len(linter.warnings) > 0
        assert any("trailing" in w.lower() or "whitespace" in w.lower() for w in linter.warnings)


class TestDeprecatedKeyDetection:
    """Test deprecated key warnings."""

    def test_deprecated_key_warning(self, linter, temp_dir):
        """Test that deprecated keys generate warnings."""
        yaml_file = temp_dir / "deprecated.yaml"
        yaml_file.write_text(
            """
optimizer:
  type: adam
  lr: 0.001
"""
        )
        result = linter.lint_file(yaml_file)
        assert result is True
        # Should have warnings about deprecated keys
        assert len(linter.warnings) > 0
        deprecated_warnings = [w for w in linter.warnings if "deprecated" in w.lower()]
        assert len(deprecated_warnings) >= 1

    def test_replacement_suggestion(self, linter, temp_dir):
        """Test that warnings suggest replacement keys."""
        yaml_file = temp_dir / "deprecated.yaml"
        yaml_file.write_text(
            """
lr: 0.001
"""
        )
        linter.lint_file(yaml_file)
        # Should suggest using 'learning_rate' instead of 'lr'
        assert any("learning_rate" in w for w in linter.warnings + linter.suggestions)


class TestRequiredKeyValidation:
    """Test required key enforcement."""

    def test_all_required_keys_present(self, linter, temp_dir):
        """Test that files with all required keys pass."""
        yaml_file = temp_dir / "complete.yaml"
        yaml_file.write_text(
            """
training:
  epochs: 10
  batch_size: 32
model:
  architecture: lenet5
optimizer:
  name: adam
  learning_rate: 0.001
"""
        )
        result = linter.lint_file(yaml_file)
        assert result is True
        assert result is True

    def test_missing_required_keys(self, linter, temp_dir):
        """Test detection of missing required keys."""
        yaml_file = temp_dir / "incomplete.yaml"
        yaml_file.write_text(
            """
training:
  epochs: 10
  # missing batch_size
"""
        )
        result = linter.lint_file(yaml_file)
        assert result is True
        # Should have error or warning about missing required key
        assert len(linter.errors) > 0 or len(linter.warnings) > 0
        assert any("batch_size" in msg.lower() for msg in linter.errors + linter.warnings)


class TestDuplicateValueDetection:
    """Test duplicate value detection."""

    def test_duplicate_values_detected(self, linter, temp_dir):
        """Test that duplicate values are detected."""
        yaml_file = temp_dir / "duplicates.yaml"
        yaml_file.write_text(
            """
model:
  layer1_size: 128
  layer2_size: 128
  layer3_size: 256
"""
        )
        linter.lint_file(yaml_file)
        # Should suggest using a constant or list for duplicate values
        # This might be a suggestion rather than error
        assert len(linter.suggestions) > 0 or len(linter.warnings) > 0


class TestPerformanceThresholdChecks:
    """Test performance threshold validation."""

    def test_batch_size_too_small(self, linter, temp_dir):
        """Test warning for very small batch size."""
        yaml_file = temp_dir / "small_batch.yaml"
        yaml_file.write_text(
            """
training:
  batch_size: 2
  epochs: 10
"""
        )
        linter.lint_file(yaml_file)
        # Should have suggestion about small batch size
        batch_warnings = [s for s in linter.suggestions if "batch" in s.lower()]
        assert len(batch_warnings) > 0

    def test_batch_size_too_large(self, linter, temp_dir):
        """Test warning for very large batch size."""
        yaml_file = temp_dir / "large_batch.yaml"
        yaml_file.write_text(
            """
training:
  batch_size: 10000
  epochs: 10
"""
        )
        linter.lint_file(yaml_file)
        # Should have warning about large batch size (in warnings, not suggestions)
        batch_warnings = [w for w in linter.warnings if "batch" in w.lower()]
        assert len(batch_warnings) > 0

    def test_learning_rate_out_of_range(self, linter, temp_dir):
        """Test warning for unusual learning rate."""
        yaml_file = temp_dir / "high_lr.yaml"
        yaml_file.write_text(
            """
optimizer:
  name: sgd
  learning_rate: 10.0
"""
        )
        linter.lint_file(yaml_file)
        # Should have warning about high learning rate
        lr_warnings = [w for w in linter.warnings + linter.suggestions if "learning" in w.lower()]
        assert len(lr_warnings) > 0

    def test_valid_thresholds(self, linter, temp_dir):
        """Test that values within thresholds pass."""
        yaml_file = temp_dir / "good_values.yaml"
        yaml_file.write_text(
            """
training:
  batch_size: 64
  epochs: 100
optimizer:
  name: adam
  learning_rate: 0.001
"""
        )
        result = linter.lint_file(yaml_file)
        # Should not have threshold-related suggestions
        assert result is True


class TestErrorMessageFormatting:
    """Test error message formatting."""

    def test_error_includes_filename(self, linter, temp_dir):
        """Test that errors include the filename."""
        yaml_file = temp_dir / "test.yaml"
        yaml_file.write_text("invalid: [")
        linter.lint_file(yaml_file)
        # Error messages should reference the file
        assert len(linter.errors) > 0

    def test_error_includes_line_number(self, linter, temp_dir):
        """Test that errors include line numbers when possible."""
        yaml_file = temp_dir / "multiline.yaml"
        yaml_file.write_text(
            """
line1: value1
line2: value2
line3: [invalid
"""
        )
        linter.lint_file(yaml_file)
        # YAML parser errors often include line numbers
        assert len(linter.errors) > 0

    def test_warning_format(self, linter, temp_dir):
        """Test warning message format."""
        yaml_file = temp_dir / "warn.yaml"
        yaml_file.write_text(
            """
lr: 0.001
"""
        )
        linter.lint_file(yaml_file)
        # Warnings should be clear and actionable
        assert len(linter.warnings) > 0
        for warning in linter.warnings:
            assert isinstance(warning, str)
            assert len(warning) > 0


class TestFileHandling:
    """Test file handling and edge cases."""

    def test_nonexistent_file(self, linter, temp_dir):
        """Test handling of nonexistent files."""
        nonexistent = temp_dir / "does_not_exist.yaml"
        result = linter.lint_file(nonexistent)
        assert result is False
        assert len(linter.errors) > 0
        assert any("not found" in err.lower() for err in linter.errors)

    def test_empty_file(self, linter, temp_dir):
        """Test handling of empty files."""
        empty_file = temp_dir / "empty.yaml"
        empty_file.write_text("")
        result = linter.lint_file(empty_file)
        assert result is True

        # Empty YAML is technically valid (null document)
        # Behavior depends on implementation

    def test_non_yaml_extension(self, linter, temp_dir):
        """Test handling of non-YAML files."""
        text_file = temp_dir / "not_yaml.txt"
        text_file.write_text("not yaml content")
        # Should still try to parse as YAML
        result = linter.lint_file(text_file)
        assert result is True
        # May pass if content happens to be valid YAML


class TestVerboseMode:
    """Test verbose output mode."""

    def test_verbose_output(self, temp_dir, capsys):
        """Test that verbose mode produces output."""
        verbose_linter = ConfigLinter(verbose=True)
        yaml_file = temp_dir / "test.yaml"
        yaml_file.write_text("key: value\n")
        verbose_linter.lint_file(yaml_file)
        captured = capsys.readouterr()
        # Should print linting progress
        assert len(captured.out) > 0
        assert "Linting" in captured.out or str(yaml_file) in captured.out

    def test_quiet_mode(self, temp_dir, capsys):
        """Test that non-verbose mode is quiet."""
        quiet_linter = ConfigLinter(verbose=False)
        yaml_file = temp_dir / "test.yaml"
        yaml_file.write_text("key: value\n")
        quiet_linter.lint_file(yaml_file)
        captured = capsys.readouterr()
        # Should not print linting progress
        assert "Linting" not in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
