#!/usr/bin/env python3
"""Tests for check_coverage.py mock warning functionality.

Tests verify that parse_coverage_report() properly displays warnings
when it returns mock coverage data.
"""

import sys
import tempfile
from io import StringIO
from pathlib import Path
from unittest import TestCase, main

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
from check_coverage import parse_coverage_report


class TestParseCoverageReport(TestCase):
    """Test cases for parse_coverage_report function."""

    def test_existing_file_returns_none(self):
        """Verify function returns None for existing files (coverage not implemented)."""
        with tempfile.NamedTemporaryFile(suffix=".xml") as f:
            result = parse_coverage_report(Path(f.name))
            self.assertIsNone(result)

    def test_nonexistent_file_returns_none(self):
        """Verify function returns None for nonexistent files."""
        result = parse_coverage_report(Path("/nonexistent/coverage.xml"))
        self.assertIsNone(result)

    def test_warning_printed_for_existing_file(self):
        """Verify warning is printed when file exists."""
        with tempfile.NamedTemporaryFile(suffix=".xml") as f:
            captured = StringIO()
            sys.stdout = captured
            try:
                parse_coverage_report(Path(f.name))
            finally:
                sys.stdout = sys.__stdout__

            output = captured.getvalue()
            self.assertIn("WARNING", output)

    def test_warning_printed_for_nonexistent_file(self):
        """Verify warning is printed when file doesn't exist."""
        captured = StringIO()
        sys.stdout = captured
        try:
            parse_coverage_report(Path("/nonexistent/coverage.xml"))
        finally:
            sys.stdout = sys.__stdout__

        output = captured.getvalue()
        self.assertIn("WARNING", output)

    def test_warning_contains_reference_to_adrs_and_issues(self):
        """Verify warning references ADR-008 and related issues."""
        with tempfile.NamedTemporaryFile(suffix=".xml") as f:
            captured = StringIO()
            sys.stdout = captured
            try:
                parse_coverage_report(Path(f.name))
            finally:
                sys.stdout = sys.__stdout__

            output = captured.getvalue()
            self.assertIn("ADR-008", output)
            self.assertIn("#2583", output)
            self.assertIn("#2612", output)

    def test_warning_explains_mojo_limitation(self):
        """Verify warning explains Mojo lacks coverage instrumentation."""
        with tempfile.NamedTemporaryFile(suffix=".xml") as f:
            captured = StringIO()
            sys.stdout = captured
            try:
                parse_coverage_report(Path(f.name))
            finally:
                sys.stdout = sys.__stdout__

            output = captured.getvalue()
            self.assertIn("Mojo does not provide coverage instrumentation", output)
            self.assertIn("mojo test --coverage", output)

    def test_warning_is_prominent(self):
        """Verify warning uses prominent formatting (emoji and borders)."""
        with tempfile.NamedTemporaryFile(suffix=".xml") as f:
            captured = StringIO()
            sys.stdout = captured
            try:
                parse_coverage_report(Path(f.name))
            finally:
                sys.stdout = sys.__stdout__

            output = captured.getvalue()
            self.assertIn("⚠️", output)
            self.assertIn("=", output)
            self.assertIn("➤", output)


if __name__ == "__main__":
    main()
