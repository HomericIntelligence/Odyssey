#!/usr/bin/env python3
"""Tests for check_coverage.py coverage report parsing functionality.

Tests verify that parse_coverage_report() correctly parses Cobertura XML
coverage reports and extracts coverage percentages.
"""

import sys
import tempfile
from pathlib import Path
from unittest import TestCase, main

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
from check_coverage import parse_coverage_report


class TestParseCoverageReport(TestCase):
    """Test cases for parse_coverage_report function."""

    def test_nonexistent_file_returns_none(self):
        """Verify function returns None for nonexistent files."""
        result = parse_coverage_report(Path("/nonexistent/coverage.xml"))
        self.assertIsNone(result)

    def test_parses_valid_cobertura_xml_with_100_percent(self):
        """Verify function parses Cobertura XML and extracts coverage percentage."""
        xml_content = b"""<?xml version="1.0" ?>
<coverage version="7.13.5" line-rate="1.0" branch-rate="0" complexity="0">
    <sources><source>/test</source></sources>
    <packages>
        <package name="test" line-rate="1.0" branch-rate="0">
            <classes>
                <class name="test.py" line-rate="1.0" branch-rate="0">
                    <lines><line number="1" hits="1"/></lines>
                </class>
            </classes>
        </package>
    </packages>
</coverage>"""
        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
            f.write(xml_content)
            f.flush()
            try:
                result = parse_coverage_report(Path(f.name))
                self.assertEqual(result, 100.0)
            finally:
                Path(f.name).unlink()

    def test_parses_valid_cobertura_xml_with_partial_coverage(self):
        """Verify function correctly converts decimal line-rate to percentage."""
        xml_content = b"""<?xml version="1.0" ?>
<coverage version="7.13.5" line-rate="0.85" branch-rate="0" complexity="0">
    <sources><source>/test</source></sources>
    <packages>
        <package name="test" line-rate="0.85" branch-rate="0">
            <classes>
                <class name="test.py" line-rate="0.85" branch-rate="0">
                    <lines><line number="1" hits="1"/></lines>
                </class>
            </classes>
        </package>
    </packages>
</coverage>"""
        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
            f.write(xml_content)
            f.flush()
            try:
                result = parse_coverage_report(Path(f.name))
                self.assertAlmostEqual(result, 85.0, places=1)
            finally:
                Path(f.name).unlink()

    def test_parses_valid_cobertura_xml_with_zero_coverage(self):
        """Verify function handles 0% coverage correctly."""
        xml_content = b"""<?xml version="1.0" ?>
<coverage version="7.13.5" line-rate="0.0" branch-rate="0" complexity="0">
    <sources><source>/test</source></sources>
    <packages/>
</coverage>"""
        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
            f.write(xml_content)
            f.flush()
            try:
                result = parse_coverage_report(Path(f.name))
                self.assertEqual(result, 0.0)
            finally:
                Path(f.name).unlink()

    def test_returns_none_for_malformed_xml(self):
        """Verify function returns None for malformed XML files."""
        xml_content = b"<invalid>xml without proper closure"
        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
            f.write(xml_content)
            f.flush()
            try:
                result = parse_coverage_report(Path(f.name))
                self.assertIsNone(result)
            finally:
                Path(f.name).unlink()

    def test_returns_none_when_line_rate_missing(self):
        """Verify function returns None when line-rate attribute is missing."""
        xml_content = b"""<?xml version="1.0" ?>
<coverage version="7.13.5" branch-rate="0" complexity="0">
    <sources><source>/test</source></sources>
</coverage>"""
        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
            f.write(xml_content)
            f.flush()
            try:
                result = parse_coverage_report(Path(f.name))
                self.assertIsNone(result)
            finally:
                Path(f.name).unlink()

    def test_returns_none_for_invalid_line_rate_value(self):
        """Verify function returns None when line-rate value is not numeric."""
        xml_content = b"""<?xml version="1.0" ?>
<coverage version="7.13.5" line-rate="invalid" branch-rate="0" complexity="0">
    <sources><source>/test</source></sources>
</coverage>"""
        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
            f.write(xml_content)
            f.flush()
            try:
                result = parse_coverage_report(Path(f.name))
                self.assertIsNone(result)
            finally:
                Path(f.name).unlink()

    def test_parses_coverage_py_generated_report(self):
        """Verify function parses real coverage.py generated reports."""
        xml_content = b"""<?xml version="1.0" ?>
<coverage version="7.13.5" timestamp="1780098551769" lines-valid="14" lines-covered="13" line-rate="0.9286" branches-covered="0" branches-valid="0" branch-rate="0" complexity="0">
	<!-- Generated by coverage.py: https://coverage.readthedocs.io/en/7.13.5 -->
	<sources>
		<source>/test</source>
	</sources>
	<packages>
		<package name="." line-rate="0.9286" branch-rate="0" complexity="0">
			<classes>
				<class name="test.py" filename="test.py" complexity="0" line-rate="0.9286" branch-rate="0">
					<methods/>
					<lines>
						<line number="1" hits="1"/>
						<line number="2" hits="1"/>
					</lines>
				</class>
			</classes>
		</package>
	</packages>
</coverage>"""
        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
            f.write(xml_content)
            f.flush()
            try:
                result = parse_coverage_report(Path(f.name))
                self.assertAlmostEqual(result, 92.86, places=2)
            finally:
                Path(f.name).unlink()


if __name__ == "__main__":
    main()
