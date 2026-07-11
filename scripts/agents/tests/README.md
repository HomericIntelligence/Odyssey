# Agent Testing Suite

Comprehensive quality assurance tests for the ML Odyssey agent system.

## Overview

This test suite validates:

- **Integration**: Agent configuration files, YAML frontmatter, references
- **Documentation**: Markdown files, links, table of contents
- **Scripts**: Utility scripts for agent management

## Test Files

### `conftest.py`

Pytest configuration and shared fixtures:

- Path configuration (repo_root, agents_dir, skills_dir, docs_agents_dir)
- File discovery fixtures (all_agent_files, all_skill_files, all_doc_files)
- YAML frontmatter parsing helpers
- Link validation utilities
- Test markers and parametrization helpers

### `test_integration.py`

Integration tests for agent configuration files:

- Agent file existence and readability
- YAML frontmatter parsing and validation
- Required frontmatter keys (name, description, tools, model)
- Skill reference resolution
- Agent reference validation
- Required section structure
- Edge cases and error handling

### `test_documentation.py`

Documentation validation tests:

- Documentation file existence
- Internal link validation
- Table of contents accuracy
- Cross-reference validation
- Content quality checks
- Markdown syntax validation
- Consistency across documentation

### `test_scripts.py`

Script validation tests:

- Script existence and executability
- Help option functionality
- validate_agents.py behavior
- list_agents.py functionality
- agent_stats.py output formats
- check_frontmatter.py validation
- Error handling
- Integration between scripts

## Running Tests

### Run All Tests

```bash
pytest scripts/agents/tests/
```text

### Run Specific Test File

```bash
# Integration tests only
pytest scripts/agents/tests/test_integration.py

# Documentation tests only
pytest scripts/agents/tests/test_documentation.py

# Script tests only
pytest scripts/agents/tests/test_scripts.py
```text

### Run Tests by Marker

```bash
# Run integration tests
pytest -m integration scripts/agents/tests/

# Run documentation tests
pytest -m documentation scripts/agents/tests/

# Run script tests
pytest -m scripts scripts/agents/tests/

# Exclude slow tests
pytest -m "not slow" scripts/agents/tests/
```text

### Run Specific Test Class

```bash
# Test agent file existence
pytest scripts/agents/tests/test_integration.py::TestAgentFilesExist

# Test YAML frontmatter
pytest scripts/agents/tests/test_integration.py::TestAgentFrontmatter

# Test documentation links
pytest scripts/agents/tests/test_documentation.py::TestInternalLinks
```text

### Run Specific Test

```bash
# Test specific functionality
pytest scripts/agents/tests/test_integration.py::TestAgentFrontmatter::test_agent_has_frontmatter

# Test specific script
pytest scripts/agents/tests/test_scripts.py::TestValidateAgentsScript::test_help_option
```text

### Verbose Output

```bash
# Show detailed output
pytest -v scripts/agents/tests/

# Show print statements
pytest -s scripts/agents/tests/

# Show detailed failures
pytest -vv scripts/agents/tests/
```text

### Coverage Report

```bash
# Run with coverage
pytest --cov=scripts/agents --cov-report=html scripts/agents/tests/

# View coverage report
open htmlcov/index.html
```text

## Test Organization

### Test Classes

Tests are organized into classes by functionality:

### Integration Tests

- `TestAgentFilesExist` - File existence and readability
- `TestAgentFrontmatter` - YAML frontmatter validation
- `TestSkillReferences` - Skill reference resolution
- `TestAgentReferences` - Agent cross-references
- `TestAgentStructure` - Required sections
- `TestEdgeCases` - Edge cases and error handling

### Documentation Tests

- `TestDocumentationFilesExist` - File existence
- `TestInternalLinks` - Link validation
- `TestTableOfContents` - TOC accuracy
- `TestCrossReferences` - Cross-document references
- `TestContentQuality` - Content quality checks
- `TestMarkdownSyntax` - Markdown validation
- `TestConsistency` - Consistency checks

### Script Tests

- `TestScriptsExist` - Script existence
- `TestValidateAgentsScript` - validate_agents.py
- `TestListAgentsScript` - list_agents.py
- `TestAgentStatsScript` - agent_stats.py
- `TestCheckFrontmatterScript` - check_frontmatter.py
- `TestScriptErrorHandling` - Error handling
- `TestScriptOutputFormats` - Output validation
- `TestScriptIntegration` - Script integration

### Test Markers

Custom markers for filtering tests:

- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.documentation` - Documentation tests
- `@pytest.mark.scripts` - Script validation tests
- `@pytest.mark.slow` - Slow-running tests

## Fixtures

### Session Fixtures

Computed once per test session:

- `repo_root` - Repository root directory
- `agents_dir` - .claude/agents directory
- `skills_dir` - .claude/skills directory
- `docs_agents_dir` - agents/ documentation directory
- `all_agent_files` - All agent configuration files
- `all_skill_files` - All skill configuration files
- `all_doc_files` - All documentation files

### Function Fixtures

Created for each test:

- `parse_agent_file` - Factory to parse agent files
- `validate_link_exists` - Factory to validate links
- `sample_valid_frontmatter` - Sample valid frontmatter
- `sample_valid_agent_content` - Sample valid agent content

## Helper Functions

### YAML Frontmatter

- `parse_frontmatter(content)` - Parse YAML frontmatter
- `validate_frontmatter_keys(frontmatter, required, optional)` - Validate keys

### Link Extraction

- `extract_links(content)` - Extract all markdown links
- `extract_skill_references(content)` - Extract skill references
- `extract_agent_references(content)` - Extract agent references

### Path Resolution

- `resolve_relative_path(base_path, relative_path)` - Resolve relative paths

### Documentation

- `extract_toc_entries(content)` - Extract TOC entries
- `extract_headings(content)` - Extract markdown headings

## CI Integration

These tests are designed to run in CI/CD pipelines:

```yaml
# .github/workflows/test-agents.yml
name: Test Agents
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install pytest pytest-cov pyyaml
      - name: Run tests
        run: |
          pytest scripts/agents/tests/ -v --cov=scripts/agents
```text

## Writing New Tests

### Test Naming Convention

- Test files: `test_*.py`
- Test classes: `Test*`
- Test methods: `test_*`

### Example Test

```python
import pytest
from pathlib import Path

@pytest.mark.integration
class TestMyFeature:
    """Test description."""

    def test_basic_functionality(self, agents_dir: Path):
        """Test that basic functionality works."""
        # Arrange
        expected = "expected_value"

        # Act
        actual = some_function()

        # Assert
        assert actual == expected, "Descriptive error message"

    @pytest.mark.parametrize("agent_file", pytest.lazy_fixture("all_agent_files"),
                           ids=lambda f: f.stem)
    def test_all_agents(self, agent_file: Path):
        """Test applied to all agent files."""
        # Test logic here
        pass
```text

### Best Practices

1. **Use descriptive test names** - Test name should describe what is being tested
1. **Include docstrings** - Explain what the test validates
1. **Use fixtures** - Leverage shared fixtures for common setup
1. **Parametrize** - Use `@pytest.mark.parametrize` for testing multiple inputs
1. **Clear assertions** - Include descriptive error messages
1. **Handle edge cases** - Test both success and failure scenarios
1. **Use markers** - Tag tests appropriately for filtering

## Troubleshooting

### Common Issues

### No agents found

```bash
# Check that .claude/agents/ exists and contains .md files
ls -la .claude/agents/
```text

### Import errors

```bash
# Install required dependencies
pip install pytest pyyaml
```text

### Fixture not found

```bash
# Make sure conftest.py is in the tests directory
ls scripts/agents/tests/conftest.py
```text

### Tests not discovered

```bash
# Run with discovery info
pytest --collect-only scripts/agents/tests/
```text

### Debugging Tests

```bash
# Run with pdb debugger
pytest --pdb scripts/agents/tests/

# Run last failed tests
pytest --lf scripts/agents/tests/

# Show local variables on failure
pytest -l scripts/agents/tests/
```text

## Dependencies

Required Python packages:

- `pytest` (>=7.0.0) - Testing framework
- `pyyaml` (>=6.0) - YAML parsing
- `pytest-cov` (optional) - Coverage reporting

Install with:

```bash
pip install pytest pyyaml pytest-cov
```text

## Contributing

When adding new agent features:

1. Add corresponding tests to appropriate test file
1. Update fixtures if new file types are added
1. Run full test suite before committing
1. Ensure 100% pass rate (or document expected failures)

## License

Same as parent project (Odyssey).
