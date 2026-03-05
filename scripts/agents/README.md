# Agent Validation Scripts

This directory contains validation and utility scripts for the agent system in `.claude/agents/`.

**Note**: Historical and experimental scripts have been moved to the `playground/` subdirectory.
See [playground/README.md](playground/README.md) for details on deprecated scripts.

## Scripts

### validate_agents.py

Comprehensive validation of all agent configuration files.

### Checks

- YAML frontmatter syntax and required fields
- Tool names are valid
- Required sections present (Role, Scope, Responsibilities, Mojo-Specific Guidelines, Workflow, Constraints)
- Mojo-specific content included
- Delegation patterns defined
- Workflow phases specified
- Links to other agents are valid

### Usage

```bash
# Validate all agents
python scripts/agents/validate_agents.py

# Validate with verbose output (shows warnings)
python scripts/agents/validate_agents.py --verbose

# Custom agents directory
python scripts/agents/validate_agents.py --agents-dir /path/to/agents
```text

### Exit Codes:

- 0 = All validations passed
- 1 = Errors found

### check_frontmatter.py

Verify YAML frontmatter in agent configuration files.

### Checks

- YAML syntax validity
- Required fields present (name, description, tools, model)
- Field types are correct (all strings)
- Model name is valid
- Name format follows conventions (lowercase-with-hyphens)
- Tools field is not empty

### Usage

```bash
# Check all agent frontmatter
python scripts/agents/check_frontmatter.py

# Check with verbose output
python scripts/agents/check_frontmatter.py --verbose

# Custom agents directory
python scripts/agents/check_frontmatter.py --agents-dir /path/to/agents
```text

### Exit Codes:

- 0 = All frontmatter is valid
- 1 = Errors found

### test_agent_loading.py

Test agent discovery and loading.

### Checks

- All markdown files can be read
- YAML frontmatter can be parsed
- Required fields are present
- No duplicate agent names
- Files are accessible

### Usage

```bash
# Test agent loading
python scripts/agents/test_agent_loading.py

# Test with verbose output
python scripts/agents/test_agent_loading.py --verbose

# Custom agents directory
python scripts/agents/test_agent_loading.py --agents-dir /path/to/agents
```text

### Exit Codes:

- 0 = All agents loaded successfully
- 1 = Errors during loading

### list_agents.py

Display all available agents organized by level.

### Features:

- Lists agents by level (0-5)
- Shows name, description, and tools
- Supports filtering by level
- Compact and verbose display modes

### Agent Levels:

- 0 - Meta-Orchestrator (Chief Architect)
- 1 - Section Orchestrators (Foundation, Shared Library, Tooling, Papers, CI/CD, Agentic Workflows)
- 2 - Design Agents (Architecture, Integration, Security)
- 3 - Component Specialists (Implementation, Test, Documentation, Performance, Security)
- 4 - Senior Engineers
- 5 - Junior Engineers

### Usage

```bash
# List all agents
python scripts/agents/list_agents.py

# List with verbose details
python scripts/agents/list_agents.py --verbose

# List only level 1 agents (Section Orchestrators)
python scripts/agents/list_agents.py --level 1

# List only level 5 agents (Junior Engineers)
python scripts/agents/list_agents.py --level 5 --verbose

# Custom agents directory
python scripts/agents/list_agents.py --agents-dir /path/to/agents
```text

### Exit Codes:

- 0 = Success
- 1 = Errors occurred

## Common Usage Patterns

### Pre-commit Validation

Run validation before committing changes to agent files:

```bash
python scripts/agents/validate_agents.py
```text

### Quick Health Check

Check if all agents can be loaded:

```bash
python scripts/agents/test_agent_loading.py
```text

### Explore Available Agents

See all available agents by category:

```bash
# All agents
python scripts/agents/list_agents.py

# Just orchestrators
python scripts/agents/list_agents.py --level 1

# Just junior engineers
python scripts/agents/list_agents.py --level 5
```text

### Detailed Validation

Get comprehensive validation with all warnings:

```bash
python scripts/agents/validate_agents.py --verbose
```text

## Integration with CI/CD

These scripts can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Validate Agent Configurations
  run: |
    python scripts/agents/validate_agents.py
    python scripts/agents/test_agent_loading.py
```text

## Requirements

All scripts require Python 3.7+ and PyYAML:

```bash
pip install pyyaml
```text

## Exit Codes

All scripts follow consistent exit code conventions:

- `0` - Success (all checks passed)
- `1` - Errors found (validation failed)

This allows them to be used in shell scripts and CI/CD pipelines:

```bash
# Example: Run validation and exit on failure
python scripts/agents/validate_agents.py || exit 1
```text

## Development

### Adding New Validations

To add new validation checks:

1. Add the check logic to `validate_agents.py`
1. Update the `ValidationResult` class if needed
1. Add corresponding test cases
1. Update this README

### Modifying Required Fields

Required fields are defined in each script:

- `REQUIRED_FIELDS` - Required frontmatter fields
- `REQUIRED_SECTIONS` - Required markdown sections
- `VALID_TOOLS` - Valid tool names
- `VALID_MODELS` - Valid model names

Update these constants to change validation requirements.

## Troubleshooting

### PyYAML Not Found

```bash
pip install pyyaml
```text

### Permission Denied

Make scripts executable:

```bash
chmod +x scripts/agents/*.py
```text

### Agent Directory Not Found

Ensure you're running from the repository root or use `--agents-dir`:

```bash
python scripts/agents/validate_agents.py --agents-dir /path/to/agents
```text

## See Also

- [Agent Documentation](../../agents/README.md) - Team documentation
- [Agent Hierarchy](../../agents/hierarchy.md) - Complete agent specifications
- [Delegation Rules](../../agents/delegation-rules.md) - Coordination patterns
