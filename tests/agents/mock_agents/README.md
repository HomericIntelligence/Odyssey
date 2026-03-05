# Mock Agent Configurations

## Purpose

This directory contains mock agent configurations for testing the agent validation and testing infrastructure
before real agents are implemented in issue #64.

## Mock Agents

The following mock agents represent one agent from each level of the 6-level hierarchy:

| Level | Agent | File | Purpose |
|-------|-------|------|---------|
| 0 | Chief Architect | `chief-architect.md` | Meta-orchestrator for system-wide decisions |
| 1 | Foundation Orchestrator | `foundation-orchestrator.md` | Section orchestrator for foundation |
| 2 | Architecture Design | `architecture-design.md` | Module-level architecture design |
| 3 | Senior Implementation Specialist | `senior-implementation-specialist.md` | Complex component design |
| 4 | Implementation Engineer | `implementation-engineer.md` | Standard Mojo function implementation |
| 5 | Junior Implementation Engineer | `junior-implementation-engineer.md` | Simple functions and boilerplate |

## Features

Each mock agent includes:

- **Valid YAML frontmatter**: name, description, tools, model
- **Delegation information**: Delegates to, Coordinates with, Escalates to
- **Workflow phase participation**: Which phases the agent participates in
- **Escalation triggers**: When to escalate to higher level
- **Examples**: Concrete examples of agent usage
- **Mojo patterns** (Levels 3-5): fn vs def, struct vs class, SIMD, memory management

## Testing

Use these mock agents to test the validation scripts:

```bash
# Run all tests against mock agents
python3 tests/agents/validate_configs.py tests/agents/mock_agents/
python3 tests/agents/test_loading.py tests/agents/mock_agents/
python3 tests/agents/test_delegation.py tests/agents/mock_agents/
python3 tests/agents/test_integration.py tests/agents/mock_agents/
python3 tests/agents/test_mojo_patterns.py tests/agents/mock_agents/
```text

## Expected Results

When testing with these mock agents, you should see:

### validate_configs.py

- All 6 agents pass validation
- Zero critical errors
- Minimal warnings (if any)
- Proper frontmatter syntax
- Required fields present

### test_loading.py

- All 6 agents discovered
- All levels 0-5 represented
- Activation keywords detected in descriptions
- Tool usage appropriate for roles
- Model distribution: mostly sonnet, opus for Chief Architect

### test_delegation.py

- Complete delegation chain: 0→1→2→3→4→5
- Escalation paths: 5→4→3→2→1→0
- Escalation triggers defined for each agent
- Horizontal coordination mentioned

### test_integration.py

- Plan phase: Levels 0-3
- Test/Impl/Package phases: Levels 3-5 (parallel)
- Cleanup phase: All levels
- Parallel execution guidance for levels 3-5
- Worktree coordination for levels 4-5

### test_mojo_patterns.py

- Levels 3-5 have Mojo guidance
- fn vs def: Present in levels 3-5
- struct vs class: Present in levels 3-5
- SIMD: Present in level 3
- Memory management: Present in levels 3-5
- High completeness scores for implementation agents

## Differences from Real Agents

Mock agents are simplified:

- Less detailed than real agents will be
- Focused on demonstrating key patterns
- May not cover all edge cases
- Serve as validation baseline

Real agents (from issue #64) will:

- Have more comprehensive guidance
- Include more examples
- Cover more scenarios
- Have domain-specific details

## Usage

1. **Initial Testing**: Run all validation scripts against mock agents
1. **Baseline**: Document results as baseline in test-results.md
1. **Script Validation**: Verify test scripts work correctly
1. **Fix Issues**: Fix any script bugs discovered
1. **Transition**: Once issue #64 completes, test against real agents

## Maintenance

These mock agents should:

- Remain stable for baseline testing
- Not be modified unless test requirements change
- Serve as examples of proper agent structure
- Be referenced in documentation

## See Also

- [Test Plan](../../../notes/issues/63/test-plan.md) - Comprehensive test plan
- [Test Results](../../../notes/issues/63/test-results.md) - Test execution results
- [Agent Hierarchy](../../../agents/hierarchy.md) - Complete specifications
