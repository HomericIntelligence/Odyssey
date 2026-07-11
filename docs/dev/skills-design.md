# Skills Design - Taxonomy and Implementation

## Overview

Skills are reusable, algorithmic capabilities that agents can invoke. Unlike sub-agents (which have separate conversation
contexts), skills are computational operations that run within an agent's current context and are automatically invoked
by Claude when appropriate.

## Skills vs Sub-Agents Decision Matrix

| Criterion | Use Skill | Use Sub-Agent |
| --- | --- | --- |
| Decision Making | No - algorithmic/template-based | Yes - requires judgment |
| Context | Runs in current context | Separate conversation thread |
| Invocation | Model-invoked automatically | Explicit delegation |
| Complexity | Simple, focused operation | Complex, multi-step process |
| State | Stateless operation | Persistent state across calls |
| Examples | generate_boilerplate, run_tests | Architecture Design Agent, Code Review Agent |

## Key Principles

1. **Skills are Computational**: Pattern matching, template application, tool orchestration
1. **Sub-Agents are Judgmental**: Require understanding, trade-offs, decision-making
1. **Skills are Reusable**: Used by multiple agent types
1. **Sub-Agents are Specialized**: Focused role with specific responsibilities

---

## Three-Tier Taxonomy

### Tier 1: Foundational Skills

**Used By**: All agents across all levels

Skills that provide basic capabilities needed universally.

#### Code Analysis Skills

#### analyze_code_structure

- Parse Python/Mojo code into AST
- Extract classes, functions, imports
- Map code organization
- **Used by**: All implementation and review agents
- **File**: `.claude/skills/tier-1/analyze-code-structure/SKILL.md`

#### detect_code_smells

- Identify anti-patterns and code smells
- Check for common mistakes
- Flag potential issues
- **Used by**: Implementation and review agents
- **File**: `.claude/skills/tier-1/detect-code-smells/SKILL.md`

#### calculate_complexity

- Compute cyclomatic complexity
- Calculate code metrics
- Identify complex functions
- **Used by**: Performance and review agents
- **File**: `.claude/skills/tier-1/calculate-complexity/SKILL.md`

#### extract_dependencies

- Map import dependencies
- Build dependency graphs
- Identify circular dependencies
- **Used by**: Architecture and integration agents
- **File**: `.claude/skills/tier-1/extract-dependencies/SKILL.md`

#### Code Generation Skills

#### generate_boilerplate

- Create standard file headers
- Generate class/function templates
- Apply code scaffolding
- **Used by**: All implementation engineers
- **File**: `.claude/skills/tier-1/generate-boilerplate/SKILL.md`

#### generate_tests

- Auto-generate test scaffolding
- Create test case templates
- Generate fixture templates
- **Used by**: Test engineers at all levels
- **File**: `.claude/skills/tier-1/generate-tests/SKILL.md`

#### refactor_code

- Apply automated refactorings
- Extract functions/methods
- Rename variables consistently
- **Used by**: Implementation and cleanup agents
- **File**: `.claude/skills/tier-1/refactor-code/SKILL.md`

#### Testing Skills

#### run_tests

- Execute test suites
- Parse test output
- Report results
- **Used by**: All test engineers
- **File**: `.claude/skills/tier-1/run-tests/SKILL.md`

#### calculate_coverage

- Compute test coverage
- Identify uncovered code
- Generate coverage reports
- **Used by**: Test specialists
- **File**: `.claude/skills/tier-1/calculate-coverage/SKILL.md`

#### generate_test_data

- Create test fixtures
- Generate mock data
- Build test databases
- **Used by**: Test engineers
- **File**: `.claude/skills/tier-1/generate-test-data/SKILL.md`

---

### Tier 2: Domain Skills

**Used By**: Specific agent types (ML, documentation, etc.)

Skills specific to particular domains or specialized tasks.

#### Paper Analysis Skills

#### extract_algorithm

- Parse research papers for algorithms
- Extract pseudocode and steps
- Identify algorithm components
- **Used by**: Paper Implementation Orchestrator, Architecture Design Agent
- **File**: `.claude/skills/tier-2/extract-algorithm/SKILL.md`

#### identify_architecture

- Extract neural network architectures from papers
- Parse layer definitions
- Identify model components
- **Used by**: Paper Implementation agents
- **File**: `.claude/skills/tier-2/identify-architecture/SKILL.md`

#### extract_hyperparameters

- Extract training hyperparameters from papers
- Identify learning rates, batch sizes, etc.
- Build hyperparameter configs
- **Used by**: Paper Implementation agents
- **File**: `.claude/skills/tier-2/extract-hyperparameters/SKILL.md`

#### analyze_equations

- Parse mathematical equations
- Convert LaTeX to code
- Extract formulas
- **Used by**: Implementation specialists
- **File**: `.claude/skills/tier-2/analyze-equations/SKILL.md`

#### ML Operations Skills

#### prepare_dataset

- Data loading and preprocessing
- Data augmentation pipelines
- Dataset splitting
- **Used by**: Paper Implementation engineers
- **File**: `.claude/skills/tier-2/prepare-dataset/SKILL.md`

#### train_model

- Training loop templates
- Checkpoint management
- Training orchestration
- **Used by**: Implementation engineers
- **File**: `.claude/skills/tier-2/train-model/SKILL.md`

#### evaluate_model

- Evaluation metrics calculation
- Results visualization
- Performance reporting
- **Used by**: Implementation and performance engineers
- **File**: `.claude/skills/tier-2/evaluate-model/SKILL.md`

#### Documentation Skills

#### generate_docstrings

- Create function/class docstrings
- Follow docstring conventions
- Extract information from code
- **Used by**: Documentation writers
- **File**: `.claude/skills/tier-2/generate-docstrings/SKILL.md`

#### generate_api_docs

- Create API reference documentation
- Build documentation from code
- Generate usage examples
- **Used by**: Documentation specialists
- **File**: `.claude/skills/tier-2/generate-api-docs/SKILL.md`

#### generate_changelog

- Create changelog entries
- Categorize changes
- Format release notes
- **Used by**: Documentation engineers
- **File**: `.claude/skills/tier-2/generate-changelog/SKILL.md`

---

### Tier 3: Specialized Skills

**Used By**: Few agents for narrow use cases

Highly specialized skills for specific scenarios.

#### Security Skills

#### scan_vulnerabilities

- Run security scanners
- Parse scanner output
- Categorize vulnerabilities
- **Used by**: Security specialists
- **File**: `.claude/skills/tier-3/scan-vulnerabilities/SKILL.md`

#### check_dependencies

- Check for vulnerable dependencies
- Scan package versions
- Suggest updates
- **Used by**: Security engineers
- **File**: `.claude/skills/tier-3/check-dependencies/SKILL.md`

#### validate_inputs

- Check input sanitization
- Identify injection risks
- Verify validation logic
- **Used by**: Security implementation specialists
- **File**: `.claude/skills/tier-3/validate-inputs/SKILL.md`

#### Performance Skills

#### profile_code

- Run performance profilers
- Parse profiling output
- Identify hotspots
- **Used by**: Performance engineers
- **File**: `.claude/skills/tier-3/profile-code/SKILL.md`

#### benchmark_functions

- Execute benchmarks
- Compare performance
- Generate benchmark reports
- **Used by**: Performance engineers
- **File**: `.claude/skills/tier-3/benchmark-functions/SKILL.md`

#### suggest_optimizations

- Analyze code for optimization opportunities
- Suggest algorithmic improvements
- Recommend data structure changes
- **Used by**: Performance specialists
- **File**: `.claude/skills/tier-3/suggest-optimizations/SKILL.md`

---

## Skill Configuration Format

Skills follow Claude Code SKILL.md format:

````markdown
---
name: skill-name
description: Brief description of what this skill does and when to use it
allowed-tools: Read,Write,Grep,Bash
---

# Skill Name

## Purpose

[What this skill does]

## When to Use

[Scenarios where this skill should be invoked]

## How It Works

[Step-by-step process]

## Inputs

- Input 1: [description]
- Input 2: [description]

## Outputs

- Output 1: [description]
- Output 2: [description]

## Examples

### Example 1: [Scenario]

**Input**:

```text

[example input]

```text
**Output**:

```text

[example output]

```text
### Example 2: [Another Scenario]

[...]

## Error Handling

- Error 1: [how to handle]
- Error 2: [how to handle]

## Dependencies

- Tool 1
- Tool 2

## Notes

- Note 1
- Note 2

````

---

## Skill Discovery and Activation

### How Skills Are Discovered

1. Claude scans `.claude/skills/` directory
1. Reads SKILL.md frontmatter (name, description)
1. Matches description against current task
1. Automatically invokes when appropriate

### Activation Triggers

Skills activate when:

- Task description matches skill description
- Agent requests capability that skill provides
- Context matches skill's usage scenarios

### Example Activation

```text
User: "Generate test boilerplate for this function"
→ Claude identifies: generate_tests skill matches
→ Skill activates automatically
→ Test scaffolding generated
```text

---

## Skills by Agent Type

### Architecture Design Agent Uses

- Tier 1: analyze_code_structure, extract_dependencies
- Tier 2: extract_algorithm, identify_architecture
- Tier 3: None typically

### Implementation Engineers Use

- Tier 1: generate_boilerplate, refactor_code, run_tests
- Tier 2: prepare_dataset, train_model (for ML code)
- Tier 3: None typically

### Test Engineers Use

- Tier 1: generate_tests, run_tests, calculate_coverage
- Tier 2: generate_test_data
- Tier 3: None typically

### Documentation Writers Use

- Tier 1: None typically
- Tier 2: generate_docstrings, generate_api_docs, generate_changelog
- Tier 3: None typically

### Security Specialists Use

- Tier 1: detect_code_smells
- Tier 2: None typically
- Tier 3: scan_vulnerabilities, check_dependencies, validate_inputs

### Performance Engineers Use

- Tier 1: calculate_complexity
- Tier 2: evaluate_model
- Tier 3: profile_code, benchmark_functions, suggest_optimizations

---

## Implementation Priority

### Phase 1: Essential Tier 1 Skills (Issues 68-73)

1. generate_boilerplate
1. run_tests
1. analyze_code_structure

### Phase 2: Core Tier 2 Skills

1. extract_algorithm
1. generate_docstrings
1. prepare_dataset

### Phase 3: Specialized Skills

1. As needed based on usage
1. Add when specific use cases emerge

---

## Testing Skills

Each skill should have:

1. **Validation tests**: Verify skill loads correctly
1. **Functionality tests**: Test skill operations
1. **Integration tests**: Test skill usage by agents
1. **Example tests**: Verify examples in SKILL.md work

---

## Skills vs Sub-Agents Examples

### Use Skill: generate_boilerplate

**Why**: Algorithmic, template-based, no decisions required
**Process**: Apply template → Fill in names → Return code

### Use Sub-Agent: Architecture Design Agent

**Why**: Requires judgment, trade-offs, understanding context
**Process**: Analyze requirements → Consider alternatives → Make decisions → Document rationale

### Use Skill: run_tests

**Why**: Execute commands, parse output, report results
**Process**: Run pytest → Parse output → Format results

### Use Sub-Agent: Test Design Specialist

**Why**: Decide what to test, how to test, coverage goals
**Process**: Analyze code → Identify test cases → Design fixtures → Plan coverage

---

## Best Practices

1. **Keep Skills Focused**: One clear capability per skill
1. **Clear Descriptions**: Make it obvious when to use the skill
1. **Complete Examples**: Show realistic usage scenarios
1. **Error Handling**: Document how to handle failures
1. **Tool Restrictions**: Only request necessary tools
1. **Tier Appropriately**: Place in correct tier based on usage breadth

---

## References

- Claude Code Skills Documentation: <https://code.claude.com/docs/en/skills>
- [Agent Hierarchy](https://github.com/HomericIntelligence/Odyssey/blob/main/agents/hierarchy.md)
- [Orchestration Patterns](./orchestration-patterns.md)
