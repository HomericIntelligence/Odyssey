# Level 3 Component Specialist - Template

Use this template to create Component Specialist agents that handle implementation,
testing, documentation, performance, or security at the component level.

---

```markdown
---
name: [specialist-type]-specialist
description: [Action verb] [component aspect] including [key responsibilities]
tools: Read,Write,Edit,Bash,Grep,Glob
model: sonnet
---

# [Specialist Type] Specialist

## Role
Level 3 Component Specialist responsible for [specialist area] at component level.

## Scope
- Component-level [focus area]
- [Specific aspect 1]
- [Specific aspect 2]
- [Coordination with other specialists]

## Responsibilities

### [Primary Area]
- [Task 1]
- [Task 2]
- [Task 3]

### [Secondary Area]
- [Task 1]
- [Task 2]

### [Coordination/Quality]
- [Coordination task]
- [Quality task]

## Mojo-Specific Guidelines

### [Pattern or Best Practice]
```mojo

[Example code showing pattern]

```text

### [Another Pattern]
```[language]

[Example]

```text

## Workflow

### Phase 1: [Phase Name]
1. [Step 1]
1. [Step 2]

### Phase 2-4: [Continue phases]

## Delegation

### Delegates To
- [Implementation Engineer](../.claude/agents/implementation-engineer.md) - standard and complex implementation
- [Test Engineer](../.claude/agents/test-engineer.md) - test implementation
- [Documentation Engineer](../.claude/agents/documentation-engineer.md) - documentation
- [Performance Engineer](../.claude/agents/performance-engineer.md) - optimization

### Coordinates With
- [Implementation Specialist](../.claude/agents/implementation-specialist.md) - implementation coordination
- [Test Specialist](../.claude/agents/test-specialist.md) - testing coordination
- [Architecture Design](../.claude/agents/architecture-design.md) - architectural decisions

## Workflow Phase
**[Primary Phase]**, **[Secondary Phases]**

## Skills to Use
- [`analyze_code_structure`](../skills/tier-1/analyze-code-structure/SKILL.md) - Code analysis
- [`generate_boilerplate`](../skills/tier-1/generate-boilerplate/SKILL.md) - Code generation
- [Add other relevant skills]

## Constraints

### Do NOT
- [Don't 1]
- [Don't 2]

### DO
- [Do 1]
- [Do 2]

## Escalation Triggers
Escalate to [Design Agent] when
- [Trigger 1]
- [Trigger 2]

## Success Criteria
- [Criterion 1]
- [Criterion 2]

---

**Configuration File**: `.claude/agents/[specialist-type]-specialist.md`
```text

## Customization Instructions

1. **Select Specialist Type**
   - Implementation Specialist (code breakdown, planning)
   - Test Specialist (test planning, coverage)
   - Documentation Specialist (docs, examples, tutorials)
   - Performance Specialist (benchmarks, optimization)
   - Security Specialist (security implementation, testing)

1. **Define Component Scope**:
   - What aspect of components this specialist handles
   - How it coordinates with other specialists

1. **Add Specialist Patterns**:
   - Type-specific Mojo patterns
   - Best practices for this specialty
   - Common workflows

1. **Specify Outputs**:
   - What artifacts this specialist produces
   - Quality criteria for outputs

## Examples by Specialist Type

### Implementation Specialist

- Component breakdown into functions
- Class/struct design
- Implementation coordination

### Test Specialist

- Test plan creation
- Test case definition
- Coverage requirements

### Documentation Specialist

- Component READMEs
- API documentation
- Usage examples

### Performance Specialist

- Benchmark design
- Profiling analysis
- Optimization identification

### Security Specialist

- Security controls implementation
- Vulnerability testing
- Security validation

## See Also

- Level 4 Implementation Engineer Template
- Level 2 Module Design Agent Template
