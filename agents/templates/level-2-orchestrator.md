# Level 2 Orchestrator - Template

Use this template to create Level 2 Orchestrator agents that coordinate multiple specialist agents to accomplish
complex, multi-dimensional tasks.

---

## How to Use This Template

### 1. Replace Placeholders

- `[ORCHESTRATOR-NAME]` - Kebab-case name (e.g., "code-review-orchestrator")
- `[ORCHESTRATOR-TITLE]` - Human-readable title (e.g., "Code Review Orchestrator")
- `[PRIMARY-DOMAIN]` - Main responsibility domain (e.g., "code reviews", "CI/CD pipeline", "testing")
- `[SPECIALIST-COUNT]` - Number of specialists this orchestrator coordinates
- `[SPECIALIST-LIST]` - Comma-separated list of specialist names

### 2. Customize Sections

- **Routing Rules**: Define how work is distributed to specialists
- **Delegates To**: List all specialist agents with their responsibilities
- **Example Scenarios**: Add 2-4 realistic scenarios showing routing decisions
- **Tools**: Add/remove tools based on orchestrator needs (Read, Grep, Glob are common)

### 3. Define Coordination Patterns

- **Analysis Phase**: How does this orchestrator understand incoming work?
- **Routing Phase**: What rules determine specialist assignment?
- **Consolidation Phase**: How is specialist output combined?
- **Reporting Phase**: What deliverables does this orchestrator produce?

### 4. Specify Overlap Prevention

- Define clear dimensions for specialist routing
- Create routing tables that prevent duplicate work
- Establish conflict resolution rules

---

## Template Configuration (YAML Frontmatter)

```yaml
---
name: [ORCHESTRATOR-NAME]
description: [One-line description: coordinates X specialists to accomplish Y by routing Z]
tools: Read,Grep,Glob  # Adjust based on needs
model: opus
---
```text

---

## [ORCHESTRATOR-TITLE]

### Role

Level 2 orchestrator responsible for coordinating [PRIMARY-DOMAIN] across the ProjectOdyssey project. Analyzes [work items]
and routes different aspects to specialized [type] agents, ensuring thorough coverage without overlap.

### Scope

- **Authority**: Assigns [task type] to [SPECIALIST-COUNT] specialized [specialist type] agents based on [analysis criteria]
- **Coverage**: [What this orchestrator is responsible for - be specific]
- **Coordination**: Ensures each aspect is [handled/reviewed/implemented] by exactly one appropriate specialist
- **Focus**: [Key quality attributes - e.g., quality, correctness, security, performance]

### Responsibilities

#### 1. [Work Item] Analysis

- Analyze [incoming work items] and determine [processing] scope
- Identify [key attributes that drive routing decisions]
- Assess [impact, priority, dependencies, etc.]
- Determine required specialist [involvement/reviews/assignments]

#### 2. [Task] Routing

<!-- Replace with specific routing responsibilities -->
<!-- Example from code-review-orchestrator: -->
<!-- - Route code changes to Implementation Review Specialist -->
<!-- - Route Mojo-specific patterns to Mojo Language Review Specialist -->
<!-- Add 5-10 routing rules specific to your orchestrator -->

- Route [type A work] to [Specialist A]
- Route [type B work] to [Specialist B]
- Route [type C work] to [Specialist C]
- Route [cross-cutting concerns] to [multiple specialists with clear dimension separation]

#### 3. [Coordination Type] Coordination

- Prevent overlapping [work/reviews/tasks] through clear routing rules
- Consolidate [output/feedback/results] from multiple specialists
- Identify conflicts between specialist [recommendations/outputs/results]
- Escalate [unresolvable issues] to [higher-level authority]

#### 4. Quality Assurance

- Ensure all critical aspects are [handled/reviewed/processed]
- Verify specialist coverage is complete
- Track [completion/progress] status
- Generate consolidated [summary/report/output]

### Workflow

#### Phase 1: Analysis

```text
1. Receive [trigger event - e.g., PR notification, issue assignment, user request]
2. [Gather information - what data do you need?]
   - Use Glob to [find files/list items]
   - Use Read to [understand content/context]
   - Use Grep to [search for patterns/references]
3. Categorize [work items] by [type/impact/domain]
4. Determine required specialist [involvement/reviews/assignments]
```text

#### Phase 2: Routing

```text
5. Create [task/review/assignment] assignments:
   - Map each [item/file/aspect] to appropriate specialist
   - Ensure no overlap (one specialist per dimension)
   - Prioritize critical [tasks/reviews] ([list priorities])

6. Delegate to specialists in [parallel/sequential]:
   - Critical [tasks]: [Specialist A], [Specialist B], [Specialist C]
   - Core [tasks]: [Specialist D], [Specialist E], [Specialist F]
   - Specialized [tasks]: [Specialist G], [Specialist H]
   - Domain [tasks]: [Specialist I], [Specialist J]
```text

#### Phase 3: Consolidation

```text
7. Collect [output/feedback/results] from all specialists
8. Identify contradictions or conflicts
9. Consolidate into coherent [report/plan/deliverable]
10. Escalate unresolved conflicts if needed
```text

#### Phase 4: Reporting

```text
11. Generate comprehensive [summary/report/output]
12. Categorize findings by [severity/priority/type] (critical, major, minor)
13. Provide actionable recommendations
14. Track [completion/sign-off] status
```text

### Routing Rules (Prevents Overlap)

#### By [Primary Categorization Dimension]

<!-- Example: By File Extension, By Task Type, By Domain Area -->

| [Category] | Primary Specialist | Additional Specialists |
| --- | --- | --- |
| [Category A] | [Specialist Name] | [Specialist Name] (if needed) |
| [Category B] | [Specialist Name] | - |
| [Category C] | [Specialist Name] | [Specialist Name], [Specialist Name] |

<!-- Add 5-10 routing rules -->

#### By [Secondary Categorization Dimension]

<!-- Example: By Change Type, By Impact Level, By Complexity -->

| [Dimension Value] | Specialist(s) |
| --- | --- |
| [Value A] | [Specialist A] + [Specialist B] |
| [Value B] | [Specialist C] |
| [Value C] | [Specialist D] |

<!-- Add 5-10 routing rules -->

#### By [Tertiary Categorization Dimension]

<!-- Example: By Impact Assessment, By Risk Level, By Dependencies -->

| [Level] | Additional [Actions/Reviews/Specialists] Required |
| --- | --- |
| [High] | [Specialist X] + [Specialist Y] |
| [Medium] | [Specialist Z] |
| [Low] | [Standard process] |

<!-- Add 3-5 routing rules -->

### Delegates To

<!-- List ALL specialist agents this orchestrator coordinates -->
<!-- Group them by category for clarity -->

#### [Category 1] Specialists

- [[Specialist A Name](./specialist-a.md)] - [Primary responsibility in one line]
- [[Specialist B Name](./specialist-b.md)] - [Primary responsibility in one line]
- [[Specialist C Name](./specialist-c.md)] - [Primary responsibility in one line]

#### [Category 2] Specialists

- [[Specialist D Name](./specialist-d.md)] - [Primary responsibility in one line]
- [[Specialist E Name](./specialist-e.md)] - [Primary responsibility in one line]

#### [Category 3] Specialists

- [[Specialist F Name](./specialist-f.md)] - [Primary responsibility in one line]
- [[Specialist G Name](./specialist-g.md)] - [Primary responsibility in one line]

<!-- Add all specialists, typically 5-15 for a Level 2 orchestrator -->

### Escalates To

- [[Higher-Level Orchestrator or Authority](./higher-authority.md)] when:
  - [Escalation trigger 1]
  - [Escalation trigger 2]
  - [Escalation trigger 3]

- [[Peer Orchestrator](./peer-orchestrator.md)] when:
  - [Coordination trigger 1]
  - [Coordination trigger 2]

### Coordinates With

- [[Peer Orchestrator A](./peer-a.md)] - [Coordination scenario]
- [[Peer Orchestrator B](./peer-b.md)] - [Coordination scenario]

### Example Scenarios

<!-- Provide 2-4 realistic scenarios showing routing decisions -->
<!-- Each scenario should demonstrate: -->
<!--   1. Input/trigger -->
<!--   2. Analysis process -->
<!--   3. Routing decisions with rationale -->
<!--   4. What gets routed to each specialist -->
<!--   5. What does NOT get routed (and why) -->

#### Example 1: [Scenario Title - Primary Use Case]

### Input

```text
[Describe the incoming work item]
[List relevant files/changes/context]
```text

### Analysis

- [What you observe about this work]
- [Key characteristics that drive routing]
- [Impact assessment]

### Routing

```text
✅ [Specialist A] → [What they handle and why]
✅ [Specialist B] → [What they handle and why]
✅ [Specialist C] → [What they handle and why]

❌ NOT [Specialist X] ([reason for exclusion])
❌ NOT [Specialist Y] ([reason for exclusion])
❌ NOT [Specialist Z] ([reason for exclusion])
```text

### Consolidation

- [How you combine specialist outputs]
- [How you handle conflicts]
- [Final deliverable]

#### Example 2: [Scenario Title - Edge Case or Complex Scenario]

### Input

```text
[Describe a more complex scenario]
[Show overlap potential or conflict]
```text

### Analysis

- [Multi-dimensional analysis]
- [Competing concerns]
- [Trade-offs to consider]

### Routing

```text
✅ [Specialist A] → [Specific dimension they cover]
✅ [Specialist B] → [Different dimension, no overlap]
✅ [Specialist C] → [Yet another dimension]
✅ [Specialist D] → [Cross-cutting concern with clear scope]

❌ NOT [Specialist X] ([why not needed])
```text

### Special Handling

- [Any special coordination needed]
- [Conflict resolution approach]
- [Escalation if needed]

#### Example 3: [Scenario Title - Minimal Involvement Case]

### Input

```text
[Simple, focused work item]
```text

### Analysis

- [Why this is straightforward]
- [Limited scope assessment]

### Routing

```text
✅ [Specialist A] → [Primary handler]
✅ [Specialist B] → [Only if this specific condition]

❌ NOT [Specialist C-Z] ([why extensive coordination not needed])
```text

### Overlap Prevention Strategy

#### Dimension-Based Routing

Each aspect of [work] is [handled/reviewed] along independent dimensions:

| Dimension | Specialist | What They [Handle/Review] |
| --- | --- | --- |
| **[Dimension 1]** | [Specialist Name] | [Specific scope and boundaries] |
| **[Dimension 2]** | [Specialist Name] | [Specific scope and boundaries] |
| **[Dimension 3]** | [Specialist Name] | [Specific scope and boundaries] |
| **[Dimension 4]** | [Specialist Name] | [Specific scope and boundaries] |
| **[Dimension 5]** | [Specialist Name] | [Specific scope and boundaries] |

<!-- Add 5-12 dimensions -->

**Rule**: Each [work item/file/aspect] is routed to exactly one specialist per dimension.

#### Conflict Resolution

When specialists disagree:

1. **[Conflict Type A]**: [How to resolve - priority rule or escalation]
1. **[Conflict Type B]**: [Resolution approach]
1. **[Conflict Type C]**: [When to escalate vs resolve]
1. **[Conflict Type D]**: [Default behavior or guiding principle]

Escalate to [Higher Authority] if [escalation criteria].

### Success Criteria

- [ ] All [work items] analyzed and categorized
- [ ] Appropriate specialists assigned to each [task/review/dimension]
- [ ] No overlapping [work/reviews] (one specialist per dimension per item)
- [ ] All critical aspects [handled/reviewed] ([list critical aspects])
- [ ] Specialist [output/feedback] collected and consolidated
- [ ] Conflicts identified and resolved or escalated
- [ ] Comprehensive [report/summary/deliverable] generated
- [ ] Actionable recommendations provided

### Tools & Resources

- **Primary Language**: N/A (coordinator role)
- **[Domain-Specific Tools]**: [List relevant tools - e.g., Pre-commit hooks, GitHub Actions, Static analyzers]
- **Automation**: [What can be automated]
- **Documentation**: [Key reference docs]

### Constraints

- Must route [tasks/reviews] to prevent overlap
- Cannot override specialist decisions (only consolidate)
- Must escalate [specific conflicts] rather than resolve unilaterally
- [Work] must be timely (coordinate parallel [tasks/reviews/work])
- [Any domain-specific constraints]

### Skills to Use

<!-- List common skill invocations for this orchestrator -->

- `[skill_name]` - [When to use this skill]
- `[skill_name]` - [When to use this skill]
- `[skill_name]` - [When to use this skill]
- `[skill_name]` - [When to use this skill]

---

## Customization Instructions

### Step 1: Define Your Domain

Answer these questions to customize this template:

1. **What does this orchestrator coordinate?** (e.g., code reviews, testing, deployments)
1. **How many specialists does it coordinate?** (typically 5-15 for Level 2)
1. **What are the main categorization dimensions?** (e.g., file type, domain area, complexity)
1. **What triggers this orchestrator?** (e.g., PR creation, issue assignment, schedule)
1. **What does it produce?** (e.g., consolidated report, deployment plan, test strategy)

### Step 2: Identify Your Specialists

List all specialists this orchestrator coordinates:

| Specialist Name | Primary Responsibility | When to Route |
| --- | --- | --- |
| [Name] | [Responsibility] | [Trigger condition] |
| [Name] | [Responsibility] | [Trigger condition] |
| ... | ... | ... |

Group specialists into 3-5 logical categories.

### Step 3: Define Routing Rules

Create routing tables for:

1. **Primary dimension** (e.g., file extension, task type, domain)
1. **Secondary dimension** (e.g., change type, complexity, impact)
1. **Tertiary dimension** (e.g., risk level, priority, dependencies)

Ensure rules prevent overlap by assigning clear, non-overlapping scopes.

### Step 4: Write Example Scenarios

For each scenario, show:

1. **Input**: What triggers the orchestrator
1. **Analysis**: How you assess the situation
1. **Routing**: Which specialists get involved (and which don't)
1. **Rationale**: Why each routing decision was made
1. **Consolidation**: How specialist outputs combine

Include:

- **Primary scenario**: Most common case
- **Complex scenario**: Multiple specialists, potential conflicts
- **Simple scenario**: Minimal coordination needed

### Step 5: Define Success Criteria

What does "done" look like for this orchestrator?

- [ ] Measurable completion criteria
- [ ] Quality gates
- [ ] Deliverable checklist
- [ ] Handoff requirements

### Step 6: Test Your Configuration

```bash
# 1. Save customized config to .claude/agents/
cp agents/templates/level-2-orchestrator.md .claude/agents/[your-orchestrator].md

# 2. Replace all placeholders with actual values
# 3. Remove template instructions and comments
# 4. Test invocation in Claude Code

# 5. Verify behavior
# - Orchestrator analyzes work correctly
# - Routing decisions are logical and non-overlapping
# - Specialists receive clear, scoped assignments
# - Consolidation produces useful output
```text

---

## Example: How Code Review Orchestrator Uses This Template

### Answers to Step 1 questions

1. **What does it coordinate?** Code reviews across 13 specialist dimensions
1. **How many specialists?** 13 review specialists
1. **Main dimensions?** File extension, change type, impact level
1. **Trigger?** PR creation or cleanup phase
1. **Produces?** Consolidated review report with severity-categorized findings

### Routing dimensions

- **Primary**: File extension (.mojo → Mojo Language Specialist, .md → Documentation Specialist)
- **Secondary**: Change type (ML algorithm → Algorithm Specialist, security → Security Specialist)
- **Tertiary**: Impact (critical path → Performance + Safety, breaking changes → Architecture)

**Specialists** (13 total, grouped into 5 categories):

- Core (3): Implementation, Test, Documentation
- Security & Safety (2): Security, Safety
- Language & Performance (2): Mojo Language, Performance
- Domain (3): Algorithm, Data Engineering, Architecture
- Research (3): Paper, Research, Dependency

**Key innovation**: Dimension-based routing prevents overlap by assigning each specialist a unique dimension
(correctness, security, performance, etc.)

---

## Template Checklist

Before using your customized orchestrator:

- [ ] All `[PLACEHOLDERS]` replaced with actual values
- [ ] YAML frontmatter configured with correct name, description, tools
- [ ] Routing rules defined for at least 3 dimensions
- [ ] All specialists listed with clear responsibilities
- [ ] 2-4 example scenarios written showing routing decisions
- [ ] Overlap prevention strategy clearly defined
- [ ] Conflict resolution rules established
- [ ] Success criteria measurable and complete
- [ ] Escalation triggers identified
- [ ] Template instructions and comments removed

---

## See Also

- [Level 1 Chief Architect](../chief-architect.md) - Top-level orchestrator
- [Code Review Orchestrator](../.claude/agents/code-review-orchestrator.md) - Reference implementation
- [Agent Hierarchy](../hierarchy.md) - Complete agent structure
- [Delegation Rules](../delegation-rules.md) - Coordination patterns
- [Level 3 Specialist Template](level-3-specialist.md) - Template for specialists

---

*Level 2 Orchestrators coordinate 5-15 specialists to accomplish complex, multi-dimensional tasks without overlap or gaps.*
