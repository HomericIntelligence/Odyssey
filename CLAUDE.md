# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ML Odyssey is a Mojo-based AI research platform for reproducing classic research papers. The project uses a
comprehensive 4-level hierarchical planning structure with automated GitHub issue creation.

**Current Status**: Planning phase - repository structure and GitHub issues are being
established before implementation begins.

## ⚠️ CRITICAL RULES - READ FIRST

### 🚫 NEVER Push Directly to Main

**The `main` branch is protected. ALL changes MUST go through a pull request.**

❌ **ABSOLUTELY PROHIBITED:**

```bash
git checkout main
git add <files>
git commit -m "changes"
git push origin main  # ❌ BLOCKED - Will be rejected by GitHub
```

**Why this is prohibited:**

- Bypasses code review and CI checks
- Can break production immediately
- Violates GitHub branch protection rules
- Makes it impossible to track changes properly

✅ **CORRECT WORKFLOW (Always Use PRs):**

```bash
# 1. Create feature branch
git checkout -b <issue-number>-description

# 2. Make changes and commit
git add <files>
git commit -m "type(scope): description"

# 3. Push feature branch
git push -u origin <issue-number>-description

# 4. Create pull request
gh pr create \
  --title "Brief description" \
  --body "Closes #<issue-number>" \
  --label "appropriate-label"

# 5. Enable auto-merge
gh pr merge --auto --rebase
```

**Emergency Situations:**

- Even for critical CI fixes, CREATE A PR
- Even for one-line changes, CREATE A PR
- Even if you're fixing your own mistake, CREATE A PR
- NO EXCEPTIONS - Always use the PR workflow

**See Also:**

- PR Best Practices: [PR Workflow](/.claude/shared/pr-workflow.md)

## Quick Links

### Core Guidelines

- [Mojo Syntax & Patterns](/.claude/shared/mojo-guidelines.md)
- [Mojo Anti-Patterns](/.claude/shared/mojo-anti-patterns.md) - 64+ failure patterns
- [PR Workflow](/.claude/shared/pr-workflow.md)
- [GitHub Issue Workflow](/.claude/shared/github-issue-workflow.md)
- [Common Constraints](/.claude/shared/common-constraints.md)
- [Documentation Rules](/.claude/shared/documentation-rules.md)
- [Error Handling](/.claude/shared/error-handling.md)

### Agent System & Skills

- [Agent Hierarchy](/agents/hierarchy.md) - 6-level hierarchy
- [Agent Configurations](/.claude/agents/) - 44 agents
- [Skills Directory](/.claude/skills/) - 82+ capabilities

## Working with Agents

This project uses a hierarchical agent system for all development work. **Always use agents** as the primary
method for completing tasks.

### Agent Hierarchy

See [agents/hierarchy.md](agents/hierarchy.md) for the complete agent hierarchy including:

- 6-level hierarchy (L0 Chief Architect → L5 Junior Engineers)
- Model assignments (Opus, Sonnet, Haiku)
- All 44 agents with roles and responsibilities

### Key Agent Principles

1. **Always start with orchestrators** for new section work
1. **All outputs** must be posted as comments on the GitHub issue
1. **Link all PRs** to issues using `gh pr create --issue <number>` or "Closes #123" in description
1. **Minimal changes only** - smallest change that solves the problem
1. **No scope creep** - focus only on issue requirements
1. **Reply to each review comment** with `✅ Fixed - [brief description]`
1. **Delegate to skills** - Use "Use the X skill to..." pattern for automation

### Skill Delegation Patterns

Agents delegate to skills for automation using five standard patterns:

**Pattern 1: Direct Delegation** - Agent needs specific automation

```markdown
Use the `skill-name` skill to [action]:
- **Invoke when**: [trigger condition]
- **The skill handles**: [specific automation]
```

**Pattern 2: Conditional Delegation** - Agent decides based on conditions

```markdown
If [condition]:
  - Use the `skill-name` skill to [action]
Otherwise:
  - [alternative approach]
```

**Pattern 3: Multi-Skill Workflow** - Agent orchestrates multiple skills

```markdown
To accomplish [goal]:
1. Use the `skill-1` skill to [step 1]
2. Use the `skill-2` skill to [step 2]
3. Review results and [decision]
```

**Pattern 4: Skill Selection** - Orchestrator chooses skill based on analysis

```markdown
Analyze [context]:
- If [scenario A]: Use `skill-A`
- If [scenario B]: Use `skill-B`
```

**Pattern 5: Background vs Foreground** - Distinguishing automatic vs explicit invocation

```markdown
Background automation: `run-precommit` (runs automatically)
Foreground tasks: `gh-create-pr-linked` (invoke explicitly)
```

**Available Skills** (82 total across 11 categories):

- **GitHub**: gh-review-pr, gh-fix-pr-feedback, gh-create-pr-linked, gh-check-ci-status,
  gh-implement-issue, gh-reply-review-comment, gh-get-review-comments, gh-batch-merge-by-labels,
  verify-pr-ready
- **Worktree**: worktree-create, worktree-cleanup, worktree-switch, worktree-sync
- **Phase Workflow**: phase-plan-generate, phase-test-tdd, phase-implement, phase-package,
  phase-cleanup
- **Mojo**: mojo-format, mojo-test-runner, mojo-build-package, mojo-simd-optimize,
  mojo-memory-check, mojo-type-safety, mojo-lint-syntax, validate-mojo-patterns,
  check-memory-safety, analyze-simd-usage
- **Agent System**: agent-validate-config, agent-test-delegation, agent-run-orchestrator,
  agent-coverage-check, agent-hierarchy-diagram
- **Documentation**: doc-generate-adr, doc-issue-readme, doc-validate-markdown,
  doc-update-blog
- **CI/CD**: run-precommit, validate-workflow, fix-ci-failures, install-workflow,
  analyze-ci-failure-logs, build-run-local
- **Quality**: quality-run-linters, quality-fix-formatting, quality-security-scan,
  quality-coverage-report, quality-complexity-check
- **Testing & Analysis**: test-diff-analyzer, extract-test-failures, generate-fix-suggestions,
  track-implementation-progress
- **Review**: review-pr-changes, create-review-checklist

See `.claude/skills/` for complete implementations. Skills use YAML frontmatter with `mcp_fallback` for MCP integration.

### Key Development Principles

1. KISS - *K*eep *I*t *S*imple *S*tupid -> Don't add complexity when a simpler solution works
1. YAGNI - *Y*ou *A*in't *G*onna *N*eed *I*t -> Don't add things until they are required
1. TDD - *T*est *D*riven *D*evelopment -> Write tests to drive the implementation
1. DRY - *D*on't *R*epeat *Y*ourself -> Don't duplicate functionality, data structures, or algorithms
1. SOLID - *S**O**L**I**D* ->
  . Single Responsibility
  . Open-Closed
  . Liskov Substitution
  . Interface Segregation
  . Dependency Inversion
1. Modularity - Develop independent modules through well defined interfaces
1. POLA - *P*rinciple *O*f *L*east *A*stonishment - Create intuitive and predictable interfaces to not surprise users

Relevant links:

- [Core Principles of Software Development](<https://softjourn.com/insights/core-principles-of-software-development>)
- [7 Common Programming Principles](<https://www.geeksforgeeks.org/blogs/7-common-programming-principles-that-every-developer-must-follow/>)
- [Software Development Principles](<https://coderower.com/blogs/software-development-principles-software-engineering>)
- [Clean Coding Principles](<https://www.pullchecklist.com/posts/clean-coding-principles>)

### Documentation Rules

- **Issue-specific outputs**: Post as comments on the GitHub issue using `gh issue comment <number>`
- **Developer documentation**: `/docs/dev/` (architectural decisions, design docs)
- **Team guides**: `/agents/` (quick start, hierarchy, templates)
- **Never duplicate** documentation across locations - link instead
- See `.claude/shared/github-issue-workflow.md` for GitHub issue read/write patterns

### Language Preference

#### Mojo First - With Pragmatic Exceptions

**Default to Mojo** for ALL ML/AI implementations:

- ✅ Neural network implementations (forward/backward passes, layers)
- ✅ Training loops and optimization algorithms
- ✅ Tensor operations and SIMD kernels
- ✅ Performance-critical data processing
- ✅ Type-safe model components
- ✅ Gradient computation and backpropagation
- ✅ Model inference engines

**Use Python for Automation** when technical limitations require it:

- ✅ Subprocess output capture (Mojo v0.26.1 limitation - cannot capture stdout/stderr)
- ✅ Regex-heavy text processing (no Mojo regex support in stdlib)
- ✅ GitHub API interaction via Python libraries (`gh` CLI, REST API)
- ⚠️ **MUST document justification** (see ADR-001 for header template)

**Rule of Thumb** (Decision Tree):

1. **ML/AI implementation?** → Mojo (required)
1. **Automation needing subprocess output?** → Python (allowed, document why)
1. **Automation needing regex?** → Python (allowed, document why)
1. **Interface with Python-only libraries?** → Python (allowed, document why)
1. **Everything else?** → Mojo (default)

### Why Mojo for ML/AI

- Performance: Faster for ML workloads
- Type safety: Catch errors at compile time
- Memory safety: Built-in ownership and borrow checking
- SIMD optimization: Parallel tensor operations
- Future-proof: Designed for AI/ML from the ground up

### Why Python for Automation

- Mojo's subprocess API lacks exit code access (causes silent failures)
- Regex support not production-ready (mojo-regex is alpha stage)
- Python is the right tool for automation - not a temporary workaround

**See**: [ADR-001: Language Selection for Tooling](docs/adr/ADR-001-language-selection-tooling.md)
for complete language selection strategy, technical evidence (test results), and justification
requirements

See `/agents/README.md` for complete agent documentation and `/agents/hierarchy.md` for visual hierarchy.

## Claude 4 & Claude Code Optimization

This section provides guidance on optimizing interactions with Claude 4 (Sonnet and Opus) and
Claude Code features including extended thinking, agent skills, sub-agents, hooks, and output
styles.

### Extended Thinking

**When to Use Extended Thinking**: Claude 4 models support extended thinking for complex
reasoning tasks. Use extended thinking when:

- Analyzing complex codebases or architectural decisions
- Debugging multi-layered issues with unclear root causes
- Planning multi-step refactoring or migrations
- Evaluating tradeoffs between multiple design approaches
- Reasoning about edge cases and failure modes

**When NOT to Use Extended Thinking**:

- Simple CRUD operations or boilerplate code
- Well-defined tasks with clear specifications
- Repetitive tasks (formatting, linting, etc.)
- Tasks with clear step-by-step instructions already provided

**Example - Extended Thinking for Architecture Analysis**:

```markdown
Task: Analyze the tradeoffs between implementing tensor operations as struct methods vs standalone
functions in Mojo.

Extended thinking helps here because:
- Multiple design patterns to evaluate (OOP vs functional)
- Mojo-specific ownership and lifetime considerations
- Performance implications (inlining, SIMD optimization)
- API ergonomics and consistency with stdlib
```

**Example - Skip Extended Thinking for Boilerplate**:

```markdown
Task: Add a new test file following the existing test pattern in tests/shared/core/test_tensor.mojo

Skip extended thinking because:
- Clear pattern already established
- Straightforward copy-paste-modify workflow
- No architectural decisions needed
```

### Thinking Budget Guidelines

Extended thinking consumes tokens. Use appropriate budgets based on task complexity:

| Task Type | Budget | Examples | Rationale |
|-----------|--------|----------|-----------|
| **Simple** | None | Fix typo | Mechanical changes |
| **Standard** | 5K-10K | Add test, function | Well-defined |
| **Complex** | 10K-20K | Restructure, migrate | Dependencies |
| **Architecture** | 20K-50K | Design pattern | Deep analysis |
| **System-wide** | 50K+ | CI failures | Root cause |

**Budget Conservation Tips**:

1. **Provide context upfront** - Include relevant file contents, error messages, and constraints
2. **Break down complex tasks** - Split large problems into smaller, focused subtasks
3. **Use examples** - Show expected patterns rather than describing them
4. **Reference existing code** - Point to similar implementations as templates

### Agent Skills vs Sub-Agents

**Decision Tree**: Choose between skills and sub-agents based on task characteristics:

```text
Is the task well-defined with predictable steps?
├─ YES → Use an Agent Skill
│   ├─ Is it a GitHub operation? → Use gh-* skills
│   ├─ Is it a Mojo operation? → Use mojo-* skills
│   ├─ Is it a CI/CD task? → Use ci-* skills
│   └─ Is it a documentation task? → Use doc-* skills
│
└─ NO → Use a Sub-Agent
    ├─ Does it require exploration/discovery? → Use sub-agent
    ├─ Does it need adaptive decision-making? → Use sub-agent
    ├─ Is the workflow dynamic/context-dependent? → Use sub-agent
    └─ Does it need extended thinking? → Use sub-agent
```

**Agent Skills** - Use for automation with predictable workflows:

- **Characteristics**: Declarative YAML, fixed steps, composable, fast
- **Best for**: GitHub API calls, running tests, formatting code, CI workflows
- **Examples**: `gh-create-pr-linked`, `mojo-format`, `run-precommit`

**Sub-Agents** - Use for tasks requiring reasoning and adaptation:

- **Characteristics**: Full Claude instance, extended thinking, exploratory, slower
- **Best for**: Architecture decisions, debugging, code review, complex refactoring
- **Examples**: Documentation Engineer, Implementation Specialist, Review Engineer

**Example - When to Use a Skill**:

```markdown
Task: Create PR linked to issue #2549, run pixi run pre-commit hooks, enable auto-merge

✅ Use Agent Skills:
1. Use `gh-create-pr-linked` skill (predictable GitHub API workflow)
2. Use `run-precommit` skill (fixed command sequence)
3. Use `gh-check-ci-status` skill (polling with clear success/failure states)

Why skills work: Every step is well-defined, no exploration needed
```

**Example - When to Use a Sub-Agent**:

```markdown
Task: Review PR #2549 and suggest improvements to new Claude 4 documentation

✅ Use Sub-Agent (Review Engineer):
- Needs to read and understand the new documentation
- Compare against Claude's official documentation
- Evaluate clarity, completeness, and accuracy
- Provide actionable feedback with examples

Why sub-agent needed: Requires comprehension, judgment, adaptive reasoning
```

**Hybrid Approach** - Sub-agents can delegate to skills:

```markdown
Sub-Agent: Documentation Engineer implementing issue #2549

Workflow:
1. [Sub-agent] Read Claude 4 docs, analyze requirements, draft section
2. [Sub-agent] Use `doc-validate-markdown` skill to check formatting
3. [Sub-agent] Use `gh-create-pr-linked` skill to create PR
4. [Sub-agent] Use `ci-check-status` skill to verify CI passes
```

### Hooks Best Practices

Hooks enable proactive automation and safety checks. Use hooks for guardrails and background tasks.

**Safety Hooks** - Prevent errors before they happen:

```yaml
# Example: Prevent direct pushes to main branch
- trigger: "on_git_push"
  condition: "branch == 'main' && !is_pr"
  action: "block"
  message: "Direct pushes to main are not allowed. Create a PR instead."

# Example: Enforce zero-warnings policy
- trigger: "on_mojo_compile"
  condition: "warnings_count > 0"
  action: "fail"
  message: "Mojo code must compile without warnings. Fix warnings before committing."

# Example: Require issue link in PR description
- trigger: "on_pr_create"
  condition: "!body.includes('Closes #')"
  action: "block"
  message: "PR must reference an issue: add 'Closes #<number>' to description."
```

**Automation Hooks** - Background tasks that run automatically:

```yaml
# Example: Auto-format Mojo code on save
- trigger: "on_file_save"
  condition: "file.endsWith('.mojo')"
  action: "run_skill"
  skill: "mojo-format"

# Example: Run pixi run pre-commit hooks before commit
- trigger: "on_git_commit"
  action: "run_skill"
  skill: "run-precommit"

# Example: Auto-assign reviewers based on changed files
- trigger: "on_pr_create"
  condition: "changed_files.includes('shared/core/')"
  action: "add_reviewers"
  reviewers: ["core-team"]
```

**Hook Design Principles**:

1. **Fail fast** - Catch errors early in the development cycle
2. **Clear messages** - Explain WHY the hook triggered and HOW to fix
3. **Strict enforcement** - NEVER use `--no-verify`. Fix hook failures instead of bypassing.
4. **Idempotent** - Hooks should be safe to run multiple times

**Common Hooks for ML Odyssey**:

| Hook Type | Trigger | Purpose | Implementation |
|-----------|---------|---------|----------------|
| **Safety** | compile | Zero-warnings | Fail on warnings |
| **Safety** | pr_create | Issue link | Block if missing |
| **Safety** | git_push | Block main | Fail if direct |
| **Automation** | file_save | Auto-format | Run pixi run mojo format |
| **Automation** | git_commit | Pre-commit | Execute hooks |
| **Automation** | pr_merge | Cleanup | Remove worktree |

See `.claude/shared/error-handling.md` for retry strategies and timeout handling in hooks.

### Output Style Guidelines

Consistent output styles improve clarity and actionability. Follow these guidelines for different contexts:

#### Code References

**DO**: Use repo-relative file paths with line numbers when referencing code:

```markdown
✅ GOOD: Updated CLAUDE.md:173-185

✅ GOOD: Modified ExTensor initialization in shared/core/extensor.mojo:45

❌ BAD: Updated CLAUDE.md (missing line numbers)

❌ BAD: Fixed the tensor file (too vague)
```

**DO**: Include relevant code snippets with context:

```markdown
✅ GOOD:
File: /home/user/ProjectOdyssey/shared/core/extensor.mojo:45-52

fn __init__(out self, shape: List[Int], dtype: DType):
    """Initialize tensor with given shape and dtype."""
    self._shape = shape^
    self._dtype = dtype
    var numel = 1
    for dim in shape:
        numel *= dim
    self._data = DTypePointer[dtype].alloc(numel)

❌ BAD: Changed the constructor
```

#### Issue and PR Formatting

**DO**: Use structured markdown with clear sections:

```markdown
✅ GOOD:

## Summary
Added comprehensive Claude 4 optimization guidance to CLAUDE.md

## Changes Made
- Added "Extended Thinking" section with when/when-not guidelines
- Added "Thinking Budget Guidelines" table with 5 task types
- Added "Agent Skills vs Sub-Agents" decision tree
- Added "Hooks Best Practices" with safety and automation examples
- Added "Output Style Guidelines" for code references and reviews

## Files Modified
- `/home/user/ProjectOdyssey/CLAUDE.md` (lines 173-500, added 327 lines)

## Verification
- [x] Markdown linting passes
- [x] All code examples use correct syntax
- [x] Cross-references point to existing sections
- [x] Integrated seamlessly with existing content

❌ BAD: Added some docs
```

**DO**: Link to related issues and PRs explicitly:

```markdown
✅ GOOD:
Related Issues:
- Closes #2549
- Related to #2548 (Markdown standards)
- Depends on #2544 (Agent hierarchy)

❌ BAD: Fixes the issue about Claude docs
```

#### Code Review Output

**DO**: Provide specific, actionable feedback with examples:

```markdown
✅ GOOD:

**Issue**: Inconsistent parameter naming in ExTensor methods

**Location**: `/home/user/ProjectOdyssey/shared/core/extensor.mojo:120-145`

**Problem**: Methods use both `mut self` and `self` inconsistently

**Recommendation**: Use implicit `read` (just `self`) for read-only methods:

# Current (line 120)
fn shape(mut self) -> List[Int]:  # ❌ mut not needed
    return self._shape

# Should be
fn shape(self) -> List[Int]:  # ✅ Implicit read
    return self._shape

**Impact**: Misleading API - suggests mutation when none occurs

❌ BAD: The shape method is wrong
```

**DO**: Prioritize feedback by severity:

```markdown
✅ GOOD:

### Critical (Must Fix Before Merge)
1. Memory leak in ExTensor.__del__() - data not freed
2. Missing bounds check in __getitem__() - potential segfault

### Important (Should Fix)
1. Inconsistent parameter naming (mut vs read)
2. Missing docstrings on public methods

### Nice to Have (Consider for Future)
1. Add SIMD optimization to fill() method
2. Consider caching numel() computation

❌ BAD: Here's 20 random issues in no particular order
```

#### Terminal Output

**DO**: Use structured formatting for command output:

```bash
$ mojo test tests/shared/core/test_tensor.mojo
Testing: /home/user/ProjectOdyssey/tests/shared/core/test_tensor.mojo
  test_tensor_creation ... PASSED
  test_tensor_indexing ... PASSED
  test_tensor_reshape ... PASSED
All tests passed (3/3)
```

**DO**: Include error context when reporting failures:

```bash
$ mojo build shared/core/extensor.mojo
error: ExTensor.mojo:145:16: cannot transfer ownership of
  non-copyable type
    return self._data
           ^
```

### Tool Use Optimization

Efficient tool use reduces latency and token consumption. Follow these patterns:

#### Parallel Tool Calls

**DO**: Make independent tool calls in parallel:

```python
# ✅ GOOD - Parallel reads
read_file_1 = Read("/path/to/file1.mojo")
read_file_2 = Read("/path/to/file2.mojo")
read_file_3 = Read("/path/to/file3.mojo")
# All three reads happen concurrently

# ❌ BAD - Sequential reads
read_file_1 = Read("/path/to/file1.mojo")
# Wait for result...
read_file_2 = Read("/path/to/file2.mojo")
# Wait for result...
read_file_3 = Read("/path/to/file3.mojo")
```

**DO**: Group related grep searches:

```python
# ✅ GOOD - Parallel greps
grep_functions = Grep(pattern="fn .*", glob="*.mojo")
grep_structs = Grep(pattern="struct .*", glob="*.mojo")
grep_tests = Grep(pattern="test_.*", glob="test_*.mojo")
# All searches run in parallel

# ❌ BAD - Sequential greps with waiting
grep_functions = Grep(pattern="fn .*", glob="*.mojo")
# Process results, then...
grep_structs = Grep(pattern="struct .*", glob="*.mojo")
```

#### Bash Command Patterns

**DO**: Use absolute paths in bash commands (cwd resets between calls):

```bash
# ✅ GOOD - Absolute paths
cd /home/user/ProjectOdyssey && pixi run mojo test tests/shared/core/test_tensor.mojo

# ❌ BAD - Relative paths (cwd not guaranteed)
cd ProjectOdyssey && pixi run mojo test tests/shared/core/test_tensor.mojo
```

**DO**: Combine related commands with && for atomicity:

```bash
# ✅ GOOD - Atomic operation
cd /home/user/ProjectOdyssey && \
  git checkout -b 2549-claude-md && \
  git add CLAUDE.md && \
  git commit -m "docs: add Claude 4 optimization guidance"

# ❌ BAD - Multiple separate bash calls (cwd resets)
cd /home/user/ProjectOdyssey
git checkout -b 2549-claude-md  # Might run in different directory!
git add CLAUDE.md
```

**DO**: Capture output explicitly when needed:

```bash
# ✅ GOOD - Capture and parse output
cd /home/user/ProjectOdyssey && \
  pixi run mojo test tests/ 2>&1 | tee test_output.log && \
  grep -c PASSED test_output.log

# ❌ BAD - Output lost between calls
cd /home/user/ProjectOdyssey && pixi run mojo test tests/
# Output is gone, can't analyze it
```

#### Tool Selection

Use the right tool for the job:

| Task | Tool | Rationale |
|------|------|-----------|
| Read file | Read | Fast, includes lines |
| Search pattern | Grep | Optimized regex |
| Find files | Glob | Fast discovery |
| Run commands | Bash | Execute shell |
| Edit lines | Edit | Precise replace |
| Write file | Write | Create/overwrite |

**DO**: Use the most specific tool:

```python
# ✅ GOOD - Use Glob to find files, then Read them
files = Glob(pattern="**/test_*.mojo")
for file in files:
    content = Read(file)

# ❌ BAD - Use Bash for file discovery
result = Bash("find . -name 'test_*.mojo'")
# Now have to parse shell output
```

### Agentic Loop Patterns

Claude Code supports iterative exploration through agentic loops. Use this pattern for complex tasks:

#### Exploration → Planning → Execution

**Phase 1: Exploration** - Gather context and understand the problem:

```markdown
Exploration Tasks:
1. Read relevant documentation (CLAUDE.md, agent files, related issues)
2. Search for existing patterns (grep for similar implementations)
3. Identify constraints and requirements (compiler version, API patterns)
4. Review recent changes (git log, PR history)

Tools: Read, Grep, Glob, Bash (git log)
Output: Problem understanding, constraints, existing patterns
```

**Phase 2: Planning** - Design the solution:

```markdown
Planning Tasks:
1. Break down the problem into subtasks
2. Identify files to modify and create
3. Design interfaces and data structures
4. Plan verification steps (tests, linting, CI)

Tools: Extended thinking, structured reasoning
Output: Implementation plan, task breakdown, success criteria
```

**Phase 3: Execution** - Implement the solution:

```markdown
Execution Tasks:
1. Make code changes (Edit, Write)
2. Run verification (Bash: mojo test, pre-commit)
3. Fix errors iteratively (Read error output → Edit → Rerun)
4. Create PR and link to issue (gh-create-pr-linked skill)

Tools: Edit, Write, Bash, agent skills
Output: Working implementation, passing tests, merged PR
```

**Example - Agentic Loop for Issue #2549**:

```markdown
Iteration 1: Exploration
- Read CLAUDE.md to understand structure (Read tool)
- Search for existing Claude guidance (Grep "Claude|agent|skill")
- Review Issue #2549 requirements (gh issue view 2549)
- Read Claude 4 docs links (external URLs)
Output: Understand where to insert new section, what to include

Iteration 2: Planning
- Design section structure (6 subsections based on requirements)
- Identify insertion point (after "Working with Agents", before "Delegation")
- Plan examples (extended thinking, skills vs sub-agents, hooks)
- Define success criteria (markdown linting, cross-references, integration)
Output: Section outline, examples drafted, verification plan

Iteration 3: Execution
- Insert new section using Edit tool
- Add cross-references to existing sections
- Run markdown linting (just pre-commit-all)
- Fix any linting errors
- Create PR with "Closes #2549"
Output: Updated CLAUDE.md, passing linting, PR created

Iteration 4: Verification & Refinement
- Review generated content for accuracy
- Verify all examples use correct syntax
- Check cross-references point to real sections
- Confirm integration with existing content
- Enable auto-merge if CI passes
Output: PR ready for merge, issue resolved
```

**Key Principles**:

1. **Iterate, don't perfect upfront** - Start with exploration, refine through execution
2. **Fail fast** - Run verification early and often
3. **Learn from errors** - Each failure provides information for the next iteration
4. **Checkpoint progress** - Commit working states, even if incomplete
5. **Adapt the plan** - If exploration reveals new constraints, update the plan

**When to Use Agentic Loops**:

- ✅ Complex refactoring across multiple files
- ✅ Debugging issues with unclear root causes
- ✅ Implementing features with design tradeoffs
- ✅ Exploring unfamiliar codebases

**When NOT to Use Agentic Loops**:

- ❌ Simple, well-defined tasks (use direct execution)
- ❌ Boilerplate code generation (use templates/examples)
- ❌ Mechanical changes (formatting, renaming)

### Cross-References

- **Agent Skills**: See available skills in [Skill Delegation Patterns](#skill-delegation-patterns)
- **Sub-Agents**: See agent hierarchy in `/agents/hierarchy.md`
- **Hooks**: See error handling patterns in `.claude/shared/error-handling.md`
- **Extended Thinking**: Referenced in [Key Agent Principles](#key-agent-principles)
- **GitHub Workflow**: See `.claude/shared/github-issue-workflow.md` for issue/PR patterns
- **Tool Use**: See tool documentation in Claude Code docs

### Further Reading

- [Claude 4 Best Practices](<https://platform.claude.com/docs/en/build-with-claude/overview>)
- [Agent Skills Best Practices](<https://platform.claude.com/docs/en/agents-and-tools/claude-for-sheets>)
- [Sub-Agents Guide](<https://code.claude.com/docs/en/sub-agents>)
- [Output Styles](<https://code.claude.com/docs/en/output-styles>)
- [Hooks Guide](<https://code.claude.com/docs/en/hooks-guide>)

## Delegation to Agent Hub

.claude/ is the centralized location for agentic descriptions and SKILLs. Sub-agents reference
`.claude/agents/*.md` and `.claude/skills/*.md` for roles, capabilities, and prod fix learnings.

### Shared Reference Files

All agents and skills reference these shared files to avoid duplication:

| File | Purpose |
|------|---------|
| `.claude/shared/common-constraints.md` | Minimal changes principle, scope discipline |
| `.claude/shared/documentation-rules.md` | Output locations, before-starting checklist |
| `.claude/shared/pr-workflow.md` | PR creation, verification, review responses |
| `.claude/shared/mojo-guidelines.md` | Mojo v0.26.1+ syntax, parameter conventions |
| `.claude/shared/mojo-anti-patterns.md` | 64+ test failure patterns from PRs |
| `.claude/shared/error-handling.md` | Retry strategy, timeout handling, escalation |

### MCP Integration

**DEPRECATED**: GitHub MCP integration is being removed. Use `gh` CLI directly for all
GitHub operations to avoid token overhead.

Skills with `mcp_fallback` in YAML frontmatter will be updated to use direct CLI calls only.

### Mojo Development Guidelines

**Quick Reference**: See [mojo-guidelines.md](/.claude/shared/mojo-guidelines.md) for v0.26.1+ syntax

**Critical Patterns**:

- **Constructors**: Use `out self` (not `mut self`)
- **Mutating methods**: Use `mut self`
- **Ownership transfer**: Use `^` operator for List/Dict/String
- **List initialization**: Use literals `[1, 2, 3]` not `List[Int](1, 2, 3)`
- **Matrix multiply**: Use `matmul(A, B)` — never call `A.__matmul__(B)` directly at call sites

**Common Mistakes**: See [mojo-anti-patterns.md](/.claude/shared/mojo-anti-patterns.md) for 64+ failure patterns

**Compiler as Truth**: When uncertain, test with `mojo build` - the compiler is authoritative

## Environment Setup

This project uses Pixi for environment management:

```bash
# Pixi is already configured - dependencies are in pixi.toml
# Mojo is the primary language target for future implementations
```

## Common Commands

### Justfile Build System

The project uses [Just](<https://just.systems/>) as a unified command runner for local development and CI/CD consistency.

#### Quick Reference

```bash
# Show all available recipes
just --list

# Get help
just help

# Development commands
just build                  # Build project in debug mode
just test                   # Run all tests
just test-mojo             # Run only Mojo tests
just format                # Format all files

# CI-specific commands (match GitHub Actions)
just validate           # Full validation (build + test)
just build              # Build shared package
just package           # Compile package (validation only)
just test-mojo          # Run all Mojo tests
just test-group PATH PATTERN  # Run specific test group
just pre-commit               # Run pre-commit hooks
just pre-commit-all               # Run pre-commit hooks on all files

# Training and inference
just train                 # Train LeNet-5 with defaults
just train lenet5 fp16 20  # Train with FP16, 20 epochs
just infer lenet5 ./weights  # Run inference

# Docker management
just docker-up             # Start development environment
just docker-down           # Stop environment
just shell          # Open shell in container
```

### Docker Registry (GHCR)

The project publishes Docker images to GitHub Container Registry (GHCR).

#### Available Images

| Image | Purpose | Size |
|-------|---------|------|
| `ghcr.io/homericintelligence/projectodyssey:main` | Runtime with Mojo/tests | ~2GB |
| `ghcr.io/homericintelligence/projectodyssey:main-ci` | CI with pre-commit | ~2.5GB |
| `ghcr.io/homericintelligence/projectodyssey:main-prod` | Minimal production | ~1.5GB |

#### Pull and Run

```bash
# Pull latest runtime image
docker pull ghcr.io/homericintelligence/projectodyssey:main

# Run tests
docker run --rm ghcr.io/homericintelligence/projectodyssey:main

# Interactive shell
docker run -it --rm ghcr.io/homericintelligence/projectodyssey:main bash

# Mount local code for development
docker run -it --rm -v $(pwd):/app ghcr.io/homericintelligence/projectodyssey:main bash
```

#### Build Locally

```bash
# Build CI image locally
just docker-build-ci runtime

# Build all targets
just docker-build-ci-all

# Push to GHCR (requires authentication)
docker login ghcr.io
just docker-push runtime
```

### Why Use Justfile?

1. **Consistency**: Same commands work locally and in CI
2. **Simplicity**: Easy-to-read recipes vs complex bash scripts
3. **Documentation**: Self-documenting with `just --list`
4. **Reliability**: Ensures identical flags between local dev and CI

### CI Integration

GitHub Actions workflows use justfile recipes to ensure consistency:

```yaml
# Example from comprehensive-tests.yml
- name: Run test group
  run: just test-group "tests/shared/core" "test_*.mojo"

# Example from build-validation.yml
- name: Build package
  run: just build
```

This ensures developers can run `just validate` locally to reproduce CI results exactly.

**See**: `justfile` for complete recipe list and implementation details.

### Development Workflows

**Pull Requests**: See [pr-workflow.md](/.claude/shared/pr-workflow.md)

- Creating PRs with `gh pr create --body "Closes #<number>"`
- Responding to review comments (use GitHub API, not `gh pr comment`)
- Post-merge cleanup (worktree removal, branch deletion)

**GitHub Issues**: See [github-issue-workflow.md](/.claude/shared/github-issue-workflow.md)

- Read context: `gh issue view <number> --comments`
- Post updates: `gh issue comment <number> --body "..."`

**Git Workflow**: Feature branch → PR → Auto-merge (never push to main)

### Agent Testing

Agent configurations are automatically validated in CI on all PRs. Run tests locally before committing:

```bash
# Validate agent YAML frontmatter and configuration
python3 tests/agents/validate_configs.py .claude/agents/

# Test agent discovery and loading
python3 tests/agents/test_loading.py .claude/agents/

# Test delegation patterns
python3 tests/agents/test_delegation.py .claude/agents/

# Test workflow integration
python3 tests/agents/test_integration.py .claude/agents/

# Test Mojo-specific patterns
python3 tests/agents/test_mojo_patterns.py .claude/agents/

# Run all tests
for script in tests/agents/test_*.py tests/agents/validate_*.py; do
    python3 "$script" .claude/agents/
done
```

### Test Coverage

- Configuration validation (YAML frontmatter, required fields, tool specifications)
- Agent discovery and loading (hierarchy coverage, activation patterns)
- Delegation patterns (chain validation, escalation paths)
- Workflow integration (5-phase coverage, parallel execution)
- Mojo patterns (fn vs def, struct vs class, SIMD, memory management)

**CI Integration**: The `.github/workflows/test-agents.yml` workflow runs these tests
automatically on all PRs affecting agent configurations.

### Pre-commit Hooks

Pre-commit hooks automatically check code quality before commits. The hooks include `pixi run mojo format`
for Mojo code and markdown linting for documentation.

```bash
# Install pre-commit hooks (one-time setup)
pixi run pre-commit install

# Run hooks manually on all files
just pre-commit-all

# Run hooks manually on staged files only
just precommit

# NEVER skip hooks with --no-verify
# If a hook fails, fix the code instead
# If a specific hook is broken, use SKIP=hook-id:
SKIP=trailing-whitespace git commit -m "message"
```

### Pre-Commit Hook Policy - STRICT ENFORCEMENT

`--no-verify` is **ABSOLUTELY PROHIBITED**. No exceptions.

**If hooks fail:**

1. Read the error message to understand what failed
2. Fix the code to pass the hook
3. Re-run `just precommit` to verify fixes
4. Commit again

**Valid alternatives to --no-verify:**

- Fix the code (preferred)
- Use `SKIP=hook-id` for specific broken hooks (must document reason)
- Disable the hook in `.pre-commit-config.yaml` if permanently problematic

**Invalid alternatives:**

- ❌ `git commit --no-verify`
- ❌ `git commit -n`
- ❌ Any command that bypasses all hooks

### Configured Hooks

- `mojo format` - Auto-format Mojo code (`.mojo`, `.🔥` files)
- `markdownlint-cli2` - Lint markdown files
- `trailing-whitespace` - Remove trailing whitespace
- `end-of-file-fixer` - Ensure files end with newline
- `check-yaml` - Validate YAML syntax
- `check-added-large-files` - Prevent large files (max 1MB)
- `mixed-line-ending` - Fix mixed line endings

**CI Enforcement**: The `.github/workflows/pre-commit.yml` workflow runs these checks on
all PRs and pushes to `main`.

**See:** [Git Commit Policy](.claude/shared/git-commit-policy.md) for complete enforcement rules.

## Repository Architecture

### Project Structure

```text
ProjectOdyssey/
├── agents/                      # Team documentation
│   ├── README.md                # Quick start guide
│   ├── hierarchy.md             # Visual hierarchy diagram and complete agent specifications
│   ├── delegation-rules.md      # Coordination patterns
│   └── templates/               # Agent configuration templates
├── notes/
│   └── review/                  # Comprehensive specs & decisions
│       ├── agent-architecture-review.md
│       ├── skills-design.md
│       └── orchestration-patterns.md
├── scripts/                     # Python automation scripts
├── logs/                        # Execution logs and state files
└── .clinerules                 # Claude Code conventions
```

### Planning Hierarchy

**4 Levels** (managed through GitHub issues):

1. **Section** (e.g., 01-foundation) - Major area of work
1. **Subsection** (e.g., 01-directory-structure) - Logical grouping
1. **Component** (e.g., 01-create-papers-dir) - Specific deliverable
1. **Subcomponent** (e.g., 01-create-base-dir) - Atomic task

All planning documentation is tracked in GitHub issues. Use `gh issue view <number>` to read plans.

### Documentation Organization

The repository uses three separate locations for documentation to avoid duplication:

#### 1. Team Documentation (`/agents/`)

**Purpose**: Quick start guides, visual references, and templates for team onboarding.

### Contents

- Quick start guides (README.md)
- Visual diagrams (hierarchy.md)
- Quick reference cards (delegation-rules.md)
- Configuration templates (templates/)

**When to Use**: Creating new documentation for team onboarding or quick reference.

#### 2. Developer Documentation (`/docs/dev/`)

**Purpose**: Detailed architectural decisions, comprehensive specifications, and design documents.

### Contents

- Mojo patterns and error handling (mojo-test-failure-patterns.md)
- Skills architecture (skills-design.md, skills-architecture.md)
- Orchestration patterns (orchestration-patterns.md)
- Backward pass catalog (backward-pass-catalog.md)

**When to Use**: Writing detailed specifications, architectural decisions, or comprehensive guides.

#### 3. Issue-Specific Documentation (GitHub Issue Comments)

**Purpose**: Implementation notes, findings, and decisions specific to a single GitHub issue.

**Location**: Post directly to the GitHub issue as comments using `gh issue comment`.

**Reading Issue Context**:

```bash
# Get issue details and body
gh issue view <number>

# Get all comments (implementation history)
gh issue view <number> --comments

# Get structured data
gh issue view <number> --json title,body,comments,labels,state
```

**Writing to Issues**:

```bash
# Post implementation notes
gh issue comment <number> --body "$(cat <<'EOF'
## Implementation Notes

### Summary
[What was implemented]

### Files Changed
- path/to/file.mojo

### Verification
- [x] Tests pass
EOF
)"
```

### Important Rules

- ✅ DO: Post issue-specific findings and decisions as comments
- ✅ DO: Link to comprehensive docs in `/agents/` and `/docs/dev/`
- ✅ DO: Reference related issues with `#<number>` format
- ❌ DON'T: Duplicate comprehensive documentation
- ❌ DON'T: Create local files for issue tracking

### 5-Phase Development Workflow

Every component follows a hierarchical workflow with clear dependencies:

**Workflow**: Plan → [Test | Implementation | Package] → Cleanup

1. **Plan** - Design and documentation (MUST complete first)
1. **Test** - Write tests following TDD (parallel after Plan)
1. **Implementation** - Build the functionality (parallel after Plan)
1. **Package** - Create distributable packages (parallel after Plan)
   - Build binary packages (`.mojopkg` files for Mojo modules)
   - Create distribution archives (`.tar.gz`, `.zip` for tooling/docs)
   - Configure package metadata and installation procedures
   - Add components to existing packages
   - Test package installation in clean environments
   - Create CI/CD packaging workflows
   - **NOT just documenting** - must create actual distributable artifacts
1. **Cleanup** - Refactor and finalize (runs after parallel phases complete)

### Key Points

- Plan phase produces specifications for all other phases
- Test/Implementation/Package can run in parallel after Plan completes
- Cleanup collects issues discovered during the parallel phases
- Each phase has a separate GitHub issue with detailed instructions

## Testing Strategy

ML Odyssey uses a comprehensive two-tier testing strategy designed for fast PR validation
and thorough weekly integration testing.

### Two-Tier Testing Architecture

**Tier 1: Layerwise Unit Tests** (Run on every PR)

- Fast, deterministic tests using FP-representable special values (0.0, 0.5, 1.0, 1.5, -1.0, -0.5)
- Tests each layer independently (forward AND backward passes)
- Validates analytical gradients against numerical gradients
- Small tensor sizes to prevent timeouts
- Runtime: ~12 minutes across all 7 models
- Location: `tests/models/test_<model>_layers.mojo`

**Tier 2: End-to-End Integration Tests** (Run weekly only)

- Full model validation with real datasets (EMNIST, CIFAR-10)
- Tests complete forward-backward pipeline
- Validates training convergence (5 epochs, ≥20% loss decrease)
- Runtime: ~20 minutes per model
- Schedule: Weekly on Sundays at 3 AM UTC
- Location: `tests/models/test_<model>_e2e.mojo`

### Test Coverage by Model

All 7 models have comprehensive test coverage:

| Model | Layers | Layerwise Tests | E2E Tests | Runtime (Layerwise) |
|-------|--------|-----------------|-----------|---------------------|
| LeNet-5 | 12 ops | 25 tests | 7 tests | ~45-55s |
| AlexNet | 15 ops | 42 tests | 9 tests | <60s |
| VGG-16 | 25 ops (13 conv) | 16 tests* | 10 tests | ~90s |
| ResNet-18 | Residual blocks | 12 tests | 9 tests | ~90s |
| MobileNetV1 | Depthwise sep. | 26 tests | 15 tests | ~90s |
| GoogLeNet | Inception modules | 18 tests | 15 tests | ~90s |

\* Heavy deduplication (13 conv → 5 unique tests)
\*\* Extreme deduplication (58 conv → 14 unique tests, 88% reduction)

### Special FP-Representable Values

Layerwise tests use special values that are exactly representable across all dtypes:

- `0.0`, `0.5`, `1.0`, `1.5` - Positive values for forward pass testing
- `-1.0`, `-0.5` - Negative values for ReLU gradient testing
- Seeded random tensors - For gradient checking reproducibility

These values work identically in FP4, FP8, BF8, FP16, FP32, BFloat16, Int8, ensuring
consistent behavior across precision levels.

### Gradient Checking

All parametric layers (Conv, Linear, BatchNorm) validate backward passes:

- Compares analytical gradients to numerical gradients (finite differences)
- Uses seeded random tensors for reproducibility (seed=42)
- Epsilon=1e-5, tolerance=1e-2 for float32
- Small tensor sizes (8×8 for conv) to prevent timeout

### Test Organization

```text
tests/
├── models/
│   ├── test_lenet5_layers.mojo      # Layerwise unit tests
│   ├── test_lenet5_e2e.mojo         # E2E integration tests
│   ├── test_alexnet_layers.mojo
│   ├── test_alexnet_e2e.mojo
│   └── ... (7 models total)
├── shared/
│   └── testing/
│       ├── special_values.mojo      # FP-representable test values
│       ├── layer_testers.mojo       # Reusable layer testing patterns
│       ├── dtype_utils.mojo         # DType iteration utilities
│       └── gradient_checker.mojo    # Numerical gradient validation
```

### CI/CD Workflows

**PR CI** (`.github/workflows/comprehensive-tests.yml`):

- Runs layerwise tests for all 7 models
- Target runtime: < 12 minutes
- 21 parallel test groups
- No dataset downloads required

**Weekly E2E** (`.github/workflows/model-e2e-tests-weekly.yml`):

- Runs E2E tests for all 7 models
- Downloads EMNIST and CIFAR-10 datasets
- Generates weekly report with 365-day retention
- Schedule: Sundays at 3 AM UTC

### Running Tests Locally

```bash
# Run layerwise tests for a specific model
pixi run mojo test tests/models/test_lenet5_layers.mojo

# Run E2E tests (requires datasets)
pixi run mojo test tests/models/test_lenet5_e2e.mojo

# Run all tests for a model
pixi run mojo test tests/models/test_lenet5_*.mojo

# Run all layerwise tests
pixi run mojo test tests/models/test_*_layers.mojo
```

See [Testing Strategy Guide](docs/dev/testing-strategy.md) for comprehensive documentation.

## GitHub Issue Structure

All planning is done through GitHub issues with clear structure:

### Issue Body Format

```markdown
## Objective
Brief description (2-3 sentences)

## Deliverables
- [ ] Deliverable 1
- [ ] Deliverable 2

## Success Criteria
- [ ] Criterion 1
- [ ] Criterion 2

## Dependencies
- Depends on #<parent-issue>
- Related: #<sibling-issue>

## Notes
Additional context
```

### Issue Labels

- `planning` - Design phase
- `testing` - Test development
- `implementation` - Code implementation
- `packaging` - Distribution packages
- `cleanup` - Finalization

### Linking Issues

- Reference in body: `Depends on #123`
- Reference in commits: `Implements #123`
- Close via PR: `Closes #123`

## Working with GitHub Issues

All planning and documentation is managed through GitHub issues directly.

### Creating New Work Items

1. Create a GitHub issue with clear description and acceptance criteria
2. Use appropriate labels (planning, testing, implementation, packaging, cleanup)
3. Link related issues using `#<number>` references

### Tracking Implementation

1. Read issue context: `gh issue view <number> --comments`
2. Post progress updates as issue comments
3. Link PRs to issues: `gh pr create --body "Closes #<number>"`

### Documentation Workflow

1. **Read context first**: `gh issue view <number> --comments`
2. **Post updates**: `gh issue comment <number> --body "..."`
3. **Reference in commits**: "Implements #<number>" or "Closes #<number>"

See `.claude/shared/github-issue-workflow.md` for complete workflow patterns.

### File Locations

- **Scripts**: `scripts/*.py`
- **Logs**: `logs/*.log`
- **Tracked Docs**: `docs/dev/`, `agents/` (reference these in commits)
- **Issue Docs**: GitHub issue comments (not local files)

## Git Workflow

### Branch Naming

- `main` - Production branch (protected, requires PR)
- `<issue-number>-<description>` - Feature/fix branches (e.g., `1928-consolidate-test-assertions`)

### Development Workflow

**IMPORTANT:** The `main` branch is protected. All changes must go through a pull request.

#### Creating a PR (Standard Workflow)

1. **Create a feature branch:**

   ```bash
   git checkout -b <issue-number>-<description>
   ```

1. **Make your changes and commit:**

   ```bash
   git add <files>
   git commit -m "$(cat <<'EOF'
   type(scope): Brief description

   Detailed explanation of changes.

   🤖 Generated with [Claude Code](<https://claude.com/claude-code>)

   Co-Authored-By: Claude <noreply@anthropic.com>
   EOF
   )"
   ```

1. **Push the feature branch:**

   ```bash
   git push -u origin <branch-name>
   ```

1. **Create pull request:**

   ```bash
   gh pr create \
     --title "[Type] Brief description" \
     --body "Closes #<issue-number>" \
     --label "appropriate-label"
   ```

1. **Enable auto-merge:**

   ```bash
   gh pr merge --auto --rebase
   ```

   **Always enable auto-merge** so PRs merge automatically once CI passes.

### 🚫 Never Push Directly to Main

**⚠️ CRITICAL:** See [CRITICAL RULES section](#️-critical-rules---read-first) at the top of this document.

**This rule has NO EXCEPTIONS - not even for emergencies.**

❌ **ABSOLUTELY PROHIBITED:**

```bash
git checkout main
git add <files>
git commit -m "changes"
git push origin main  # Will be rejected - main is protected
```

**Even These Are WRONG:**

```bash
# ❌ WRONG - Bypassing with force push
git push --force origin main

# ❌ WRONG - Directly committing on main
git checkout main && git commit -am "quick fix"

# ❌ WRONG - Emergency fix without PR
git checkout main && git cherry-pick <commit> && git push
```

✅ **CORRECT - ALWAYS Use Pull Requests:**

```bash
# 1. Create feature branch from main
git checkout main
git pull origin main
git checkout -b <issue-number>-description

# 2. Make changes and commit
git add <files>
git commit -m "type(scope): description"

# 3. Push feature branch
git push -u origin <issue-number>-description

# 4. Create and auto-merge PR
gh pr create \
  --title "Brief description" \
  --body "Closes #<issue-number>" \
  --label "appropriate-label"
gh pr merge --auto --rebase
```

**Why This Rule Exists:**

1. **Code Review** - All changes must be reviewed
2. **CI Validation** - All changes must pass automated tests
3. **Audit Trail** - Track what changed, why, and who approved it
4. **Prevent Breakage** - Catch issues before they hit production
5. **Branch Protection** - GitHub enforces this rule automatically

**What If CI Is Already Broken?**

- Still create a PR to fix it
- PR description should explain the emergency
- Enable auto-merge so it merges immediately when CI passes
- Example: PR #2689 (cleanup) followed emergency fix commit 4446eba2

## Commit Message Format

Follow conventional commits:

```text
feat(section): Add new component
fix(scripts): Correct parsing issue
docs(readme): Update instructions
refactor(plans): Standardize to Template 1
```

### Worktree and PR Discipline

**One PR per Issue:**

- Each GitHub issue should have exactly ONE pull request
- Do not combine multiple issues into a single PR
- Branch naming: `<issue-number>-<description>`

**Worktree Directory:**

- Create all worktrees in the `worktrees/` subdirectory within the repo
- Naming convention: `<issue-number>-<description>`
- Example: `git worktree add worktrees/123-fix-bug 123-fix-bug`

**Post-Merge Cleanup:**

After a PR is merged/rebased to main:

1. Remove the worktree: `git worktree remove worktrees/<issue-number>-<description>`
2. Delete local branch: `git branch -d <branch-name>`
3. Delete remote branch: `git push origin --delete <branch-name>`
4. Prune stale references: `git worktree prune`

## Labels

Standard labels automatically created by scripts:

- `planning` - Design phase (light purple: #d4c5f9)
- `documentation` - Documentation work (blue: #0075ca)
- `testing` - Testing phase (yellow: #fbca04)
- `tdd` - Test-driven development (yellow: #fbca04)
- `implementation` - Implementation phase (dark blue: #1d76db)
- `packaging` - Integration/packaging (light green: #c2e0c6)
- `integration` - Integration tasks (light green: #c2e0c6)
- `cleanup` - Cleanup/finalization (red: #d93f0b)

## Python Coding Standards

```python

#!/usr/bin/env python3

"""
Script description

Usage:
    python scripts/script_name.py [options]
"""

# Standard imports first

import sys
import re
from pathlib import Path
from typing import List, Dict, Optional

def function_name(param: str) -> bool:
    """Clear docstring with purpose, params, returns."""
    pass
```

### Requirements

- Python 3.7+
- Type hints required for all functions
- Clear docstrings for public functions
- Comprehensive error handling
- Logging for important operations

## Markdown Standards

All markdown files must follow these standards to pass `markdownlint-cli2` linting:

### Code Blocks (MD031, MD040)

**Rule**: Fenced code blocks must be:

1. Surrounded by blank lines (before and after)
1. Have a language specified on the opening backticks
1. Don't put anything on the closing backticks

### Language Examples

- Python: ` ```python `
- Bash: ` ```bash `
- Text/plain: ` ```text `
- Mojo: ` ```mojo `
- YAML: ` ```yaml `
- JSON: ` ```json `
- Markdown: ` ```markdown `

### Lists (MD032)

**Rule**: Lists must be surrounded by blank lines (before and after)

### Correct

```markdown
Some text before.

- Item 1
- Item 2
- Item 3

Some text after.
```

### Incorrect

```markdown
Some text before.
- Item 1
- Item 2
Some text after.
```

### Headings (MD022)

**Rule**: Headings must be surrounded by blank lines (one blank line before and after)

### Correct

```markdown
Some content here.

## Section Heading

More content here.
```

### Incorrect

```markdown
Some content here.
## Section Heading
More content here.
```

### Line Length (MD013)

**Rule**: Lines should not exceed 120 characters

### Guidelines

- Break long lines at 120 characters
- Break at natural boundaries (clauses, lists, etc.)
- Code in code blocks is exempt
- URLs in links are exempt

### Example

```markdown
This is a very long sentence that exceeds the 120 character limit
and should be broken into multiple lines at a natural boundary point
for better readability.
```

### Best Practices

1. **Always add blank lines around code blocks and lists** - This is the #1 cause of linting failures
1. **Always specify language for code blocks** - Use appropriate language tags
1. **Check headings have surrounding blank lines** - Especially after subheadings
1. **Use reference-style links for long URLs** - Helps avoid line length issues

### Quick Checklist for New Content

Before committing markdown files:

- [ ] All code blocks have a language specified (` ```python ` not ` ``` `)
- [ ] All code blocks have blank lines before and after
- [ ] All lists have blank lines before and after
- [ ] All headings have blank lines before and after
- [ ] No lines exceed 120 characters
- [ ] File ends with newline (enforced by pre-commit)
- [ ] No trailing whitespace (enforced by pre-commit)

### Running Markdown Linting Locally

```bash
# Check specific file
pixi run npx markdownlint-cli2 path/to/file.md

# Check all markdown files
zust pre-commit-all

# View detailed errors
pixi run npx markdownlint-cli2 path/to/file.md 2>&1
```

## Debugging

### Check Logs

```bash
# View script logs
tail -100 logs/*.log

# View specific log
cat logs/<script>_*.log
```

## Troubleshooting

### GitHub CLI Issues

```bash
# Check authentication
gh auth status

# If missing scopes, refresh authentication
gh auth refresh -h github.com
```

### Issue Access Problems

- Check GitHub CLI auth: `gh auth status`
- Verify repository access: `gh repo view`
- Test issue access: `gh issue list`

### Script Errors

- Verify Python version: `python3 --version` (requires 3.7+)
- Check file permissions
- Review error logs in `logs/` directory

## Important Files

- `.clinerules` - Comprehensive Claude Code conventions
- `docs/dev/` - Developer documentation (Mojo patterns, skills architecture)
- `scripts/README.md` - Complete scripts documentation
- `README.md` - Main project documentation
- `.claude/shared/github-issue-workflow.md` - GitHub issue read/write patterns
