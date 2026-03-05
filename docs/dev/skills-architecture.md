# ML Odyssey Claude Skills Architecture

## Executive Summary

This document defines a comprehensive suite of 35+ Claude skills designed to automate and
simplify agent workflows in the ML Odyssey project. Skills are organized into functional
categories aligned with the project's 5-phase development workflow (Plan → Test →
Implementation → Package → Cleanup) and support the hierarchical agent system from Level 0
(Chief Architect) through Level 5 (Junior Engineers).

Skills are designed to be:

- **Composable**: Work together seamlessly
- **Focused**: Single responsibility per skill
- **Efficient**: Under 500 lines with progressive disclosure
- **Robust**: Comprehensive error handling
- **Model-invoked**: Claude decides when to use them

## Skill Categories

### 1. GitHub Integration Skills (7 skills)

Handle all GitHub operations including PR management, issue tracking, and review workflows.

### 2. Worktree Management Skills (4 skills)

Manage git worktrees for parallel development across multiple features.

### 3. Phase Workflow Skills (5 skills)

Automate the 5-phase development workflow (Plan, Test, Implementation, Package, Cleanup).

### 4. Mojo Development Skills (6 skills)

Support Mojo-specific development including SIMD optimization and memory safety.

### 5. Agent System Skills (5 skills)

Validate, test, and orchestrate the hierarchical agent system.

### 6. Documentation Skills (4 skills)

Generate and maintain various documentation types including ADRs and blog posts.

### 7. CI/CD Skills (4 skills)

Handle continuous integration and deployment operations.

### 8. Plan Management Skills (3 skills)

Manage hierarchical planning structure and GitHub issue generation.

### 9. Code Quality Skills (5 skills)

Ensure code quality through linting, formatting, and security scanning.

## Skill Specifications

### 1. GitHub Integration Skills

#### gh-review-pr

**Priority**: High
**Description**: Comprehensively review a pull request including code changes, CI status, and test coverage.
**Purpose**: Automate thorough PR reviews following project standards.

### Use Cases

- Review incoming PRs
- Check for adherence to coding standards
- Validate CI passes
- Ensure proper issue linking

**Tool Requirements**: `Read`, `Bash`, `Grep`

### File Structure

```text
gh-review-pr/
├── SKILL.md (main skill with review checklist)
├── reference.md (review standards and criteria)
├── scripts/
│   ├── check_pr_status.sh
│   ├── analyze_changes.sh
│   └── validate_tests.sh
└── templates/
    └── review_comment.md
```text

#### gh-get-review-comments

**Priority**: High
**Description**: Retrieve all open review comments from a PR using the correct GitHub API.
**Purpose**: Collect feedback that needs to be addressed.

### Use Cases

- Get all unresolved comments
- Filter comments by reviewer
- Check comment status

**Tool Requirements**: `Bash`

### File Structure

```text
gh-get-review-comments/
├── SKILL.md (API interaction logic)
└── scripts/
    └── fetch_comments.sh
```text

#### gh-reply-review-comment

**Priority**: High
**Description**: Reply to PR review comments using the correct API (not gh pr comment).
**Purpose**: Properly respond to inline code review feedback.

### Use Cases

- Reply to specific review comments
- Mark comments as resolved
- Provide fix confirmations

**Tool Requirements**: `Bash`

### File Structure

```text
gh-reply-review-comment/
├── SKILL.md (correct API usage)
├── reference.md (API documentation)
└── scripts/
    └── reply_to_comment.sh
```text

#### gh-fix-pr-feedback

**Priority**: High
**Description**: Automatically fix issues identified in PR reviews.
**Purpose**: Streamline the feedback incorporation process.

### Use Cases

- Apply requested changes
- Update code based on feedback
- Push fixes and reply to comments

**Tool Requirements**: `Read`, `Write`, `Bash`, `Grep`

### File Structure

```text
gh-fix-pr-feedback/
├── SKILL.md (feedback processing logic)
├── scripts/
│   ├── apply_fixes.sh
│   └── verify_fixes.sh
└── templates/
    └── fix_confirmation.md
```text

#### gh-create-pr-linked

**Priority**: High
**Description**: Create a pull request with proper issue linking using --issue flag or "Closes #" pattern.
**Purpose**: Ensure all PRs are properly linked to GitHub issues.

### Use Cases

- Create new PR from branch
- Link to existing issue
- Verify linkage in GitHub

**Tool Requirements**: `Bash`

### File Structure

```text
gh-create-pr-linked/
├── SKILL.md (PR creation with validation)
└── scripts/
    └── create_linked_pr.sh
```text

#### gh-implement-issue

**Priority**: High
**Description**: Coordinate sub-agents to implement a GitHub issue end-to-end.
**Purpose**: Automate complete issue implementation workflow.

### Use Cases

- Parse issue requirements
- Delegate to appropriate agents
- Track implementation progress

**Tool Requirements**: `Read`, `Bash`

### File Structure

```text
gh-implement-issue/
├── SKILL.md (orchestration logic)
├── reference.md (delegation patterns)
└── scripts/
    ├── parse_issue.sh
    └── track_progress.sh
```text

#### gh-check-ci-status

**Priority**: Medium
**Description**: Check CI status for a PR or commit.
**Purpose**: Monitor build and test status.

### Use Cases

- Check if CI passes
- Get failure details
- Monitor CI progress

**Tool Requirements**: `Bash`

### File Structure

```text
gh-check-ci-status/
├── SKILL.md (CI monitoring)
└── scripts/
    └── check_ci.sh
```text

### 2. Worktree Management Skills

#### worktree-create

**Priority**: High
**Description**: Create a new git worktree for feature development.
**Purpose**: Enable parallel development across multiple features.

### Use Cases

- Create worktree for new feature
- Set up worktree from issue branch
- Configure worktree environment

**Tool Requirements**: `Bash`

### File Structure

```text
worktree-create/
├── SKILL.md (worktree creation logic)
└── scripts/
    └── create_worktree.sh
```text

#### worktree-cleanup

**Priority**: High
**Description**: Clean up unused worktrees and their associated branches.
**Purpose**: Maintain clean development environment.

### Use Cases

- Remove merged worktrees
- Clean orphaned worktrees
- Prune worktree list

**Tool Requirements**: `Bash`

### File Structure

```text
worktree-cleanup/
├── SKILL.md (cleanup logic)
└── scripts/
    └── cleanup_worktrees.sh
```text

#### worktree-switch

**Priority**: Medium
**Description**: Switch between existing worktrees.
**Purpose**: Navigate between parallel development efforts.

### Use Cases

- Switch to different feature
- List available worktrees
- Check worktree status

**Tool Requirements**: `Bash`

### File Structure

```text
worktree-switch/
├── SKILL.md (switching logic)
└── scripts/
    └── switch_worktree.sh
```text

#### worktree-sync

**Priority**: Medium
**Description**: Sync worktree with upstream changes.
**Purpose**: Keep worktrees up to date.

### Use Cases

- Pull upstream changes
- Rebase worktree
- Resolve conflicts

**Tool Requirements**: `Bash`

### File Structure

```text
worktree-sync/
├── SKILL.md (sync logic)
└── scripts/
    └── sync_worktree.sh
```text

### 3. Phase Workflow Skills

#### phase-plan-generate

**Priority**: High
**Description**: Generate comprehensive plan documentation for a component.
**Purpose**: Automate planning phase of 5-phase workflow.

### Use Cases

- Create plan.md files
- Generate specifications
- Update parent/child links

**Tool Requirements**: `Read`, `Write`, `Glob`

### File Structure

```text
phase-plan-generate/
├── SKILL.md (plan generation)
├── reference.md (plan template)
└── templates/
    ├── plan.md
    └── github_issue.md
```text

#### phase-test-tdd

**Priority**: High
**Description**: Generate and run tests following TDD practices.
**Purpose**: Automate test-driven development workflow.

### Use Cases

- Generate test scaffolding
- Run test suites
- Check coverage

**Tool Requirements**: `Read`, `Write`, `Bash`

### File Structure

```text
phase-test-tdd/
├── SKILL.md (TDD workflow)
├── reference.md (TDD practices)
└── scripts/
    ├── generate_tests.sh
    └── run_tests.sh
```text

#### phase-implement

**Priority**: High
**Description**: Coordinate implementation phase across multiple engineers.
**Purpose**: Orchestrate parallel implementation work.

### Use Cases

- Delegate implementation tasks
- Track progress
- Integrate components

**Tool Requirements**: `Read`, `Write`, `Bash`

### File Structure

```text
phase-implement/
├── SKILL.md (implementation orchestration)
└── scripts/
    └── track_implementation.sh
```text

#### phase-package

**Priority**: High
**Description**: Create distributable packages (.mojopkg, .tar.gz, etc.).
**Purpose**: Automate packaging phase.

### Use Cases

- Build Mojo packages
- Create distribution archives
- Test package installation

**Tool Requirements**: `Bash`, `Read`

### File Structure

```text
phase-package/
├── SKILL.md (packaging logic)
├── reference.md (package formats)
└── scripts/
    ├── build_mojopkg.sh
    └── create_archive.sh
```text

#### phase-cleanup

**Priority**: Medium
**Description**: Refactor and finalize code after implementation.
**Purpose**: Ensure code quality and consistency.

### Use Cases

- Apply refactoring
- Update documentation
- Clean up technical debt

**Tool Requirements**: `Read`, `Write`, `Grep`

### File Structure

```text
phase-cleanup/
├── SKILL.md (cleanup workflow)
└── scripts/
    └── refactor_code.sh
```text

### 4. Mojo Development Skills

#### mojo-format

**Priority**: High
**Description**: Format Mojo code according to project standards.
**Purpose**: Ensure consistent Mojo code formatting.

### Use Cases

- Format .mojo files
- Format .🔥 files
- Check formatting compliance

**Tool Requirements**: `Read`, `Write`, `Bash`

### File Structure

```text
mojo-format/
├── SKILL.md (formatting logic)
└── scripts/
    └── format_mojo.sh
```text

#### mojo-test-runner

**Priority**: High
**Description**: Run Mojo test suites and parse results.
**Purpose**: Execute and analyze Mojo tests.

### Use Cases

- Run unit tests
- Parse test output
- Generate test reports

**Tool Requirements**: `Bash`, `Read`

### File Structure

```text
mojo-test-runner/
├── SKILL.md (test execution)
└── scripts/
    └── run_mojo_tests.sh
```text

#### mojo-build-package

**Priority**: High
**Description**: Build .mojopkg packages from Mojo modules.
**Purpose**: Create distributable Mojo packages.

### Use Cases

- Compile Mojo modules
- Create package manifest
- Build .mojopkg file

**Tool Requirements**: `Bash`, `Read`

### File Structure

```text
mojo-build-package/
├── SKILL.md (package building)
├── reference.md (package structure)
└── scripts/
    └── build_package.sh
```text

#### mojo-simd-optimize

**Priority**: Medium
**Description**: Apply SIMD optimizations to Mojo code.
**Purpose**: Optimize performance-critical code paths.

### Use Cases

- Vectorize loops
- Apply SIMD operations
- Optimize tensor operations

**Tool Requirements**: `Read`, `Write`

### File Structure

```text
mojo-simd-optimize/
├── SKILL.md (optimization patterns)
├── reference.md (SIMD guidelines)
└── templates/
    └── simd_kernel.mojo
```text

#### mojo-memory-check

**Priority**: Medium
**Description**: Verify memory safety in Mojo code.
**Purpose**: Ensure proper memory management.

### Use Cases

- Check ownership patterns
- Verify borrow checking
- Detect memory leaks

**Tool Requirements**: `Read`, `Grep`

### File Structure

```text
mojo-memory-check/
├── SKILL.md (memory checking)
└── scripts/
    └── check_memory.sh
```text

#### mojo-type-safety

**Priority**: Medium
**Description**: Validate type safety and type hints in Mojo code.
**Purpose**: Ensure type correctness.

### Use Cases

- Check type annotations
- Validate generic types
- Verify trait implementations

**Tool Requirements**: `Read`, `Grep`

### File Structure

```text
mojo-type-safety/
├── SKILL.md (type checking)
└── scripts/
    └── check_types.sh
```text

### 5. Agent System Skills

#### agent-validate-config

**Priority**: High
**Description**: Validate agent YAML configurations and frontmatter.
**Purpose**: Ensure agent configurations are correct.

### Use Cases

- Check YAML syntax
- Validate required fields
- Verify tool specifications

**Tool Requirements**: `Read`, `Bash`

### File Structure

```text
agent-validate-config/
├── SKILL.md (validation logic)
└── scripts/
    └── validate_agents.py
```text

#### agent-test-delegation

**Priority**: High
**Description**: Test agent delegation patterns and chains.
**Purpose**: Verify delegation works correctly.

### Use Cases

- Test delegation chains
- Verify escalation paths
- Check skip-level rules

**Tool Requirements**: `Read`, `Bash`

### File Structure

```text
agent-test-delegation/
├── SKILL.md (delegation testing)
└── scripts/
    └── test_delegation.py
```text

#### agent-run-orchestrator

**Priority**: High
**Description**: Run a specific orchestrator as a sub-agent.
**Purpose**: Delegate work to section orchestrators.

### Use Cases

- Run Foundation Orchestrator
- Run CI/CD Orchestrator
- Coordinate multiple orchestrators

**Tool Requirements**: `Bash`

### File Structure

```text
agent-run-orchestrator/
├── SKILL.md (orchestrator execution)
├── reference.md (orchestrator list)
└── scripts/
    └── run_orchestrator.sh
```text

#### agent-hierarchy-diagram

**Priority**: Low
**Description**: Generate visual hierarchy diagrams for agents.
**Purpose**: Visualize agent relationships.

### Use Cases

- Create hierarchy diagrams
- Update visual documentation
- Generate team charts

**Tool Requirements**: `Read`, `Write`

### File Structure

```text
agent-hierarchy-diagram/
├── SKILL.md (diagram generation)
└── scripts/
    └── generate_diagram.py
```text

#### agent-coverage-check

**Priority**: Low
**Description**: Check agent coverage across workflow phases.
**Purpose**: Ensure all phases have agent support.

### Use Cases

- Verify phase coverage
- Identify gaps
- Generate coverage reports

**Tool Requirements**: `Read`, `Grep`

### File Structure

```text
agent-coverage-check/
├── SKILL.md (coverage analysis)
└── scripts/
    └── check_coverage.py
```text

### 6. Documentation Skills

#### doc-update-blog

**Priority**: Medium
**Description**: Update blog posts to current format and standards.
**Purpose**: Maintain consistent blog documentation.

### Use Cases

- Update formatting
- Fix broken links
- Add metadata

**Tool Requirements**: `Read`, `Write`

### File Structure

```text
doc-update-blog/
├── SKILL.md (blog update logic)
└── templates/
    └── blog_template.md
```text

#### doc-generate-adr

**Priority**: High
**Description**: Create Architectural Decision Records.
**Purpose**: Document architectural decisions.

### Use Cases

- Create new ADR
- Update ADR status
- Link related ADRs

**Tool Requirements**: `Write`

### File Structure

```text
doc-generate-adr/
├── SKILL.md (ADR generation)
└── templates/
    └── adr_template.md
```text

#### doc-issue-readme

**Priority**: High
**Description**: Generate issue-specific README documentation.
**Purpose**: Create focused issue documentation.

### Use Cases

- Initialize issue directory
- Update issue status
- Link to shared docs

**Tool Requirements**: `Write`, `Read`

### File Structure

```text
doc-issue-readme/
├── SKILL.md (issue doc generation)
└── templates/
    └── issue_readme.md
```text

#### doc-validate-markdown

**Priority**: Medium
**Description**: Validate markdown against linting rules.
**Purpose**: Ensure markdown quality.

### Use Cases

- Check markdown syntax
- Validate link formatting
- Fix common issues

**Tool Requirements**: `Read`, `Bash`

### File Structure

```text
doc-validate-markdown/
├── SKILL.md (markdown validation)
└── scripts/
    └── validate_markdown.sh
```text

### 7. CI/CD Skills

#### run-precommit

**Priority**: High
**Description**: Run pre-commit hooks locally.
**Purpose**: Validate code before committing.

### Use Cases

- Run all hooks
- Run specific hooks
- Fix hook failures

**Tool Requirements**: `Bash`, `Read`, `Write`

### File Structure

```text
run-precommit/
├── SKILL.md (pre-commit execution)
└── scripts/
    └── run_precommit.sh
```text

#### validate-workflow

**Priority**: Medium
**Description**: Validate GitHub Actions workflow files.
**Purpose**: Ensure CI/CD workflows are correct.

### Use Cases

- Check workflow syntax
- Validate job dependencies
- Test workflow locally

**Tool Requirements**: `Read`, `Bash`

### File Structure

```text
validate-workflow/
├── SKILL.md (workflow validation)
└── scripts/
    └── validate_workflow.sh
```text

#### fix-ci-failures

**Priority**: High
**Description**: Diagnose and fix CI failures.
**Purpose**: Quickly resolve CI issues.

### Use Cases

- Analyze failure logs
- Identify root cause
- Apply fixes

**Tool Requirements**: `Read`, `Write`, `Bash`

### File Structure

```text
fix-ci-failures/
├── SKILL.md (failure diagnosis)
└── scripts/
    └── analyze_failures.sh
```text

#### install-workflow

**Priority**: Medium
**Description**: Create CI/CD packaging workflows.
**Purpose**: Automate package building in CI.

### Use Cases

- Generate workflow files
- Configure package jobs
- Set up artifact uploads

**Tool Requirements**: `Write`

### File Structure

```text
install-workflow/
├── SKILL.md (workflow generation)
└── templates/
    └── package_workflow.yml
```text

### 8. Code Quality Skills

#### quality-run-linters

**Priority**: High
**Description**: Run all configured linters on codebase.
**Purpose**: Ensure code quality standards.

### Use Cases

- Run Python linters
- Run Mojo linters
- Check markdown

**Tool Requirements**: `Bash`, `Read`

### File Structure

```text
quality-run-linters/
├── SKILL.md (linter execution)
└── scripts/
    └── run_all_linters.sh
```text

#### quality-fix-formatting

**Priority**: High
**Description**: Automatically fix formatting issues.
**Purpose**: Apply consistent formatting.

### Use Cases

- Fix Python formatting
- Fix Mojo formatting
- Fix markdown issues

**Tool Requirements**: `Read`, `Write`, `Bash`

### File Structure

```text
quality-fix-formatting/
├── SKILL.md (formatting fixes)
└── scripts/
    └── fix_formatting.sh
```text

#### quality-security-scan

**Priority**: Medium
**Description**: Run security vulnerability scans.
**Purpose**: Identify security issues.

### Use Cases

- Scan dependencies
- Check for vulnerabilities
- Generate security report

**Tool Requirements**: `Bash`, `Read`

### File Structure

```text
quality-security-scan/
├── SKILL.md (security scanning)
└── scripts/
    └── security_scan.sh
```text

#### quality-complexity-check

**Priority**: Low
**Description**: Analyze code complexity metrics.
**Purpose**: Identify complex code needing refactoring.

### Use Cases

- Calculate cyclomatic complexity
- Identify long functions
- Find deep nesting

**Tool Requirements**: `Read`, `Grep`

### File Structure

```text
quality-complexity-check/
├── SKILL.md (complexity analysis)
└── scripts/
    └── check_complexity.py
```text

#### quality-coverage-report

**Priority**: Medium
**Description**: Generate test coverage reports.
**Purpose**: Track test coverage metrics.

### Use Cases

- Run coverage analysis
- Generate HTML reports
- Identify untested code

**Tool Requirements**: `Bash`, `Read`

### File Structure

```text
quality-coverage-report/
├── SKILL.md (coverage reporting)
└── scripts/
    └── generate_coverage.sh
```text

## Priority Levels

### High Priority (Must Have) - 18 skills

These skills are essential for basic workflow automation:

- All GitHub integration skills (7)
- Core worktree skills (2)
- Phase workflow skills (4)
- Essential Mojo skills (3)
- Core agent skills (3)
- Critical quality skills (2)
- Essential documentation (2)

### Medium Priority (Should Have) - 12 skills

These skills enhance productivity:

- Additional worktree skills (2)
- Phase cleanup skill (1)
- Mojo optimization skills (3)
- Documentation skills (2)
- CI/CD skills (3)
- Plan management skills (2)
- Quality analysis (2)

### Low Priority (Nice to Have) - 5 skills

These skills provide additional capabilities:

- Agent visualization (2)
- Complex analysis (1)
- Advanced reporting (2)

## Implementation Roadmap

### Phase 1: Foundation (Week 1)

1. **GitHub Integration Core**
   - gh-review-pr
   - gh-get-review-comments
   - gh-reply-review-comment
   - gh-create-pr-linked

1. **Basic Workflow**
   - phase-plan-generate
   - phase-test-tdd
   - worktree-create

### Phase 2: Core Workflows (Week 2)

1. **Complete GitHub Suite**
   - gh-fix-pr-feedback
   - gh-implement-issue
   - gh-check-ci-status

1. **Mojo Essentials**
   - mojo-format
   - mojo-test-runner
   - mojo-build-package

1. **Agent Validation**
   - agent-validate-config
   - agent-test-delegation

### Phase 3: Advanced Features (Week 3)

1. **Phase Automation**
   - phase-implement
   - phase-package
   - phase-cleanup

1. **Documentation**
   - doc-generate-adr
   - doc-issue-readme

1. **Quality Tools**
   - quality-run-linters
   - quality-fix-formatting
   - run-precommit

### Phase 4: Optimization (Week 4)

1. **Mojo Advanced**
   - mojo-simd-optimize
   - mojo-memory-check
   - mojo-type-safety

1. **CI/CD Suite**
   - validate-workflow
   - fix-ci-failures
   - install-workflow

### Phase 5: Polish (Week 5)

1. **Remaining Skills**
   - All low priority skills
   - Additional quality tools
   - Visualization tools

## Integration Patterns

### Skill Composition Examples

#### Example 1: Complete PR Review and Fix

```text
1. gh-review-pr → Identify issues
2. gh-get-review-comments → Collect feedback
3. gh-fix-pr-feedback → Apply fixes
4. quality-fix-formatting → Clean up code
5. run-precommit → Validate changes
6. gh-reply-review-comment → Respond to reviewers
```text

#### Example 2: Implement New Feature

```text
1. gh-implement-issue → Parse requirements
2. worktree-create → Set up workspace
3. phase-plan-generate → Create plan
4. phase-test-tdd → Write tests
5. phase-implement → Build feature
6. mojo-format → Format code
7. mojo-test-runner → Run tests
8. phase-package → Create package
9. gh-create-pr-linked → Submit PR
```text

#### Example 3: Fix CI Failure

```text
1. gh-check-ci-status → Get failure details
2. fix-ci-failures → Diagnose issue
3. quality-fix-formatting → Fix formatting
4. run-precommit → Validate locally
5. worktree-sync → Update branch
6. validate-workflow → Check workflow
```text

### Skill Dependencies

```mermaid
graph TD
    A[gh-implement-issue] --> B[phase-plan-generate]
    B --> C[phase-test-tdd]
    B --> D[phase-implement]
    C --> E[mojo-test-runner]
    D --> F[mojo-format]
    D --> G[quality-run-linters]
    E --> H[quality-coverage-report]
    F --> I[gh-create-pr-linked]
    G --> I
    H --> I
    I --> J[gh-review-pr]
    J --> K[gh-get-review-comments]
    K --> L[gh-fix-pr-feedback]
    L --> M[gh-reply-review-comment]
```text

## Testing Strategy

### Skill Validation Levels

1. **Unit Testing**
   - Test individual skill functions
   - Validate input/output handling
   - Check error conditions

1. **Integration Testing**
   - Test skill combinations
   - Validate data flow between skills
   - Check composition patterns

1. **End-to-End Testing**
   - Test complete workflows
   - Validate real-world scenarios
   - Check performance metrics

### Test Coverage Requirements

- **High Priority Skills**: 90% coverage required
- **Medium Priority Skills**: 80% coverage required
- **Low Priority Skills**: 70% coverage required

### Testing Tools

```bash
# Validate skill configuration
python3 tests/skills/validate_skills.py .claude/skills/

# Test skill loading
python3 tests/skills/test_loading.py .claude/skills/

# Test skill execution
python3 tests/skills/test_execution.py .claude/skills/

# Test skill composition
python3 tests/skills/test_composition.py .claude/skills/
```text

## Best Practices

### Skill Design Guidelines

1. **Single Responsibility**
   - Each skill does ONE thing well
   - Clear, focused purpose
   - No feature creep

1. **Progressive Disclosure**
   - Essential info in SKILL.md (< 500 lines)
   - Details in reference.md
   - Scripts in separate files

1. **Error Handling**
   - Explicit error cases
   - Clear error messages
   - Recovery strategies

1. **Documentation**
   - Clear description in frontmatter
   - Usage examples
   - Error scenarios

1. **Tool Efficiency**
   - Only request needed tools
   - Minimize tool calls
   - Cache results when possible

### Naming Conventions

- **Skill Names**: lowercase-with-hyphens
- **Categories**: domain-action format (e.g., gh-review-pr)
- **Scripts**: snake_case.sh or snake_case.py
- **Templates**: template_type.md

### File Structure Standards

```text
skill-name/
├── SKILL.md          # Required: Main skill definition
├── reference.md      # Optional: Detailed documentation
├── scripts/          # Optional: Executable scripts
│   ├── main.sh       # Primary script
│   └── helper.py     # Supporting scripts
└── templates/        # Optional: File templates
    └── template.md   # Template files
```text

## Success Metrics

### Adoption Metrics

- **Usage Rate**: % of agents using skills
- **Invocation Frequency**: Skills used per task
- **Success Rate**: % of successful skill executions

### Quality Metrics

- **Error Rate**: Failures per 100 invocations
- **Performance**: Average execution time
- **Token Usage**: Tokens consumed per skill

### Impact Metrics

- **Time Saved**: Hours saved per week
- **Quality Improvement**: Reduction in bugs/issues
- **Developer Satisfaction**: Survey feedback

## Notes

- Skills are model-invoked based on description matching
- Skills share the conversation's context window
- Progressive disclosure is critical for token efficiency
- Skills can be composed for complex workflows
- Both project (.claude/skills/) and personal (~/.claude/skills/) locations supported
- Skills should align with existing agent hierarchy
- Focus on automating repetitive tasks
- Prioritize high-impact workflows

## References

- [Claude Code Skills Documentation](https://code.claude.com/docs/en/skills)
- [Agent Hierarchy](https://github.com/HomericIntelligence/ProjectOdyssey/blob/main/agents/hierarchy.md)
- [Orchestration Patterns](./orchestration-patterns.md)
- [CLAUDE.md](https://github.com/HomericIntelligence/ProjectOdyssey/blob/main/CLAUDE.md)
- [5-Phase Workflow](./phases.md)
