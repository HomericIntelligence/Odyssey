#!/usr/bin/env python3
"""
Update all agent files with Claude 4-specific sections.

This script adds standardized sections to all 44 agent files:
1. Thinking Guidance
2. Output Preferences
3. Delegation Patterns
4. Sub-Agent Usage

Usage:
    python scripts/update_agents_claude4.py
"""

import re
import sys
from pathlib import Path
from typing import Any, Dict

# Enable importing from scripts/common.py
sys.path.insert(0, str(Path(__file__).parent))
from common import get_agents_dir


# Agent role categories for customization
AGENT_ROLES: Dict[str, Dict[str, Any]] = {
    # Level 0: Chief Architect
    "chief-architect": {
        "level": 0,
        "thinking_tasks": [
            "System-wide architectural decisions affecting multiple sections",
            "Research paper feasibility analysis and component decomposition",
            "Resolving complex cross-section dependency conflicts",
            "Technology stack evaluations with long-term implications",
            "ADR creation requiring comprehensive trade-off analysis",
        ],
        "thinking_budget": {
            "simple": "Simple delegation tasks: Standard thinking",
            "complex": "Paper selection and architecture design: Extended thinking enabled",
            "debug": "Cross-section conflict resolution: Extended thinking enabled",
        },
        "style": "Strategic and high-level",
        "style_notes": [
            'Focus on "what" and "why", not "how"',
            "Clear rationale for architectural decisions",
            "Explicit success criteria and quality gates",
            "Visual diagrams for complex architectures (when applicable)",
        ],
        "code_examples": "Not applicable at this level (delegates to specialists)",
        "use_skills": True,
        "use_subagents": True,
    },
    # Level 1: Section Orchestrators
    "orchestrator": {
        "level": 1,
        "thinking_tasks": [
            "Section-wide architecture decisions with multiple subsections",
            "Breaking down complex specifications into subsections",
            "Resolving subsection dependency conflicts",
            "Resource allocation across parallel work streams",
        ],
        "thinking_budget": {
            "simple": "Routine delegation: Standard thinking",
            "complex": "Section planning and architecture: Extended thinking enabled",
            "debug": "Dependency conflict resolution: Extended thinking enabled",
        },
        "style": "Structured and architectural",
        "style_notes": [
            "Clear breakdown of section into subsections",
            "Dependency graphs and integration points",
            "Phase coordination across subsections",
            "Explicit delegation with success criteria",
        ],
        "code_examples": "Minimal - focus on architecture and delegation",
        "use_skills": True,
        "use_subagents": True,
    },
    # Level 2: Module Design
    "design": {
        "level": 2,
        "thinking_tasks": [
            "Module interface design with complex trade-offs",
            "Algorithm selection requiring deep domain knowledge",
            "API design balancing usability and performance",
            "Refactoring strategies for large-scale changes",
        ],
        "thinking_budget": {
            "simple": "Routine design tasks: Standard thinking",
            "complex": "Complex API design and algorithm selection: Extended thinking enabled",
            "debug": "Refactoring impact analysis: Extended thinking enabled",
        },
        "style": "Design-focused with clear specifications",
        "style_notes": [
            "Detailed interface specifications",
            "Algorithm pseudo-code and complexity analysis",
            "Design pattern rationale",
            "Integration points with other modules",
        ],
        "code_examples": "Interface signatures and usage examples with file paths",
        "use_skills": True,
        "use_subagents": True,
    },
    # Level 3: Component Specialists
    "specialist": {
        "level": 3,
        "thinking_tasks": [
            "Component architecture with multiple interdependent parts",
            "Complex algorithm implementation strategies",
            "Performance optimization trade-off analysis",
            "Integration patterns requiring coordination",
        ],
        "thinking_budget": {
            "simple": "Standard component tasks: Standard thinking",
            "complex": "Complex algorithms and optimizations: Extended thinking enabled",
            "debug": "Performance debugging and profiling: Extended thinking enabled",
        },
        "style": "Detailed and technical",
        "style_notes": [
            "Complete component specifications",
            "Algorithm details with complexity analysis",
            "Performance requirements and benchmarks",
            "Testing strategy and acceptance criteria",
        ],
        "code_examples": "Detailed with full file paths and line numbers",
        "use_skills": True,
        "use_subagents": True,
    },
    # Level 4: Implementation Engineers
    "engineer": {
        "level": 4,
        "thinking_tasks": [
            "Complex algorithm implementation with multiple edge cases",
            "Debugging subtle ownership or lifetime issues in Mojo",
            "Optimizing SIMD operations for performance-critical paths",
            "Resolving type system constraints for generic implementations",
        ],
        "thinking_budget": {
            "simple": "Standard function implementation: Standard thinking",
            "complex": "Complex tensor operations with SIMD: Extended thinking enabled",
            "debug": "Memory management debugging: Extended thinking enabled",
        },
        "style": "Implementation-focused and detail-oriented",
        "style_notes": [
            "Clear code examples with syntax highlighting",
            "Inline comments explaining non-obvious logic",
            "Step-by-step implementation breakdown",
            "Error handling patterns explicitly shown",
        ],
        "code_examples": "Always include full file paths and line numbers",
        "use_skills": True,
        "use_subagents": False,  # Level 4 rarely needs sub-agents
    },
    # Level 5: Junior Engineers
    "junior": {
        "level": 5,
        "thinking_tasks": [
            "Understanding specifications with ambiguous requirements",
            "Learning new Mojo patterns from existing code",
            "Debugging simple compilation errors",
        ],
        "thinking_budget": {
            "simple": "Routine tasks: Standard thinking",
            "complex": "Learning new patterns: Standard thinking with careful reading",
            "debug": "Simple debugging: Standard thinking",
        },
        "style": "Clear and learning-focused",
        "style_notes": [
            "Step-by-step approach",
            "Questions when unclear",
            "Reference to examples and patterns",
            "Testing verification at each step",
        ],
        "code_examples": "Simple examples with file paths",
        "use_skills": True,
        "use_subagents": False,  # Level 5 should not spawn sub-agents
    },
}


def determine_agent_category(agent_name: str) -> str:
    """Determine agent category based on name."""
    if "chief-architect" in agent_name:
        return "chief-architect"
    elif "orchestrator" in agent_name:
        return "orchestrator"
    elif "design" in agent_name:
        return "design"
    elif "junior" in agent_name:
        return "junior"
    elif "specialist" in agent_name:
        return "specialist"
    elif "engineer" in agent_name or "analyzer" in agent_name or "validator" in agent_name or "writer" in agent_name:
        return "engineer"
    else:
        # Default to specialist for review agents and others
        return "specialist"


def generate_thinking_guidance(category: str) -> str:
    """Generate Thinking Guidance section."""
    config = AGENT_ROLES[category]

    section = ["## Thinking Guidance", ""]
    section.append("**When to use extended thinking:**")
    section.append("")
    for task in config["thinking_tasks"]:
        section.append(f"- {task}")

    section.append("")
    section.append("**Thinking budget:**")
    section.append("")
    for key, value in config["thinking_budget"].items():
        section.append(f"- {value}")

    return "\n".join(section)


def generate_output_preferences(category: str, agent_name: str) -> str:
    """Generate Output Preferences section."""
    config = AGENT_ROLES[category]

    section = ["## Output Preferences", ""]
    section.append("**Format:** Structured Markdown with clear sections")
    section.append("")
    section.append(f"**Style:** {config['style']}")
    section.append("")
    for note in config["style_notes"]:
        section.append(f"- {note}")

    section.append("")
    section.append(f"**Code examples:** {config['code_examples']}")

    # Add file path format if code examples are used
    if "file paths" in config["code_examples"].lower():
        section.append("")
        section.append("- Use repo-relative paths: `path/to/file.mojo:line-range`")
        section.append("- Include line numbers when referencing existing code")
        section.append("- Show complete function signatures with parameter types")

    section.append("")
    # Customize decision sections by level
    if config["level"] <= 1:
        section.append(
            '**Decisions:** Always include explicit "Architectural Decision" or "Recommendation" sections with:'
        )
    elif config["level"] == 2:
        section.append('**Decisions:** Include "Design Decision" sections with:')
    else:
        section.append('**Decisions:** Include "Implementation Notes" sections with:')

    section.append("")
    if config["level"] <= 1:
        section.append("- Problem statement")
        section.append("- Considered alternatives")
        section.append("- Selected approach with rationale")
        section.append("- Impact analysis")
    elif config["level"] == 2:
        section.append("- Interface design rationale")
        section.append("- Algorithm selection reasoning")
        section.append("- Trade-offs and alternatives")
        section.append("- Integration requirements")
    else:
        section.append("- Algorithm choice rationale")
        section.append("- Performance trade-offs")
        section.append("- Edge case handling approach")
        section.append("- Testing strategy")

    return "\n".join(section)


def generate_delegation_patterns(category: str, agent_name: str) -> str:
    """Generate Delegation Patterns section."""
    config = AGENT_ROLES[category]

    section = ["## Delegation Patterns", ""]

    # Skills section
    section.append("**Use skills for:**")
    section.append("")

    # Add role-specific skills
    if category == "chief-architect":
        section.append("- `agent-run-orchestrator` - Spawning section orchestrators with clear objectives")
        section.append("- `agent-validate-config` - Validating agent configuration changes")
        section.append("- `agent-test-delegation` - Testing delegation patterns")
        section.append("- `agent-coverage-check` - Verifying workflow phase coverage")
        section.append("- `doc-generate-adr` - Creating Architectural Decision Records")
    elif category == "orchestrator":
        section.append("- `phase-plan-generate` - Generating detailed subsection plans")
        section.append("- `agent-run-orchestrator` - Delegating to subsection specialists")
        section.append("- `gh-create-pr-linked` - Creating section-level pull requests")
    elif category == "design":
        section.append("- `doc-generate-adr` - Documenting design decisions")
        section.append("- `validate-mojo-patterns` - Validating Mojo design patterns")
    elif category == "specialist":
        section.append("- `phase-test-tdd` - Coordinating TDD with test engineers")
        section.append("- `quality-run-linters` - Running quality checks")
        section.append("- `gh-create-pr-linked` - Creating component PRs")
    elif category == "engineer":
        section.append("- `mojo-format` - Formatting code before commits")
        section.append("- `mojo-test-runner` - Running test suites locally")
        section.append("- `quality-run-linters` - Pre-PR validation checks")
        section.append("- `gh-create-pr-linked` - Creating PRs linked to issues")
    elif category == "junior":
        section.append("- `mojo-format` - Formatting code")
        section.append("- `mojo-test-runner` - Running tests")
        section.append("- `quality-run-linters` - Basic quality checks")

    section.append("")

    # Sub-agents section
    if config["use_subagents"]:
        section.append("**Use sub-agents for:**")
        section.append("")

        if category == "chief-architect":
            section.append("- Strategic architectural analysis requiring deep domain expertise")
            section.append("- Cross-section dependency analysis and conflict resolution")
            section.append("- Research paper feasibility studies and algorithm extraction")
            section.append("- System-wide refactoring impact analysis")
        elif category == "orchestrator":
            section.append("- Complex subsection planning requiring specialized knowledge")
            section.append("- Investigating integration issues across subsections")
            section.append("- Technical spike research for section architecture")
        elif category == "design":
            section.append("- Researching algorithm implementations in literature")
            section.append("- Analyzing performance characteristics of design alternatives")
            section.append("- Investigating API design patterns in related projects")
        elif category == "specialist":
            section.append("- Deep algorithm research requiring literature review")
            section.append("- Performance profiling and optimization analysis")
            section.append("- Complex debugging requiring root cause investigation")
        elif category == "engineer":
            section.append("- Researching Mojo standard library APIs for unfamiliar features")
            section.append("- Analyzing existing codebase patterns for consistency")
            section.append("- Debugging complex compilation errors with unclear messages")

        section.append("")
        section.append("**Do NOT use sub-agents for:**")
        section.append("")

        if category in ["chief-architect", "orchestrator"]:
            section.append("- Simple delegation to direct reports (use direct assignment)")
            section.append("- Routine status updates (read issue comments)")
            section.append("- Standard skill invocations (use skills directly)")
        else:
            section.append("- Standard implementation tasks (your core responsibility)")
            section.append("- Running tests or formatting (use skills)")
            section.append("- Simple debugging (analyze directly)")
    else:
        section.append("**Sub-agents:** Not recommended at this level")
        section.append("")
        section.append("- Level 4/5 agents should complete tasks directly")
        section.append("- Escalate complex issues to specialists instead")
        section.append("- Use skills for automation, not sub-agents")

    return "\n".join(section)


def generate_subagent_usage(category: str, agent_name: str) -> str:
    """Generate Sub-Agent Usage section."""
    config = AGENT_ROLES[category]

    if not config["use_subagents"]:
        return ""

    section = ["## Sub-Agent Usage", ""]
    section.append("**When to spawn sub-agents:**")
    section.append("")

    if category == "chief-architect":
        section.append("- Analyzing complex research papers requiring algorithm extraction")
        section.append("- Evaluating architectural alternatives with detailed technical trade-offs")
        section.append("- Investigating system-wide refactoring impact across multiple sections")
        section.append("- Resolving ambiguous cross-section interface specifications")
    elif category == "orchestrator":
        section.append("- Complex subsection planning requiring specialized domain knowledge")
        section.append("- Investigating integration issues with external dependencies")
        section.append("- Technical feasibility analysis for section architecture")
        section.append("- Researching best practices for section-specific patterns")
    elif category == "design":
        section.append("- Researching algorithm variants and performance characteristics")
        section.append("- Analyzing API design patterns from related projects")
        section.append("- Investigating optimization techniques for specific operations")
        section.append("- Evaluating library dependencies for design decisions")
    elif category == "specialist":
        section.append("- Deep dive into algorithm implementation details from papers")
        section.append("- Performance profiling requiring extensive benchmarking")
        section.append("- Root cause analysis for complex bugs spanning multiple files")
        section.append("- Researching Mojo optimization patterns for specific scenarios")
    elif category == "engineer":
        section.append("- Encountering unclear Mojo compiler errors requiring investigation")
        section.append("- Needing to understand complex existing code patterns")
        section.append("- Investigating performance issues requiring profiling")
        section.append("- Researching Mojo best practices for unfamiliar features")

    section.append("")
    section.append("**Context to provide:**")
    section.append("")

    # Context varies by level
    if config["level"] == 0:
        section.append("- Relevant ADR file paths with line numbers")
        section.append("- Section orchestrator configuration files")
        section.append("- GitHub issue numbers with `gh issue view <number> --comments`")
        section.append("- Clear success criteria and scope boundaries")
    elif config["level"] == 1:
        section.append("- Section specification with file paths and line numbers")
        section.append("- Related subsection issue numbers")
        section.append("- Dependency graph or architecture diagrams")
        section.append("- Clear deliverables and success criteria")
    else:
        section.append("- Specification file path with line numbers")
        section.append("- Related source files with specific line ranges")
        section.append("- Failing test output or compiler errors (full text)")
        section.append("- Clear question or objective")
        section.append('- Success criteria: "Working implementation passing test X"')

    section.append("")
    section.append("**Example sub-agent invocation:**")
    section.append("")
    section.append("```markdown")

    # Example varies by level
    if category == "chief-architect":
        section.append("Spawn sub-agent: Analyze ResNet-50 paper feasibility")
        section.append("")
        section.append("**Objective:** Extract core components and identify dependencies")
        section.append("")
        section.append("**Context:**")
        section.append("- Paper: `/papers/resnet-50/paper.pdf`")
        section.append("- ADR template: `/docs/adr/template.md:1-50`")
        section.append("- Related issue: #1234 (gh issue view 1234)")
        section.append("")
        section.append("**Deliverables:**")
        section.append("1. Component breakdown (data loader, model, training loop)")
        section.append("2. Dependency graph with external libraries")
        section.append("3. Mojo feasibility assessment")
        section.append("")
        section.append("**Success criteria:**")
        section.append("- All required components identified")
        section.append("- Dependencies mapped with version requirements")
        section.append("- Feasibility report with risk assessment")
    elif category == "orchestrator":
        section.append("Spawn sub-agent: Research optimal data loading patterns")
        section.append("")
        section.append("**Objective:** Identify best practices for batched data loading in Mojo")
        section.append("")
        section.append("**Context:**")
        section.append("- Section spec: `/docs/specs/data-loading.md:20-80`")
        section.append("- Existing loader: `/shared/data/loader.mojo:1-150`")
        section.append("- Performance requirement: >1000 samples/sec")
        section.append("")
        section.append("**Deliverables:**")
        section.append("1. Pattern comparison (memory-mapped, prefetch, streaming)")
        section.append("2. Mojo implementation examples")
        section.append("3. Performance trade-off analysis")
        section.append("")
        section.append("**Success criteria:**")
        section.append("- 3+ patterns evaluated with benchmarks")
        section.append("- Recommendation with rationale")
    else:
        section.append("Spawn sub-agent: Investigate SIMD vectorization pattern")
        section.append("")
        section.append("**Objective:** Understand optimal SIMD approach for tensor operations")
        section.append("")
        section.append("**Context:**")
        section.append("- Current implementation: `/shared/core/ops.mojo:200-250`")
        section.append("- Test file: `/tests/shared/core/test_ops.mojo:45-60`")
        section.append('- Specification: "Must handle non-aligned sizes"')
        section.append('- Compiler error: "cannot vectorize with dynamic size"')
        section.append("")
        section.append("**Deliverables:**")
        section.append("1. Working SIMD pattern handling edge cases")
        section.append("2. Performance comparison with scalar")
        section.append("3. Test coverage for boundary conditions")
        section.append("")
        section.append("**Success criteria:**")
        section.append("- Code compiles without warnings")
        section.append("- All tests pass")
        section.append("- Performance >2x scalar baseline")

    section.append("```")

    return "\n".join(section)


def find_insertion_point(content: str) -> int:
    """Find the best insertion point for new sections."""
    # Try different ending patterns
    patterns = [
        r"\n---\n\n\*\*References\*\*:",  # Most common: --- followed by **References**:
        r"\n---\n\n\*[A-Z]",  # Alternative: --- followed by italicized summary
        r"\n## (Coordinates With|Escalates To)\n",  # Review specialists format
    ]

    for pattern in patterns:
        match = re.search(pattern, content)
        if match:
            return match.start()

    # Otherwise insert at the end, but before final newline if it exists
    content = content.rstrip()
    return len(content)


def update_agent_file(agent_path: Path) -> bool:
    """Update a single agent file with Claude 4 sections."""
    agent_name = agent_path.stem

    # Skip if already updated
    content = agent_path.read_text()
    if "## Thinking Guidance" in content:
        print(f"  ✓ {agent_name} already updated")
        return False

    # Determine category
    category = determine_agent_category(agent_name)

    print(f"  Updating {agent_name} (category: {category})...")

    # Generate sections
    thinking = generate_thinking_guidance(category)
    output = generate_output_preferences(category, agent_name)
    delegation = generate_delegation_patterns(category, agent_name)
    subagent = generate_subagent_usage(category, agent_name)

    # Combine sections
    new_sections = [thinking, "", output, "", delegation]
    if subagent:
        new_sections.extend(["", subagent])

    new_content = "\n".join(new_sections)

    # Find insertion point
    insertion_point = find_insertion_point(content)

    # Insert new sections
    updated_content = content[:insertion_point] + "\n\n" + new_content + "\n" + content[insertion_point:]

    # Write back
    agent_path.write_text(updated_content)
    print(f"  ✓ {agent_name} updated successfully")
    return True


def main():
    """Main entry point."""
    agents_dir = get_agents_dir()

    if not agents_dir.exists():
        print(f"Error: Agents directory not found: {agents_dir}")
        return 1

    agent_files = sorted(agents_dir.glob("*.md"))

    print(f"Found {len(agent_files)} agent files")
    print()

    updated_count = 0
    for agent_path in agent_files:
        if update_agent_file(agent_path):
            updated_count += 1

    print()
    print(f"Updated {updated_count}/{len(agent_files)} agent files")

    return 0


if __name__ == "__main__":
    exit(main())
