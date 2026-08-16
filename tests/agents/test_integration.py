#!/usr/bin/env python3
"""
Test agent integration with 5-phase workflow and git worktrees.

This script validates:
- 5-phase workflow integration (Plan → Test/Impl/Package → Cleanup)
- Git worktree compatibility
- Parallel execution scenarios
- Coordination patterns
- Issue-based workflow

Usage:
    python3 tests/agents/test_integration.py [agent_dir]
"""

import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Set
from dataclasses import dataclass


# Security limits
MAX_FILE_SIZE = 102400  # 100KB max file size to prevent DoS


@dataclass
class WorkflowInfo:
    """Information about agent workflow participation."""

    agent_name: str
    level: int
    phases: Set[str]  # Which phases this agent participates in
    supports_parallel: bool
    worktree_compatible: bool


class IntegrationTester:
    """Test agent integration with workflows."""

    PHASES = {"Plan", "Test", "Implementation", "Packaging", "Cleanup"}

    # Phase participation by level (expected)
    EXPECTED_PHASES = {
        0: {"Plan", "Cleanup"},  # Chief Architect
        1: {"Plan", "Cleanup"},  # Section Orchestrators
        2: {"Plan", "Cleanup"},  # Module Design
        3: {"Plan", "Test", "Implementation", "Packaging", "Cleanup"},  # Specialists
        4: {"Test", "Implementation", "Packaging", "Cleanup"},  # Engineers
        5: {"Test", "Implementation", "Packaging"},  # Junior Engineers
    }

    def __init__(self, agents_dir: Path):
        """Initialize tester.

        Args:
            agents_dir: Directory containing agent .md files
        """
        self.agents_dir = agents_dir
        self.workflow_info: List[WorkflowInfo] = []
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def analyze_all(self) -> List[WorkflowInfo]:
        """Analyze workflow integration for all agents.

        Returns:
            List of workflow information
        """
        if not self.agents_dir.exists():
            self.errors.append(f"Agents directory not found: {self.agents_dir}")
            return []

        agent_files = sorted(self.agents_dir.glob("*.md"))

        if not agent_files:
            self.errors.append(f"No agent files found in {self.agents_dir}")
            return []

        print(f"Analyzing workflow integration in {len(agent_files)} agents...")

        for agent_file in agent_files:
            # Check file size before reading
            file_size = agent_file.stat().st_size
            if file_size > MAX_FILE_SIZE:
                logging.warning(f"Skipping {agent_file.name} (file too large: {file_size} bytes)")
                self.errors.append(f"{agent_file.name}: File too large ({file_size} bytes)")
                continue

            info = self._analyze_workflow(agent_file)
            if info:
                self.workflow_info.append(info)

        return self.workflow_info

    def _analyze_workflow(self, file_path: Path) -> WorkflowInfo:
        """Analyze workflow integration for a single agent.

        Args:
            file_path: Path to agent .md file

        Returns:
            WorkflowInfo
        """
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            self.errors.append(f"Failed to read {file_path.name}: {e}")
            logging.error(f"Failed to read {file_path.name}: {e}")
            return None

        # Extract agent name and level
        name_match = re.search(r"name:\s*(.+)", content)
        agent_name = name_match.group(1).strip() if name_match else file_path.stem

        level = self._detect_level(agent_name, content)
        phases = self._extract_phases(content)
        supports_parallel = self._check_parallel_support(content)
        worktree_compatible = self._check_worktree_compatibility(content)

        return WorkflowInfo(
            agent_name=agent_name,
            level=level if level is not None else -1,
            phases=phases,
            supports_parallel=supports_parallel,
            worktree_compatible=worktree_compatible,
        )

    def _detect_level(self, name: str, content: str) -> int:
        """Detect agent level."""
        level_match = re.search(r"Level (\d)", content)
        if level_match:
            return int(level_match.group(1))

        if "chief-architect" in name:
            return 0
        elif "orchestrator" in name:
            return 1
        elif "design" in name:
            return 2
        elif "specialist" in name:
            return 3
        elif "junior" in name and "engineer" in name:
            return 5
        elif "engineer" in name:
            return 4

        return -1

    def _extract_phases(self, content: str) -> Set[str]:
        """Extract workflow phases mentioned in agent."""
        phases = set()
        content_lower = content.lower()

        # Look for explicit phase mentions
        for phase in self.PHASES:
            phase_lower = phase.lower()
            # Check for "Workflow Phase:" section or phase mentions
            if re.search(rf"\b{phase_lower}\b", content_lower):
                phases.add(phase)

        return phases

    def _check_parallel_support(self, content: str) -> bool:
        """Check if agent supports parallel execution."""
        content_lower = content.lower()

        parallel_indicators = [
            "parallel",
            "concurrent",
            "simultaneously",
            "independent",
            "can run in parallel",
        ]

        return any(indicator in content_lower for indicator in parallel_indicators)

    def _check_worktree_compatibility(self, content: str) -> bool:
        """Check if agent mentions worktree compatibility."""
        content_lower = content.lower()

        worktree_indicators = [
            "worktree",
            "git worktree",
            "separate worktree",
            "issue-specific",
            "per-issue",
        ]

        return any(indicator in content_lower for indicator in worktree_indicators)

    def test_phase_coverage(self) -> None:
        """Test that agents cover all workflow phases appropriately."""
        print("\n" + "=" * 80)
        print("5-PHASE WORKFLOW COVERAGE")
        print("=" * 80)

        # Group by phase
        agents_by_phase: Dict[str, List[str]] = {phase: [] for phase in self.PHASES}

        for info in self.workflow_info:
            for phase in info.phases:
                agents_by_phase[phase].append(f"{info.agent_name} (L{info.level})")

        # Check each phase
        for phase in ["Plan", "Test", "Implementation", "Packaging", "Cleanup"]:
            agents = agents_by_phase[phase]
            print(f"\n{phase} Phase: {len(agents)} agent(s)")

            if not agents:
                self.warnings.append(f"No agents defined for {phase} phase")
                print("  ⚠ WARNING: No agents for this phase")
            else:
                for agent in agents[:5]:
                    print(f"  - {agent}")
                if len(agents) > 5:
                    print(f"  ... and {len(agents) - 5} more")

    def test_level_phase_alignment(self) -> None:
        """Test that agent levels participate in expected phases."""
        print("\n" + "=" * 80)
        print("LEVEL-PHASE ALIGNMENT")
        print("=" * 80)

        for info in self.workflow_info:
            level = info.level
            if level < 0 or level > 5:
                continue

            expected = self.EXPECTED_PHASES.get(level, set())
            actual = info.phases

            print(f"\n{info.agent_name} (Level {level}):")
            print(f"  Expected phases: {', '.join(sorted(expected)) if expected else 'N/A'}")
            print(f"  Actual phases: {', '.join(sorted(actual)) if actual else 'None found'}")

            # Check for missing expected phases
            missing = expected - actual
            if missing:
                self.warnings.append(f"{info.agent_name} (L{level}) missing expected phases: {', '.join(missing)}")
                print(f"  ⚠ Missing: {', '.join(sorted(missing))}")

            # Check for unexpected phases
            unexpected = actual - expected
            if unexpected:
                self.warnings.append(f"{info.agent_name} (L{level}) has unexpected phases: {', '.join(unexpected)}")
                print(f"  ⚠ Unexpected: {', '.join(sorted(unexpected))}")

    def test_parallel_execution(self) -> None:
        """Test parallel execution support."""
        print("\n" + "=" * 80)
        print("PARALLEL EXECUTION SUPPORT")
        print("=" * 80)

        parallel_agents = [info for info in self.workflow_info if info.supports_parallel]
        non_parallel = [info for info in self.workflow_info if not info.supports_parallel]

        print(f"\nAgents supporting parallel execution: {len(parallel_agents)}")
        for info in parallel_agents:
            print(f"  - {info.agent_name} (L{info.level})")

        print(f"\nAgents without parallel execution guidance: {len(non_parallel)}")

        # Level 3-5 agents should support parallel execution (Test/Impl/Package phases)
        for info in non_parallel:
            if info.level in [3, 4, 5]:
                if any(phase in info.phases for phase in ["Test", "Implementation", "Packaging"]):
                    self.warnings.append(
                        f"{info.agent_name} (L{info.level}) should mention parallel execution capability"
                    )

    def test_worktree_integration(self) -> None:
        """Test git worktree integration."""
        print("\n" + "=" * 80)
        print("GIT WORKTREE INTEGRATION")
        print("=" * 80)

        worktree_agents = [info for info in self.workflow_info if info.worktree_compatible]

        print(f"\nAgents with worktree guidance: {len(worktree_agents)}")
        for info in worktree_agents:
            print(f"  - {info.agent_name} (L{info.level})")

        no_worktree = [info for info in self.workflow_info if not info.worktree_compatible]

        if no_worktree:
            print(f"\nAgents without worktree guidance: {len(no_worktree)}")
            for info in no_worktree[:10]:
                print(f"  - {info.agent_name} (L{info.level})")

            # This is informational, not necessarily a warning
            print("\nNote: Worktree guidance is recommended for agents that work in parallel phases")

    def test_coordination_scenarios(self) -> None:
        """Test common coordination scenarios."""
        print("\n" + "=" * 80)
        print("COORDINATION SCENARIOS")
        print("=" * 80)

        # Scenario 1: Plan → Parallel (Test/Impl/Package) → Cleanup
        print("\nScenario 1: Full 5-Phase Workflow")
        print("Plan phase agents (should complete first):")

        plan_agents = [info for info in self.workflow_info if "Plan" in info.phases]
        for info in plan_agents[:5]:
            print(f"  - {info.agent_name} (L{info.level})")

        print("\nParallel phase agents (Test/Impl/Package):")
        parallel_phase_agents = [
            info
            for info in self.workflow_info
            if any(phase in info.phases for phase in ["Test", "Implementation", "Packaging"])
        ]
        for info in parallel_phase_agents[:5]:
            phases = [p for p in ["Test", "Implementation", "Packaging"] if p in info.phases]
            print(f"  - {info.agent_name} (L{info.level}): {', '.join(phases)}")

        print("\nCleanup phase agents (should run after parallel phases):")
        cleanup_agents = [info for info in self.workflow_info if "Cleanup" in info.phases]
        for info in cleanup_agents[:5]:
            print(f"  - {info.agent_name} (L{info.level})")

        # Scenario 2: Cross-worktree coordination
        print("\n\nScenario 2: Cross-Worktree Coordination")
        print("Agents that need to coordinate across worktrees:")

        # Test and Implementation engineers often coordinate
        test_impl_agents = [
            info
            for info in self.workflow_info
            if info.level in [4, 5] and any(phase in info.phases for phase in ["Test", "Implementation"])
        ]

        for info in test_impl_agents[:5]:
            print(f"  - {info.agent_name} (L{info.level})")

        if test_impl_agents:
            print("\nThese agents should have coordination guidance for TDD workflow")

    def print_summary(self) -> None:
        """Print test summary."""
        print("\n" + "=" * 80)
        print("INTEGRATION TEST SUMMARY")
        print("=" * 80)
        print(f"Agents analyzed: {len(self.workflow_info)}")
        print(f"Errors: {len(self.errors)}")
        print(f"Warnings: {len(self.warnings)}")

        if self.errors:
            print("\nErrors:")
            for error in self.errors:
                print(f"  - {error}")

        if self.warnings:
            print("\nWarnings:")
            for warning in self.warnings[:10]:  # Show first 10
                print(f"  - {warning}")
            if len(self.warnings) > 10:
                print(f"  ... and {len(self.warnings) - 10} more warnings")

        print("=" * 80)


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success, 1 for errors).
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)

    # Determine agents directory
    agents_dir_arg = sys.argv[1] if len(sys.argv) > 1 else None

    # Validate input
    if agents_dir_arg:
        agents_dir = Path(agents_dir_arg)
        if not agents_dir.exists():
            logger.error(f"Directory does not exist: {agents_dir_arg}")
            return 1

        if not agents_dir.is_dir():
            logger.error(f"Path is not a directory: {agents_dir_arg}")
            return 1
    else:
        current = Path.cwd()
        agents_dir = current / ".claude" / "agents"

        if not agents_dir.exists():
            for parent in current.parents:
                test_dir = parent / ".claude" / "agents"
                if test_dir.exists():
                    agents_dir = test_dir
                    break

    print(f"Testing workflow integration in: {agents_dir}\n")

    tester = IntegrationTester(agents_dir)
    workflow_info = tester.analyze_all()

    if not workflow_info:
        print("\nNo workflow information found - cannot run tests")
        return 1

    tester.test_phase_coverage()
    tester.test_level_phase_alignment()
    tester.test_parallel_execution()
    tester.test_worktree_integration()
    tester.test_coordination_scenarios()
    tester.print_summary()

    return 1 if tester.errors else 0


if __name__ == "__main__":
    sys.exit(main())
