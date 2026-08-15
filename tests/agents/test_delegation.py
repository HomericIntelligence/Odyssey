#!/usr/bin/env python3
"""
Test agent delegation patterns across the 6-level hierarchy.

This script validates:
- Level 0 → Level 1 delegation
- Level 1 → Level 2 delegation
- Level 2 → Level 3 delegation
- Level 3 → Level 4 delegation
- Level 4 → Level 5 delegation
- Escalation triggers are defined
- Horizontal coordination patterns

Usage:
    python3 tests/agents/test_delegation.py [agent_dir]
"""

import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass


# Security limits
MAX_FILE_SIZE = 102400  # 100KB max file size to prevent DoS


@dataclass
class DelegationInfo:
    """Information about delegation patterns."""

    agent_name: str
    level: int
    delegates_to: List[str]
    coordinates_with: List[str]
    escalates_to: Optional[str]
    escalation_triggers: List[str]


class DelegationTester:
    """Test delegation patterns across agent hierarchy."""

    LEVEL_NAMES = {
        0: "Meta-Orchestrator",
        1: "Section Orchestrators",
        2: "Module Design Agents",
        3: "Component Specialists",
        4: "Implementation Engineers",
        5: "Junior Engineers",
    }

    def __init__(self, agents_dir: Path):
        """Initialize tester.

        Args:
            agents_dir: Directory containing agent .md files
        """
        self.agents_dir = agents_dir
        self.delegation_info: List[DelegationInfo] = []
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def analyze_all(self) -> List[DelegationInfo]:
        """Analyze delegation patterns in all agents.

        Returns:
            List of delegation information
        """
        if not self.agents_dir.exists():
            self.errors.append(f"Agents directory not found: {self.agents_dir}")
            return []

        agent_files = sorted(self.agents_dir.glob("*.md"))

        if not agent_files:
            self.errors.append(f"No agent files found in {self.agents_dir}")
            return []

        print(f"Analyzing delegation patterns in {len(agent_files)} agents...")

        for agent_file in agent_files:
            # Check file size before reading
            file_size = agent_file.stat().st_size
            if file_size > MAX_FILE_SIZE:
                logging.warning(f"Skipping {agent_file.name} (file too large: {file_size} bytes)")
                self.errors.append(f"{agent_file.name}: File too large ({file_size} bytes)")
                continue

            info = self._analyze_delegation(agent_file)
            if info:
                self.delegation_info.append(info)

        return self.delegation_info

    def _analyze_delegation(self, file_path: Path) -> Optional[DelegationInfo]:
        """Analyze delegation patterns in a single agent.

        Args:
            file_path: Path to agent .md file

        Returns:
            DelegationInfo or None
        """
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            self.errors.append(f"Failed to read {file_path.name}: {e}")
            logging.error(f"Failed to read {file_path.name}: {e}")
            return None

        # Extract agent name and level
        name_match = re.search(r"name:\s*(.+)", content)
        if not name_match:
            self.errors.append(f"No name found in {file_path.name}")
            return None

        agent_name = name_match.group(1).strip()
        level = self._detect_level(agent_name, content)

        if level is None:
            self.warnings.append(f"{agent_name}: Could not determine hierarchy level")
            level = -1

        # Extract delegation information
        delegates_to = self._extract_delegates_to(content)
        coordinates_with = self._extract_coordinates_with(content)
        escalates_to = self._extract_escalates_to(content, level)
        escalation_triggers = self._extract_escalation_triggers(content)

        return DelegationInfo(
            agent_name=agent_name,
            level=level,
            delegates_to=delegates_to,
            coordinates_with=coordinates_with,
            escalates_to=escalates_to,
            escalation_triggers=escalation_triggers,
        )

    def _detect_level(self, name: str, content: str) -> Optional[int]:
        """Detect agent level.

        Args:
            name: Agent name
            content: Agent content

        Returns:
            Level number (0-5) or None
        """
        # Check content for explicit level
        level_match = re.search(r"Level (\d)", content)
        if level_match:
            return int(level_match.group(1))

        # Infer from name
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

        return None

    def _extract_delegates_to(self, content: str) -> List[str]:
        """Extract delegation targets.

        Args:
            content: Agent content

        Returns:
            List of agent types/names delegated to
        """
        delegates = []

        # Look for "Delegates To:" or "Delegates to:" sections
        delegate_pattern = r"(?:Delegates [Tt]o|delegates to):\s*(.+?)(?=\n\n|\n#|\Z)"
        matches = re.finditer(delegate_pattern, content, re.DOTALL)

        for match in matches:
            delegate_text = match.group(1)
            # Extract agent names/types
            # Common patterns: "Agent Name", "Agent Name (Level X)", etc.
            lines = delegate_text.strip().split("\n")
            for line in lines:
                line = line.strip()
                if line and not line.startswith("#"):
                    # Clean up markdown formatting
                    line = re.sub(r"^\*\*|\*\*$", "", line)  # Remove bold
                    line = re.sub(r"^\*|-", "", line)  # Remove list markers
                    line = line.strip()
                    if line and len(line) < 100:  # Sanity check
                        delegates.append(line)

        return delegates

    def _extract_coordinates_with(self, content: str) -> List[str]:
        """Extract coordination partners.

        Args:
            content: Agent content

        Returns:
            List of agent types/names coordinated with
        """
        coordinates = []

        # Look for "Coordinates With:" sections
        coord_pattern = r"Coordinates [Ww]ith:\s*(.+?)(?=\n\n|\n#|\Z)"
        matches = re.finditer(coord_pattern, content, re.DOTALL)

        for match in matches:
            coord_text = match.group(1)
            lines = coord_text.strip().split("\n")
            for line in lines:
                line = line.strip()
                if line and not line.startswith("#"):
                    line = re.sub(r"^\*\*|\*\*$", "", line)
                    line = re.sub(r"^\*|-", "", line)
                    line = line.strip()
                    if line and len(line) < 100:
                        coordinates.append(line)

        return coordinates

    def _extract_escalates_to(self, content: str, level: int) -> Optional[str]:
        """Extract escalation target.

        Args:
            content: Agent content
            level: Agent level

        Returns:
            Escalation target or None
        """
        # Look for explicit escalation mentions
        escalate_pattern = r"Escalate[s]? to:\s*(.+?)(?=\n|\.|,)"
        match = re.search(escalate_pattern, content, re.IGNORECASE)

        if match:
            return match.group(1).strip()

        # Default: escalate to level above (except level 0)
        if level > 0 and level <= 5:
            return self.LEVEL_NAMES.get(level - 1, "Superior")

        return None

    def _extract_escalation_triggers(self, content: str) -> List[str]:
        """Extract escalation triggers.

        Args:
            content: Agent content

        Returns:
            List of escalation triggers
        """
        triggers = []

        # Look for "Escalation Triggers" or "Escalate when" sections
        trigger_patterns = [
            r"Escalation Triggers[:\s]+(.+?)(?=\n##|\Z)",
            r"Escalate when[:\s]+(.+?)(?=\n##|\Z)",
            r"Must [Ee]scalate[:\s]+(.+?)(?=\n##|\Z)",
        ]

        for pattern in trigger_patterns:
            matches = re.finditer(pattern, content, re.DOTALL | re.IGNORECASE)
            for match in matches:
                trigger_text = match.group(1)
                # Extract bullet points or lines
                lines = trigger_text.strip().split("\n")
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        line = re.sub(r"^\*|-", "", line)
                        line = line.strip()
                        if line and len(line) > 10:  # Meaningful trigger
                            triggers.append(line)

        return triggers

    def test_delegation_chains(self) -> None:
        """Test that delegation chains are properly defined.

        Validates Level N → Level N+1 delegation for all levels.
        """
        print("\n" + "=" * 80)
        print("DELEGATION CHAIN VALIDATION")
        print("=" * 80)

        # Group agents by level
        agents_by_level: Dict[int, List[DelegationInfo]] = {}
        for info in self.delegation_info:
            level = info.level
            if level not in agents_by_level:
                agents_by_level[level] = []
            agents_by_level[level].append(info)

        # Check each level
        for level in range(6):
            print(f"\n{self.LEVEL_NAMES.get(level, f'Level {level}')}:")

            agents = agents_by_level.get(level, [])

            if not agents:
                print("  ⚠ No agents at this level")
                continue

            for agent in agents:
                print(f"\n  {agent.agent_name}:")

                if level < 5:  # All but junior engineers should delegate
                    if agent.delegates_to:
                        print(f"    Delegates to: {', '.join(agent.delegates_to[:3])}")
                        if len(agent.delegates_to) > 3:
                            print(f"      ... and {len(agent.delegates_to) - 3} more")
                    else:
                        self.warnings.append(f"{agent.agent_name} (Level {level}) has no delegation targets defined")
                        print("    ⚠ WARNING: No delegation targets defined")
                else:  # Junior engineers (level 5) should NOT delegate
                    if agent.delegates_to:
                        self.warnings.append(f"{agent.agent_name} (Level 5 - Junior) should not delegate further")
                        print("    ⚠ WARNING: Junior engineer delegates (should not)")

    def test_escalation_paths(self) -> None:
        """Test that escalation paths are defined."""
        print("\n" + "=" * 80)
        print("ESCALATION PATH VALIDATION")
        print("=" * 80)

        for info in self.delegation_info:
            print(f"\n{info.agent_name} (Level {info.level}):")

            if info.level == 0:
                # Chief Architect has no one to escalate to
                if info.escalates_to:
                    self.warnings.append(f"{info.agent_name} is Level 0 but has escalation target: {info.escalates_to}")
                print("  Escalates to: N/A (top level)")
            else:
                if info.escalates_to:
                    print(f"  Escalates to: {info.escalates_to}")
                else:
                    self.warnings.append(f"{info.agent_name} has no escalation target defined")
                    print("  ⚠ WARNING: No escalation target")

            # Check for escalation triggers
            if info.escalation_triggers:
                print(f"  Escalation triggers ({len(info.escalation_triggers)}):")
                for trigger in info.escalation_triggers[:3]:
                    print(f"    - {trigger[:80]}...")
                if len(info.escalation_triggers) > 3:
                    print(f"    ... and {len(info.escalation_triggers) - 3} more")
            else:
                self.warnings.append(f"{info.agent_name} has no escalation triggers defined")
                print("  ⚠ WARNING: No escalation triggers defined")

    def test_horizontal_coordination(self) -> None:
        """Test horizontal coordination patterns."""
        print("\n" + "=" * 80)
        print("HORIZONTAL COORDINATION VALIDATION")
        print("=" * 80)

        agents_with_coordination = [info for info in self.delegation_info if info.coordinates_with]

        print(f"\nAgents with coordination defined: {len(agents_with_coordination)}/{len(self.delegation_info)}")

        for info in agents_with_coordination:
            print(f"\n{info.agent_name} (Level {info.level}):")
            print(f"  Coordinates with: {', '.join(info.coordinates_with[:3])}")
            if len(info.coordinates_with) > 3:
                print(f"    ... and {len(info.coordinates_with) - 3} more")

        # Agents without coordination (may be OK, just informational)
        agents_without = [info for info in self.delegation_info if not info.coordinates_with]
        if agents_without:
            print(f"\n\nAgents without coordination defined: {len(agents_without)}")
            for info in agents_without:
                print(f"  - {info.agent_name} (Level {info.level})")

    def print_summary(self) -> None:
        """Print test summary."""
        print("\n" + "=" * 80)
        print("DELEGATION TEST SUMMARY")
        print("=" * 80)
        print(f"Agents analyzed: {len(self.delegation_info)}")
        print(f"Errors: {len(self.errors)}")
        print(f"Warnings: {len(self.warnings)}")

        if self.errors:
            print("\nErrors:")
            for error in self.errors:
                print(f"  - {error}")

        if self.warnings:
            print("\nWarnings:")
            for warning in self.warnings:
                print(f"  - {warning}")

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

    print(f"Testing delegation patterns in: {agents_dir}\n")

    tester = DelegationTester(agents_dir)
    delegation_info = tester.analyze_all()

    if not delegation_info:
        print("\nNo delegation information found - cannot run tests")
        return 1

    tester.test_delegation_chains()
    tester.test_escalation_paths()
    tester.test_horizontal_coordination()
    tester.print_summary()

    return 1 if tester.errors else 0


if __name__ == "__main__":
    sys.exit(main())
