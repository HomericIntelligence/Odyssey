#!/usr/bin/env python3
"""
Test agent discovery and loading in Claude Code.

This script tests:
- Agent file discovery in .claude/agents/
- Configuration loading without errors
- Agent activation pattern detection
- Directory structure validation

Usage:
    python3 tests/agents/test_loading.py [agent_dir]
"""

import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

# Add scripts/agents to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts" / "agents"))

from agent_utils import AgentInfo as BaseAgentInfo, extract_frontmatter_raw
import yaml

# Security limits
MAX_FILE_SIZE = 102400  # 100KB max file size to prevent DoS


class AgentInfo(BaseAgentInfo):
    """Extended AgentInfo with test-specific attributes."""

    def __init__(self, file_path: Path, frontmatter: Dict, role: Optional[str] = None):
        """Initialize with base AgentInfo plus role.

        Args:
            file_path: Path to agent markdown file
            frontmatter: Parsed frontmatter dictionary
            role: Optional role description from content
        """
        super().__init__(file_path, frontmatter)
        self.role = role
        # Convert tools string to list for backwards compatibility
        self.tools = self.get_tools_list()


class AgentLoadingTester:
    """Test agent discovery and loading."""

    def __init__(self, agents_dir: Path):
        """Initialize tester.

        Args:
            agents_dir: Directory containing agent .md files
        """
        self.agents_dir = agents_dir
        self.agents: List[AgentInfo] = []
        self.errors: List[str] = []

    def discover_agents(self) -> List[AgentInfo]:
        """Discover all agent configuration files.

        Returns:
            List of discovered agents
        """
        if not self.agents_dir.exists():
            self.errors.append(f"Agents directory not found: {self.agents_dir}")
            return []

        agent_files = sorted(self.agents_dir.glob("*.md"))

        if not agent_files:
            self.errors.append(f"No agent files found in {self.agents_dir}")
            return []

        print(f"Discovered {len(agent_files)} agent configuration files:")

        for agent_file in agent_files:
            # Check file size before reading
            file_size = agent_file.stat().st_size
            if file_size > MAX_FILE_SIZE:
                logging.warning(f"Skipping {agent_file.name} (file too large: {file_size} bytes)")
                print(f"  ⚠ {agent_file.name} (file too large)")
                self.errors.append(f"{agent_file.name}: File too large ({file_size} bytes)")
                continue

            agent_info = self._load_agent(agent_file)
            if agent_info:
                self.agents.append(agent_info)
                print(f"  ✓ {agent_file.name}")
            else:
                print(f"  ✗ {agent_file.name} (failed to load)")

        return self.agents

    def _load_agent(self, file_path: Path) -> Optional[AgentInfo]:
        """Load agent configuration from file.

        Args:
            file_path: Path to agent .md file

        Returns:
            AgentInfo if loaded successfully, None otherwise
        """
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            self.errors.append(f"Failed to read {file_path.name}: {e}")
            logging.error(f"Failed to read {file_path.name}: {e}")
            return None

        # Extract frontmatter using shared utility
        frontmatter_text = extract_frontmatter_raw(content)
        if not frontmatter_text:
            self.errors.append(f"No frontmatter in {file_path.name}")
            return None

        # Parse YAML frontmatter
        try:
            frontmatter = yaml.safe_load(frontmatter_text)
        except yaml.YAMLError as e:
            self.errors.append(f"Invalid YAML in {file_path.name}: {e}")
            return None

        if not isinstance(frontmatter, dict):
            self.errors.append(f"Frontmatter is not a dictionary in {file_path.name}")
            return None

        # Extract required fields
        name = frontmatter.get("name", "")
        description = frontmatter.get("description", "")
        tools_str = frontmatter.get("tools", "")
        model = frontmatter.get("model", "")

        if not all([name, description, tools_str, model]):
            self.errors.append(f"Missing required fields in {file_path.name}")
            return None

        # Extract additional info from content
        role = self._extract_role(content)

        return AgentInfo(file_path=file_path, frontmatter=frontmatter, role=role)

    def _extract_role(self, content: str) -> Optional[str]:
        """Extract role description from content.

        Args:
            content: Agent content

        Returns:
            Role description or None
        """
        # Look for "## Role" section
        role_match = re.search(r"^##\s+Role\s*\n(.+?)(?=\n##|\Z)", content, re.MULTILINE | re.DOTALL)
        if role_match:
            role_text = role_match.group(1).strip()
            # Return first line/paragraph
            return role_text.split("\n")[0].strip()

        return None

    def test_activation_patterns(self) -> Dict[str, List[str]]:
        """Test that agent descriptions would trigger auto-invocation.

        Returns:
            Dict mapping agent names to activation keywords
        """
        print("\n" + "=" * 80)
        print("ACTIVATION PATTERN ANALYSIS")
        print("=" * 80)

        activation_map = {}

        # Common activation keywords
        action_keywords = {
            "implement",
            "design",
            "test",
            "write",
            "create",
            "build",
            "architect",
            "orchestrate",
            "coordinate",
            "review",
            "analyze",
            "optimize",
            "document",
            "package",
            "integrate",
            "validate",
        }

        for agent in self.agents:
            desc_lower = agent.description.lower()
            found_keywords = [kw for kw in action_keywords if kw in desc_lower]
            activation_map[agent.name] = found_keywords

            if found_keywords:
                print(f"\n{agent.name}:")
                print(f"  Description: {agent.description}")
                print(f"  Activation keywords: {', '.join(found_keywords)}")
            else:
                print(f"\n⚠ {agent.name}: No clear activation keywords")
                print(f"  Description: {agent.description}")

        return activation_map

    def test_hierarchy_coverage(self) -> None:
        """Test that all hierarchy levels are covered.

        Prints coverage report for levels 0-5.
        """
        print("\n" + "=" * 80)
        print("HIERARCHY COVERAGE")
        print("=" * 80)

        level_names = {
            0: "Meta-Orchestrator",
            1: "Section Orchestrators",
            2: "Module Design Agents",
            3: "Component Specialists",
            4: "Implementation Engineers",
            5: "Junior Engineers",
        }

        agents_by_level: Dict[int, List[AgentInfo]] = {i: [] for i in range(6)}

        for agent in self.agents:
            if agent.level is not None:
                agents_by_level[agent.level].append(agent)

        for level in range(6):
            agents = agents_by_level[level]
            count = len(agents)
            level_name = level_names[level]

            print(f"\nLevel {level} ({level_name}): {count} agent(s)")
            for agent in agents:
                print(f"  - {agent.name}")

            if count == 0:
                print("  ⚠ WARNING: No agents defined for this level")

    def test_tool_usage(self) -> None:
        """Analyze tool usage across agents.

        Prints report of which tools are used and by which agents.
        """
        print("\n" + "=" * 80)
        print("TOOL USAGE ANALYSIS")
        print("=" * 80)

        all_tools: Set[str] = set()
        tool_usage: Dict[str, List[str]] = {}

        for agent in self.agents:
            for tool in agent.tools:
                all_tools.add(tool)
                if tool not in tool_usage:
                    tool_usage[tool] = []
                tool_usage[tool].append(agent.name)

        print(f"\nTotal unique tools: {len(all_tools)}")
        print("\nTool usage:")

        for tool in sorted(all_tools):
            agents = tool_usage[tool]
            print(f"\n{tool}: {len(agents)} agent(s)")
            for agent_name in agents[:3]:  # Show first 3
                print(f"  - {agent_name}")
            if len(agents) > 3:
                print(f"  ... and {len(agents) - 3} more")

    def test_model_distribution(self) -> None:
        """Analyze model selection across agents.

        Prints distribution of models (sonnet, opus, haiku).
        """
        print("\n" + "=" * 80)
        print("MODEL DISTRIBUTION")
        print("=" * 80)

        model_counts = {}
        model_agents: Dict[str, List[str]] = {}

        for agent in self.agents:
            model = agent.model
            model_counts[model] = model_counts.get(model, 0) + 1

            if model not in model_agents:
                model_agents[model] = []
            model_agents[model].append(agent.name)

        for model in sorted(model_counts.keys()):
            count = model_counts[model]
            percentage = (count / len(self.agents) * 100) if self.agents else 0

            print(f"\n{model}: {count} agent(s) ({percentage:.1f}%)")
            for agent_name in model_agents[model][:5]:
                print(f"  - {agent_name}")
            if len(model_agents[model]) > 5:
                print(f"  ... and {len(model_agents[model]) - 5} more")

    def print_summary(self) -> None:
        """Print overall summary."""
        print("\n" + "=" * 80)
        print("LOADING TEST SUMMARY")
        print("=" * 80)
        print(f"Agents directory: {self.agents_dir}")
        print(f"Agents discovered: {len(self.agents)}")
        print(f"Errors encountered: {len(self.errors)}")

        if self.errors:
            print("\nErrors:")
            for error in self.errors:
                print(f"  - {error}")

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

    print(f"Testing agent loading from: {agents_dir}\n")

    tester = AgentLoadingTester(agents_dir)

    # Run tests
    agents = tester.discover_agents()

    if not agents:
        print("\nNo agents discovered - cannot run further tests")
        return 1

    tester.test_activation_patterns()
    tester.test_hierarchy_coverage()
    tester.test_tool_usage()
    tester.test_model_distribution()
    tester.print_summary()

    return 1 if tester.errors else 0


if __name__ == "__main__":
    sys.exit(main())
