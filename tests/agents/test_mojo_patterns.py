#!/usr/bin/env python3
"""
Test Mojo-specific patterns and guidelines in agent configurations.

This script validates:
- Mojo-specific guidelines presence in relevant agents
- fn vs def guidance
- struct vs class guidance
- SIMD optimization context
- Memory management patterns (owned, borrowed)
- Performance optimization guidance

Usage:
    python3 tests/agents/test_mojo_patterns.py [agent_dir]
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
class MojoPatternInfo:
    """Information about Mojo patterns in an agent."""

    agent_name: str
    level: int
    is_implementation_agent: bool
    patterns_found: Set[str]
    pattern_details: Dict[str, List[str]]
    completeness_score: float


class MojoPatternTester:
    """Test Mojo-specific patterns in agent configurations."""

    # Mojo patterns to check for
    MOJO_PATTERNS = {
        "fn_vs_def": {
            "keywords": ["fn vs def", "fn versus def", "`fn`", "`def`"],
            "description": "Function definition guidance (fn vs def)",
        },
        "struct_vs_class": {
            "keywords": [
                "struct vs class",
                "struct versus class",
                "`struct`",
                "`class`",
            ],
            "description": "Type definition guidance (struct vs class)",
        },
        "simd": {
            "keywords": ["simd", "vectorization", "vectorize", "vector"],
            "description": "SIMD/vectorization optimization",
        },
        "memory_management": {
            "keywords": ["owned", "borrowed", "inout", "lifetime"],
            "description": "Memory management (owned, borrowed, inout)",
        },
        "performance": {
            "keywords": ["@parameter", "compile-time", "performance", "optimization"],
            "description": "Performance optimization",
        },
        "type_safety": {
            "keywords": ["type safety", "type-safe", "raises", "error handling"],
            "description": "Type safety and error handling",
        },
        "traits": {
            "keywords": ["trait", "protocol", "interface"],
            "description": "Traits/protocols usage",
        },
    }

    def __init__(self, agents_dir: Path):
        """Initialize tester.

        Args:
            agents_dir: Directory containing agent .md files
        """
        self.agents_dir = agents_dir
        self.pattern_info: List[MojoPatternInfo] = []
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def analyze_all(self) -> List[MojoPatternInfo]:
        """Analyze Mojo patterns in all agents.

        Returns:
            List of Mojo pattern information
        """
        if not self.agents_dir.exists():
            self.errors.append(f"Agents directory not found: {self.agents_dir}")
            return []

        agent_files = sorted(self.agents_dir.glob("*.md"))

        if not agent_files:
            self.errors.append(f"No agent files found in {self.agents_dir}")
            return []

        print(f"Analyzing Mojo patterns in {len(agent_files)} agents...")

        for agent_file in agent_files:
            # Check file size before reading
            file_size = agent_file.stat().st_size
            if file_size > MAX_FILE_SIZE:
                logging.warning(f"Skipping {agent_file.name} (file too large: {file_size} bytes)")
                self.errors.append(f"{agent_file.name}: File too large ({file_size} bytes)")
                continue

            info = self._analyze_mojo_patterns(agent_file)
            if info:
                self.pattern_info.append(info)

        return self.pattern_info

    def _analyze_mojo_patterns(self, file_path: Path) -> MojoPatternInfo:
        """Analyze Mojo patterns in a single agent.

        Args:
            file_path: Path to agent .md file

        Returns:
            MojoPatternInfo
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
        is_implementation_agent = self._is_implementation_agent(agent_name, level)

        # Analyze patterns
        patterns_found = set()
        pattern_details = {}

        content_lower = content.lower()

        for pattern_name, pattern_config in self.MOJO_PATTERNS.items():
            keywords = pattern_config["keywords"]
            found_keywords = []

            for keyword in keywords:
                if keyword.lower() in content_lower:
                    patterns_found.add(pattern_name)
                    found_keywords.append(keyword)

            if found_keywords:
                pattern_details[pattern_name] = found_keywords

        # Calculate completeness score
        if is_implementation_agent:
            # Implementation agents should have most patterns
            expected_patterns = {
                "fn_vs_def",
                "struct_vs_class",
                "memory_management",
                "type_safety",
            }
            completeness_score = len(patterns_found & expected_patterns) / len(expected_patterns)
        else:
            # Non-implementation agents may have fewer patterns
            completeness_score = 1.0 if patterns_found else 0.5

        return MojoPatternInfo(
            agent_name=agent_name,
            level=level if level is not None else -1,
            is_implementation_agent=is_implementation_agent,
            patterns_found=patterns_found,
            pattern_details=pattern_details,
            completeness_score=completeness_score,
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

    def _is_implementation_agent(self, name: str, level: int) -> bool:
        """Check if this is an implementation-level agent.

        Implementation agents (levels 3-5) should have comprehensive Mojo guidance.

        Args:
            name: Agent name
            level: Agent level

        Returns:
            True if implementation agent
        """
        if level in [3, 4, 5]:
            return True

        impl_keywords = ["engineer", "specialist", "implementation", "writer"]
        return any(keyword in name.lower() for keyword in impl_keywords)

    def test_pattern_coverage(self) -> None:
        """Test Mojo pattern coverage across agents."""
        print("\n" + "=" * 80)
        print("MOJO PATTERN COVERAGE")
        print("=" * 80)

        # Overall statistics
        total_agents = len(self.pattern_info)
        impl_agents = [info for info in self.pattern_info if info.is_implementation_agent]
        non_impl_agents = [info for info in self.pattern_info if not info.is_implementation_agent]

        print(f"\nTotal agents: {total_agents}")
        print(f"Implementation agents: {len(impl_agents)}")
        print(f"Non-implementation agents: {len(non_impl_agents)}")

        # Pattern coverage by pattern type
        print("\n\nPattern Coverage by Type:")
        for pattern_name, pattern_config in self.MOJO_PATTERNS.items():
            agents_with_pattern = [info for info in self.pattern_info if pattern_name in info.patterns_found]

            impl_with_pattern = [info for info in impl_agents if pattern_name in info.patterns_found]

            print(f"\n{pattern_config['description']}:")
            print(f"  Total agents: {len(agents_with_pattern)}/{total_agents}")
            print(f"  Implementation agents: {len(impl_with_pattern)}/{len(impl_agents)}")

            if len(impl_agents) > 0:
                coverage = len(impl_with_pattern) / len(impl_agents) * 100
                if coverage < 50:
                    self.warnings.append(f"Low coverage for '{pattern_name}' in implementation agents: {coverage:.1f}%")

    def test_implementation_agent_patterns(self) -> None:
        """Test that implementation agents have comprehensive Mojo guidance."""
        print("\n" + "=" * 80)
        print("IMPLEMENTATION AGENT MOJO GUIDANCE")
        print("=" * 80)

        impl_agents = [info for info in self.pattern_info if info.is_implementation_agent]

        if not impl_agents:
            print("\nNo implementation agents found")
            return

        # Critical patterns for implementation agents
        critical_patterns = {
            "fn_vs_def",
            "struct_vs_class",
            "memory_management",
            "type_safety",
        }

        for info in impl_agents:
            print(f"\n{info.agent_name} (Level {info.level}):")
            print(f"  Completeness score: {info.completeness_score:.1%}")

            # Check critical patterns
            found_critical = info.patterns_found & critical_patterns
            missing_critical = critical_patterns - info.patterns_found

            if found_critical:
                print(f"  Critical patterns found: {', '.join(sorted(found_critical))}")

            if missing_critical:
                print(f"  ⚠ Missing critical patterns: {', '.join(sorted(missing_critical))}")
                self.warnings.append(f"{info.agent_name} missing critical Mojo patterns: {', '.join(missing_critical)}")

            # Show all patterns
            if info.patterns_found:
                print(f"  All patterns found ({len(info.patterns_found)}):")
                for pattern in sorted(info.patterns_found):
                    desc = self.MOJO_PATTERNS[pattern]["description"]
                    print(f"    - {desc}")
            else:
                print("  ⚠ No Mojo patterns found!")
                self.warnings.append(f"{info.agent_name} has no Mojo-specific guidance")

    def test_fn_vs_def_guidance(self) -> None:
        """Test fn vs def guidance specifically."""
        print("\n" + "=" * 80)
        print("FN VS DEF GUIDANCE")
        print("=" * 80)

        agents_with_guidance = [info for info in self.pattern_info if "fn_vs_def" in info.patterns_found]

        print(f"\nAgents with fn vs def guidance: {len(agents_with_guidance)}")

        for info in agents_with_guidance:
            print(f"\n{info.agent_name} (Level {info.level}):")
            keywords = info.pattern_details.get("fn_vs_def", [])
            print(f"  Keywords found: {', '.join(keywords)}")

        # Check implementation agents specifically
        impl_agents = [info for info in self.pattern_info if info.is_implementation_agent]
        impl_with_guidance = [info for info in impl_agents if "fn_vs_def" in info.patterns_found]

        if impl_agents:
            coverage = len(impl_with_guidance) / len(impl_agents) * 100
            print(f"\nImplementation agent coverage: {len(impl_with_guidance)}/{len(impl_agents)} ({coverage:.1f}%)")

            if coverage < 80:
                self.warnings.append(f"Low fn vs def guidance coverage in implementation agents: {coverage:.1f}%")

    def test_simd_optimization_guidance(self) -> None:
        """Test SIMD optimization guidance."""
        print("\n" + "=" * 80)
        print("SIMD OPTIMIZATION GUIDANCE")
        print("=" * 80)

        agents_with_simd = [info for info in self.pattern_info if "simd" in info.patterns_found]

        print(f"\nAgents with SIMD guidance: {len(agents_with_simd)}")

        for info in agents_with_simd:
            print(f"\n{info.agent_name} (Level {info.level}):")
            keywords = info.pattern_details.get("simd", [])
            print(f"  Keywords found: {', '.join(keywords)}")

        # SIMD is especially important for performance-critical agents
        perf_agents = [
            info
            for info in self.pattern_info
            if "performance" in info.agent_name.lower() or "optimization" in info.agent_name.lower()
        ]

        if perf_agents:
            perf_with_simd = [info for info in perf_agents if "simd" in info.patterns_found]
            print(f"\nPerformance-focused agents with SIMD: {len(perf_with_simd)}/{len(perf_agents)}")

            for info in perf_agents:
                if "simd" not in info.patterns_found:
                    self.warnings.append(f"{info.agent_name} is performance-focused but lacks SIMD guidance")

    def test_memory_management_guidance(self) -> None:
        """Test memory management guidance."""
        print("\n" + "=" * 80)
        print("MEMORY MANAGEMENT GUIDANCE")
        print("=" * 80)

        agents_with_mem = [info for info in self.pattern_info if "memory_management" in info.patterns_found]

        print(f"\nAgents with memory management guidance: {len(agents_with_mem)}")

        for info in agents_with_mem:
            print(f"\n{info.agent_name} (Level {info.level}):")
            keywords = info.pattern_details.get("memory_management", [])
            print(f"  Keywords found: {', '.join(keywords)}")

        # Check implementation engineers specifically
        impl_engineers = [info for info in self.pattern_info if info.level in [4, 5]]

        if impl_engineers:
            engineers_with_mem = [info for info in impl_engineers if "memory_management" in info.patterns_found]

            coverage = len(engineers_with_mem) / len(impl_engineers) * 100
            print(
                f"\nImplementation engineer coverage: {len(engineers_with_mem)}/{len(impl_engineers)} ({coverage:.1f}%)"
            )

    def print_summary(self) -> None:
        """Print test summary."""
        print("\n" + "=" * 80)
        print("MOJO PATTERNS TEST SUMMARY")
        print("=" * 80)

        impl_agents = [info for info in self.pattern_info if info.is_implementation_agent]

        if impl_agents:
            avg_completeness = sum(info.completeness_score for info in impl_agents) / len(impl_agents)
            print(f"Implementation agents average completeness: {avg_completeness:.1%}")

            high_quality = [info for info in impl_agents if info.completeness_score >= 0.75]
            print(f"High quality (>=75%): {len(high_quality)}/{len(impl_agents)}")

            low_quality = [info for info in impl_agents if info.completeness_score < 0.5]
            print(f"Low quality (<50%): {len(low_quality)}/{len(impl_agents)}")

            if low_quality:
                print("\nAgents needing improvement:")
                for info in low_quality:
                    print(f"  - {info.agent_name} ({info.completeness_score:.1%})")

        print(f"\nTotal errors: {len(self.errors)}")
        print(f"Total warnings: {len(self.warnings)}")

        if self.errors:
            print("\nErrors:")
            for error in self.errors:
                print(f"  - {error}")

        if self.warnings:
            print("\nWarnings (first 10):")
            for warning in self.warnings[:10]:
                print(f"  - {warning}")
            if len(self.warnings) > 10:
                print(f"  ... and {len(self.warnings) - 10} more")

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

    print(f"Testing Mojo patterns in: {agents_dir}\n")

    tester = MojoPatternTester(agents_dir)
    pattern_info = tester.analyze_all()

    if not pattern_info:
        print("\nNo pattern information found - cannot run tests")
        return 1

    tester.test_pattern_coverage()
    tester.test_implementation_agent_patterns()
    tester.test_fn_vs_def_guidance()
    tester.test_simd_optimization_guidance()
    tester.test_memory_management_guidance()
    tester.print_summary()

    return 1 if tester.errors else 0


if __name__ == "__main__":
    sys.exit(main())
