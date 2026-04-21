#!/usr/bin/env python3
"""List all available agents."""
# Thin re-export wrapper — functionality moved to hephaestus.
# Remove in next release cycle after consumers are updated.
# See: HomericIntelligence/ProjectHephaestus v0.7.0
from hephaestus.agents.loader import *  # noqa: F401,F403


def main() -> None:
    """List all available agents by loading from hephaestus.agents.loader."""
    from pathlib import Path

    from hephaestus.agents.loader import load_all_agents

    # Resolve the .claude/agents directory relative to this script's repo root
    repo_root = Path(__file__).resolve().parent.parent.parent
    agents_dir = repo_root / ".claude" / "agents"
    agents = load_all_agents(agents_dir)
    for agent in agents:
        print(agent)


if __name__ == "__main__":
    main()
