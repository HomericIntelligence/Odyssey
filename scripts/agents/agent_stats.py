#!/usr/bin/env python3
"""agent_stats.py - Generate usage statistics for the agent system."""

# Thin re-export wrapper — functionality moved to hephaestus.
# Remove in next release cycle after consumers are updated.
# See: HomericIntelligence/ProjectHephaestus v0.7.0
from hephaestus.agents.stats import *  # noqa: F401,F403

if __name__ == "__main__":
    import sys
    from hephaestus.agents.stats import main

    sys.exit(main())
