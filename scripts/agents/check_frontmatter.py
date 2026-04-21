#!/usr/bin/env python3
"""Check YAML frontmatter in agent configuration files."""

# Thin re-export wrapper — functionality moved to hephaestus.
# Remove in next release cycle after consumers are updated.
# See: HomericIntelligence/ProjectHephaestus v0.7.0
from hephaestus.agents.frontmatter import *  # noqa: F401,F403

if __name__ == "__main__":
    import sys
    from hephaestus.agents.frontmatter import validate_agents_main

    sys.exit(validate_agents_main())
