#!/usr/bin/env python3
"""List all available agents."""
# Thin re-export wrapper — functionality moved to hephaestus.
# Remove in next release cycle after consumers are updated.
# See: HomericIntelligence/ProjectHephaestus v0.7.0
from hephaestus.agents.loader import *  # noqa: F401,F403

if __name__ == "__main__":
    import sys
    from hephaestus.agents.loader import main
    sys.exit(main())
