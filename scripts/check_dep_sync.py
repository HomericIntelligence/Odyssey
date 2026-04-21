#!/usr/bin/env python3
"""Validate dependency consistency across project configuration files."""
# Thin re-export wrapper — functionality moved to hephaestus.
# Remove in next release cycle after consumers are updated.
# See: HomericIntelligence/ProjectHephaestus v0.7.0
from hephaestus.config.dep_sync import *  # noqa: F401,F403

if __name__ == "__main__":
    import sys
    from hephaestus.config.dep_sync import check_dep_sync_main as main
    sys.exit(main())
