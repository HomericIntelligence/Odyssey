#!/usr/bin/env python3
"""Synchronize requirements*.txt from pixi.toml resolved versions."""
# Thin re-export wrapper — functionality moved to hephaestus.
# Remove in next release cycle after consumers are updated.
# See: HomericIntelligence/ProjectHephaestus v0.7.0
from hephaestus.config.dep_sync import *  # noqa: F401,F403

if __name__ == "__main__":
    import sys
    from hephaestus.config.dep_sync import main
    sys.exit(main())
