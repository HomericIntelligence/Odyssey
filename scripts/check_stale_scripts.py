#!/usr/bin/env python3
"""Detect scripts/*.py files with no references in .github/, justfile, or other scripts/."""
# Thin re-export wrapper — functionality moved to hephaestus.
# Remove in next release cycle after consumers are updated.
# See: HomericIntelligence/ProjectHephaestus v0.7.0
from hephaestus.validation.stale_scripts import *  # noqa: F401,F403

if __name__ == "__main__":
    import sys
    from hephaestus.validation.stale_scripts import main
    sys.exit(main())
