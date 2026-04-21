#!/usr/bin/env python3
"""Check pre-commit version consistency against pixi.toml."""

# Thin re-export wrapper — functionality moved to hephaestus.
# Remove in next release cycle after consumers are updated.
# See: HomericIntelligence/ProjectHephaestus v0.7.0
from hephaestus.ci.precommit import *  # noqa: F401,F403

if __name__ == "__main__":
    import sys
    from hephaestus.ci.precommit import check_precommit_versions_main as main

    sys.exit(main())
