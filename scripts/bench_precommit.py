#!/usr/bin/env python3
"""Pre-commit hook performance benchmark helper."""
# Thin re-export wrapper — functionality moved to hephaestus.
# Remove in next release cycle after consumers are updated.
# See: HomericIntelligence/ProjectHephaestus v0.7.0
from hephaestus.ci.precommit import *  # noqa: F401,F403

if __name__ == "__main__":
    import sys
    from hephaestus.ci.precommit import bench_precommit_main as main
    sys.exit(main())
