#!/usr/bin/env python3
"""Check workflow inventory drift between .github/workflows/*.yml files and README.md table."""
# Thin re-export wrapper — functionality moved to hephaestus.
# Remove in next release cycle after consumers are updated.
# See: HomericIntelligence/ProjectHephaestus v0.7.0
from hephaestus.ci.workflows import *  # noqa: F401,F403

if __name__ == "__main__":
    import sys
    from hephaestus.ci.workflows import check_workflow_inventory_main as main
    sys.exit(main())
