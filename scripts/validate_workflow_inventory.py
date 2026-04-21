#!/usr/bin/env python3
"""Validate that GitHub Actions workflows match documentation."""
# Thin re-export wrapper — functionality moved to hephaestus.
# Remove in next release cycle after consumers are updated.
# See: HomericIntelligence/ProjectHephaestus v0.7.0
from hephaestus.ci.workflows import *  # noqa: F401,F403

if __name__ == "__main__":
    import sys
    from hephaestus.ci.workflows import validate_workflow_checkout_main as main
    sys.exit(main())
