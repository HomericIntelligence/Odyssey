#!/usr/bin/env python3
"""Shared utilities for agent configuration scripts."""

# Thin re-export wrapper — functionality moved to hephaestus.
# Remove in next release cycle after consumers are updated.
# See: HomericIntelligence/ProjectHephaestus v0.7.0
from hephaestus.agents import *  # noqa: F401,F403
