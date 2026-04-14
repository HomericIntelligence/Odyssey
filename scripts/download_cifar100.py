#!/usr/bin/env python3
"""Download CIFAR-100 dataset for ML Odyssey."""
# Thin re-export wrapper — functionality moved to hephaestus.
# Remove in next release cycle after consumers are updated.
# See: HomericIntelligence/ProjectHephaestus v0.7.0
from hephaestus.datasets.downloader import CIFAR100Downloader, main  # noqa: F401

if __name__ == "__main__":
    import sys
    from hephaestus.datasets.downloader import main
    sys.exit(main())
