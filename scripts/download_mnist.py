#!/usr/bin/env python3
"""Download MNIST dataset for ML Odyssey."""
# Thin re-export wrapper — functionality moved to hephaestus.
# Remove in next release cycle after consumers are updated.
# See: HomericIntelligence/ProjectHephaestus v0.7.0
from hephaestus.datasets.downloader import MNISTDownloader, main  # noqa: F401

if __name__ == "__main__":
    from hephaestus.datasets.downloader import main
    main()
