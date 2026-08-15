#!/usr/bin/env python3
"""
Download and prepare CIFAR-10 dataset for ML Odyssey examples.

This is a wrapper script that imports the shared download utility.
For the implementation, see hephaestus.datasets.downloader.
"""

from hephaestus.datasets.downloader import main

if __name__ == "__main__":
    main()
