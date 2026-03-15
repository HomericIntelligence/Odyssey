#!/usr/bin/env python3
"""
Download and prepare CIFAR-10 dataset for ML Odyssey examples.

This is a wrapper script that imports the shared download utility.
For the implementation, see scripts/download_cifar10.py.
"""

import sys
from pathlib import Path

# Add scripts directory to path for import
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

# noqa: E402 - import must follow sys.path modification
from download_cifar10 import main  # noqa: E402

if __name__ == "__main__":
    main()
