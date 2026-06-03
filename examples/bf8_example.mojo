"""Example demonstrating BF8 data type usage.

This example shows:
1. Creating BF8 values from Float32
2. Converting BF8 back to Float32
3. Converting tensors to/from BF8 format
4. Memory savings with BF8 (8-bit vs 32-bit)
5. Comparing BF8 (E5M2) vs FP8 (E4M3) characteristics
"""

from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.tensor.tensor_creation import zeros


def main() raises:
    print("\n=== BF8 Data Type Example ===\n")
    print("BF8 support is not yet implemented in the shared library.")
    print("This example demonstrates the expected API structure.")
    print("When BF8 conversion methods are available in AnyTensor,")
    print("this example can be fully implemented.")
    print("")
