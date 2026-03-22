"""Example demonstrating FP8 data type usage.

This example shows:
1. Creating FP8 values from Float32
2. Converting FP8 back to Float32
3. Converting tensors to/from FP8 format
4. Memory savings with FP8 (8-bit vs 32-bit)
"""

from shared.core import AnyTensor, zeros


fn main() raises:
    print("\n=== FP8 Data Type Example ===\n")
    print(
        "NOTE: FP8 support is not yet implemented in the shared library."
    )
    print("This example demonstrates the expected API structure.")
    print(
        "When FP8 conversion methods are available in AnyTensor,"
    )
    print("this example can be fully implemented.")
    print("")
