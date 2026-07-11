"""Basic AnyTensor usage example.

Demonstrates creation operations and basic tensor manipulation.
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import (
    zeros,
    ones,
    full,
    arange,
    eye,
    linspace,
)


def main() raises:
    """Run basic AnyTensor examples."""
    print("AnyTensor Basic Usage Examples")
    print("=" * 50)

    # Create tensors with different shapes
    print("\n1. Creation Operations:")
    print("-" * 50)

    # zeros
    var shape_2d = List[Int]()
    var z = zeros(shape_2d, DType.float32)
    print("Created zeros tensor with shape (3, 4)")
    print("  - numel:", z.numel())
    print("  - dim:", z.dim())
    print("  - dtype:", z.dtype())
    print("  - is_contiguous:", z.is_contiguous())

    # ones
    var o = ones(shape_2d, DType.float32)
    print("\nCreated ones tensor with shape (3, 4)")
    print("  - numel:", o.numel())

    # full
    var f = full(shape_2d, 42.0, DType.float32)
    print("\nCreated full tensor with value 42.0 and shape (3, 4)")
    print("  - numel:", f.numel())

    # arange
    var a = arange(0.0, 10.0, 1.0, DType.float32)
    print("\nCreated arange tensor [0, 10) with step 1")
    print("  - numel:", a.numel())
    print("  - shape:", a.shape()[0])

    # eye
    var e = eye(3, 3, 0, DType.float32)
    print("\nCreated 3x3 identity matrix")
    print("  - numel:", e.numel())

    # linspace
    var l = linspace(0.0, 1.0, 11, DType.float32)
    print("\nCreated linspace tensor [0.0, 1.0] with 11 points")
    print("  - numel:", l.numel())

    # Different dtypes
    print("\n2. Different Data Types:")
    print("-" * 50)

    var shape_1d = List[Int]()

    var float16_tensor = zeros(shape_1d, DType.float16)
    print("float16 tensor:", float16_tensor.dtype())

    var float64_tensor = zeros(shape_1d, DType.float64)
    print("float64 tensor:", float64_tensor.dtype())

    var int32_tensor = zeros(shape_1d, DType.int32)
    print("int32 tensor:", int32_tensor.dtype())

    var uint8_tensor = zeros(shape_1d, DType.uint8)
    print("uint8 tensor:", uint8_tensor.dtype())

    var bool_tensor = zeros(shape_1d, DType.bool)
    print("bool tensor:", bool_tensor.dtype())

    # Different dimensions
    print("\n3. Different Dimensions:")
    print("-" * 50)

    var shape_0d = List[Int]()
    var scalar = zeros(shape_0d, DType.float32)
    print("0D scalar:", scalar.dim(), "dimensions, numel:", scalar.numel())

    var shape_1d_test = List[Int]()
    var vector = zeros(shape_1d_test, DType.float32)
    print("1D vector:", vector.dim(), "dimensions, numel:", vector.numel())

    var shape_2d_test = List[Int]()
    var matrix = zeros(shape_2d_test, DType.float32)
    print("2D matrix:", matrix.dim(), "dimensions, numel:", matrix.numel())

    var shape_3d = List[Int]()
    var tensor_3d = zeros(shape_3d, DType.float32)
    print(
        "3D tensor:", tensor_3d.dim(), "dimensions, numel:", tensor_3d.numel()
    )

    var shape_4d = List[Int]()
    var tensor_4d = zeros(shape_4d, DType.float32)
    print(
        "4D tensor:", tensor_4d.dim(), "dimensions, numel:", tensor_4d.numel()
    )

    print("\n" + "=" * 50)
    print("All examples completed successfully!")
