"""Example: Mojo Patterns - SIMD Optimization

This example demonstrates vectorized operations using SIMD.

Usage:
    pixi run mojo run examples/mojo_patterns/simd_example.mojo

See documentation: docs/core/mojo-patterns.md
"""

from std.algorithm import vectorize
from sys.info import simd_width_of
from shared.tensor.any_tensor import AnyTensor


def simple_simd_add(tensor1: AnyTensor, tensor2: AnyTensor) raises -> AnyTensor:
    """Demonstrate SIMD addition using vectorize.

    Args:
        tensor1: First input tensor.
        tensor2: Second input tensor.

    Returns:
        Result of element-wise addition.
    """
    if tensor1.numel() != tensor2.numel():
        raise Error("Tensors must have same number of elements")

    var result = AnyTensor(tensor1._shape, tensor1._dtype)

    if tensor1._dtype == DType.float32:
        comptime simd_width = simd_width_of[DType.float32]()
        var size = tensor1.numel()
        var ptr1 = tensor1._data.bitcast[Float32]()
        var ptr2 = tensor2._data.bitcast[Float32]()
        var out_ptr = result._data.bitcast[Float32]()

        @parameter
        def vectorized_add[width: Int](idx: Int) unified {mut}:
            var a = ptr1.load[width=width](idx)
            var b = ptr2.load[width=width](idx)
            out_ptr.store[width=width](idx, a + b)

        vectorize[simd_width](size, vectorized_add)

    return result^


def main() raises:
    """Demonstrate SIMD optimization."""
    print("SIMD Optimization Examples")
    print("=" * 50)

    # Example 1: SIMD-accelerated element-wise addition
    print("\n1. SIMD Element-wise Addition:")
    print("-" * 50)

    # Create two tensors
    var data1 = List[Float32]()
    data1.append(1.0)
    data1.append(2.0)
    data1.append(3.0)
    data1.append(4.0)
    var tensor1 = AnyTensor(data1^)

    var data2 = List[Float32]()
    data2.append(10.0)
    data2.append(20.0)
    data2.append(30.0)
    data2.append(40.0)
    var tensor2 = AnyTensor(data2^)

    print("Input tensors: 4 elements each")
    print("Using SIMD width for float32:", simd_width_of[DType.float32]())

    # Apply SIMD addition
    var result = simple_simd_add(tensor1, tensor2)
    print("SIMD addition completed")
    print("Output shape:", result.shape()[0], "elements")

    # Example 2: SIMD width information
    print("\n2. SIMD Widths by DType:")
    print("-" * 50)
    print("  float32:", simd_width_of[DType.float32](), "elements per vector")
    print("  float64:", simd_width_of[DType.float64](), "elements per vector")
    print("  int32:  ", simd_width_of[DType.int32](), "elements per vector")
    print("  int64:  ", simd_width_of[DType.int64](), "elements per vector")

    # Example 3: Performance benefits explanation
    print("\n3. SIMD Performance Benefits:")
    print("-" * 50)
    print("SIMD vectorization provides:")
    print("  - Process multiple elements in a single CPU instruction")
    print("  - ~4x speedup for float32 on modern CPUs (AVX2/AVX-512)")
    print("  - ~2x speedup for float64 (half SIMD width of float32)")
    print("  - Compile-time width selection based on dtype")
    print("  - Better cache utilization and memory bandwidth")

    print("\n" + "=" * 50)
    print("SIMD example complete!")
