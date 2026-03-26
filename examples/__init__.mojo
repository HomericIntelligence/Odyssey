"""Examples package for ML Odyssey models.

This package contains example scripts demonstrating ML Odyssey functionality:
- basic_usage.mojo          - Basic tensor operations and model usage
- fp8_example.mojo          - FP8 data type usage and conversion
- bf8_example.mojo          - BF8 data type usage and conversion
- integer_example.mojo      - Integer tensor operations
- mixed_precision_training.mojo - Multi-precision training
- data_pipeline_demo.mojo   - Data loading and preprocessing
- alexnet_cifar10/          - AlexNet training on CIFAR-10
- mojo_patterns/            - Mojo language pattern demonstrations

Run individual examples directly:
    mojo examples/basic_usage.mojo
    mojo examples/fp8_example.mojo
"""


fn main() raises:
    """Examples package entry point.

    This serves as a placeholder entry point for the examples package.
    Individual example files have their own main() functions and should
    be run directly:

        mojo examples/basic_usage.mojo
        mojo examples/fp8_example.mojo
        mojo examples/data_pipeline_demo.mojo

    Future enhancement: Add interactive menu to select and run examples.
    """
    print("\n=== ML Odyssey Examples ===\n")
    print("Available examples:")
    print("  1. basic_usage.mojo          - Basic tensor operations")
    print("  2. fp8_example.mojo          - FP8 data type usage")
    print("  3. bf8_example.mojo          - BF8 data type usage")
    print("  4. integer_example.mojo      - Integer tensor operations")
    print("  5. mixed_precision_training.mojo - Multi-precision training")
    print("  6. data_pipeline_demo.mojo   - Data pipeline demonstration")
    print("  7. alexnet_cifar10/          - AlexNet on CIFAR-10")
    print("  8. mojo_patterns/            - Mojo language patterns")
    print("\nRun examples directly:")
    print("    mojo examples/<example_name>.mojo")
    print()
