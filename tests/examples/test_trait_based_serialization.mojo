"""Tests for trait-based layer serialization.

Tests serialization of layers implementing the Serializable trait:
- Named tensor collection save/load
- Parameter persistence through checkpoint format
- Round-trip integrity (save -> load -> verify)

Runs as: pixi run mojo ./tests/examples/test_trait_based_serialization.mojo
"""

from shared.core.extensor import AnyTensor, zeros, ones
from shared.utils.serialization import (
    save_named_tensors,
    load_named_tensors,
    NamedTensor,
)


fn test_simple_named_tensor_roundtrip() raises:
    """Test simple save/load of a single named tensor."""
    print("\n" + "=" * 80)
    print("Test: Simple Named Tensor Roundtrip")
    print("=" * 80)

    # Create a simple tensor
    var shape = List[Int]()
    shape.append(8)
    var weights = ones(shape, DType.float32)

    # Set custom values
    for i in range(weights.numel()):
        weights._set_float64(i, 1.5)

    print("✓ Created tensor with shape:", String(weights.numel()))

    # Save
    var tensors: List[NamedTensor] = []
    tensors.append(NamedTensor("weights", weights))

    var path = "/tmp/test_simple/"
    save_named_tensors(tensors, path)
    print("✓ Saved tensor")

    # Load
    var loaded = load_named_tensors(path)
    print("✓ Loaded", String(len(loaded)), "tensors")

    # Verify
    if len(loaded) == 1:
        var tensor = loaded[0].tensor
        var all_match = True
        for i in range(tensor.numel()):
            if tensor._get_float64(i) != 1.5:
                all_match = False
                print(
                    "  Value mismatch at index",
                    String(i),
                    ":",
                    String(tensor._get_float64(i)),
                )
                break

        if all_match:
            print("✓ All values match correctly")
        else:
            raise Error("Values don't match")
    else:
        raise Error("Unexpected number of tensors loaded")

    print()


fn test_multiple_named_tensors_roundtrip() raises:
    """Test saving and loading multiple named tensors."""
    print("Test: Multiple Named Tensors Roundtrip")
    print("-" * 40)

    # Create test tensors
    var shape = List[Int]()
    shape.append(4)
    var gamma = ones(shape, DType.float32)
    var beta = zeros(shape, DType.float32)

    # Set values
    for i in range(gamma.numel()):
        gamma._set_float64(i, 0.5)
    for i in range(beta.numel()):
        beta._set_float64(i, 0.25)

    print("✓ Created 2 tensors")

    # Save
    var tensors: List[NamedTensor] = []
    tensors.append(NamedTensor("gamma", gamma))
    tensors.append(NamedTensor("beta", beta))

    var path = "/tmp/test_multi/"
    save_named_tensors(tensors, path)
    print("✓ Saved 2 named tensors")

    # Load
    var loaded = load_named_tensors(path)
    print("✓ Loaded", String(len(loaded)), "tensors")

    # Verify count
    if len(loaded) != 2:
        raise Error("Expected 2 tensors, got " + String(len(loaded)))

    # Verify each tensor
    for i in range(len(loaded)):
        var name = loaded[i].name
        var tensor = loaded[i].tensor

        if name == "gamma":
            var gamma_match = True
            for j in range(tensor.numel()):
                if tensor._get_float64(j) != 0.5:
                    gamma_match = False
                    break
            if gamma_match:
                print("✓ gamma values correct")
            else:
                raise Error("gamma values mismatch")

        elif name == "beta":
            var beta_match = True
            for j in range(tensor.numel()):
                if tensor._get_float64(j) != 0.25:
                    beta_match = False
                    break
            if beta_match:
                print("✓ beta values correct")
            else:
                raise Error("beta values mismatch")

    print()


fn test_different_shapes_roundtrip() raises:
    """Test saving/loading tensors with different shapes."""
    print("Test: Different Shapes Roundtrip")
    print("-" * 40)

    # Create tensors with different shapes
    var shape_1d = List[Int]()
    shape_1d.append(5)
    var tensor_1d = ones(shape_1d, DType.float32)

    var shape_2d = List[Int]()
    shape_2d.append(2)
    shape_2d.append(3)
    var tensor_2d = ones(shape_2d, DType.float32)

    print(
        "✓ Created tensors with shapes:",
        String(tensor_1d.numel()),
        ",",
        String(tensor_2d.numel()),
    )

    # Save
    var tensors: List[NamedTensor] = []
    tensors.append(NamedTensor("vector", tensor_1d))
    tensors.append(NamedTensor("matrix", tensor_2d))

    var path = "/tmp/test_shapes/"
    save_named_tensors(tensors, path)
    print("✓ Saved tensors")

    # Load
    var loaded = load_named_tensors(path)
    print("✓ Loaded tensors")

    # Verify shapes
    if len(loaded) == 2:
        for i in range(len(loaded)):
            var name = loaded[i].name
            var shape = loaded[i].tensor.shape()
            if name == "vector":
                if len(shape) == 1 and shape[0] == 5:
                    print("✓ vector shape correct (1D)")
                else:
                    raise Error("vector shape incorrect")
            elif name == "matrix":
                if len(shape) == 2 and shape[0] == 2 and shape[1] == 3:
                    print("✓ matrix shape correct (2D)")
                else:
                    raise Error("matrix shape incorrect")
    else:
        raise Error("Wrong number of tensors")

    print()


fn main() raises:
    """Run all named tensor serialization tests."""
    print("\n" + "#" * 80)
    print("# Named Tensor Serialization Tests")
    print("#" * 80)

    test_simple_named_tensor_roundtrip()
    test_multiple_named_tensors_roundtrip()
    test_different_shapes_roundtrip()

    print("=" * 80)
    print("Summary: All named tensor serialization tests passed!")
    print("=" * 80 + "\n")
