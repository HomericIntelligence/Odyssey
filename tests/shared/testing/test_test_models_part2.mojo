"""Tests for SimpleMLP initialization and forward pass.

Split from test_test_models.mojo per ADR-009 to avoid Mojo heap corruption.

Coverage:
    - SimpleMLP initialization (1 and 2 hidden layers)
    - SimpleMLP forward pass (List[Float32] and ExTensor inputs)
"""

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_test_models.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

from shared.testing import (
    SimpleMLP,
    assert_true,
    assert_equal,
)
from shared.core import (
    ExTensor,
    zeros,
    ones,
)


# ============================================================================
# SimpleMLP Initialization Tests
# ============================================================================


fn test_simple_mlp_initialization_1_hidden() raises:
    """Test SimpleMLP with 1 hidden layer."""
    var mlp = SimpleMLP(10, 20, 5, num_hidden_layers=1)
    assert_equal(mlp.input_dim, 10)
    assert_equal(mlp.hidden_dim, 20)
    assert_equal(mlp.output_dim, 5)
    assert_equal(mlp.num_hidden_layers, 1)

    # Check weight dimensions
    assert_equal(len(mlp.layer1_weights), 10 * 20)
    assert_equal(len(mlp.layer1_bias), 20)
    assert_equal(len(mlp.layer2_weights), 20 * 5)
    assert_equal(len(mlp.layer2_bias), 5)
    assert_equal(len(mlp.layer3_weights), 0)
    assert_equal(len(mlp.layer3_bias), 0)


fn test_simple_mlp_initialization_2_hidden() raises:
    """Test SimpleMLP with 2 hidden layers."""
    var mlp = SimpleMLP(10, 20, 5, num_hidden_layers=2)
    assert_equal(mlp.input_dim, 10)
    assert_equal(mlp.hidden_dim, 20)
    assert_equal(mlp.output_dim, 5)
    assert_equal(mlp.num_hidden_layers, 2)

    # Check weight dimensions
    assert_equal(len(mlp.layer1_weights), 10 * 20)
    assert_equal(len(mlp.layer1_bias), 20)
    assert_equal(len(mlp.layer2_weights), 20 * 20)
    assert_equal(len(mlp.layer2_bias), 20)
    assert_equal(len(mlp.layer3_weights), 20 * 5)
    assert_equal(len(mlp.layer3_bias), 5)


# ============================================================================
# SimpleMLP Forward Pass Tests
# ============================================================================


fn test_simple_mlp_forward_1_hidden() raises:
    """Test SimpleMLP forward pass with 1 hidden layer."""
    var mlp = SimpleMLP(10, 20, 5, num_hidden_layers=1)

    # Test with List[Float32] input
    var input = List[Float32]()
    for _ in range(10):
        input.append(1.0)

    var output = mlp.forward(input)
    assert_equal(len(output), 5)


fn test_simple_mlp_forward_2_hidden() raises:
    """Test SimpleMLP forward pass with 2 hidden layers."""
    var mlp = SimpleMLP(10, 20, 5, num_hidden_layers=2)

    var input = List[Float32]()
    for _ in range(10):
        input.append(1.0)

    var output = mlp.forward(input)
    assert_equal(len(output), 5)


fn test_simple_mlp_forward_extensor_1_hidden() raises:
    """Test SimpleMLP ExTensor forward pass with 1 hidden layer."""
    var mlp = SimpleMLP(10, 20, 5, num_hidden_layers=1)
    var input_shape = [10]
    var input = zeros(input_shape, DType.float32)

    var output = mlp.forward(input)

    # Check output shape
    assert_equal(len(output._shape), 1)
    assert_equal(output._shape[0], 5)

    # Check dtype preserved
    assert_true(
        output._dtype == DType.float32, "Output dtype should be float32"
    )


fn test_simple_mlp_forward_extensor_2_hidden() raises:
    """Test SimpleMLP ExTensor forward pass with 2 hidden layers."""
    var mlp = SimpleMLP(10, 20, 5, num_hidden_layers=2)
    var input_shape = [10]
    var input = ones(input_shape, DType.float32)

    var output = mlp.forward(input)

    # Check output shape
    assert_equal(len(output._shape), 1)
    assert_equal(output._shape[0], 5)


fn main() raises:
    """Run all tests."""
    print("Testing SimpleMLP initialization...")
    test_simple_mlp_initialization_1_hidden()
    test_simple_mlp_initialization_2_hidden()

    print("Testing SimpleMLP forward pass...")
    test_simple_mlp_forward_1_hidden()
    test_simple_mlp_forward_2_hidden()
    test_simple_mlp_forward_extensor_1_hidden()
    test_simple_mlp_forward_extensor_2_hidden()

    print("All tests passed!")
