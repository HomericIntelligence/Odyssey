"""Tests for SimpleMLP parameter counting, weights, state dict, and grad methods.

Split from test_test_models.mojo per ADR-009 to avoid Mojo heap corruption.

Coverage:
    - SimpleMLP num_parameters (1 and 2 hidden layers)
    - SimpleMLP get_weights method
    - SimpleMLP parameters method
    - SimpleMLP state_dict (1 and 2 hidden layers)
    - SimpleMLP zero_grad method
"""

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_test_models.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

from shared.testing import (
    SimpleMLP,
    assert_equal,
)


# ============================================================================
# SimpleMLP Parameter and State Tests
# ============================================================================


fn test_simple_mlp_num_parameters_1_hidden() raises:
    """Test SimpleMLP parameter count with 1 hidden layer."""
    var mlp = SimpleMLP(10, 20, 5, num_hidden_layers=1)
    # Layer1: 10*20 weights + 20 bias = 220
    # Layer2: 20*5 weights + 5 bias = 105
    # Total: 325
    assert_equal(mlp.num_parameters(), 325)


fn test_simple_mlp_num_parameters_2_hidden() raises:
    """Test SimpleMLP parameter count with 2 hidden layers."""
    var mlp = SimpleMLP(10, 20, 5, num_hidden_layers=2)
    # Layer1: 10*20 weights + 20 bias = 220
    # Layer2: 20*20 weights + 20 bias = 420
    # Layer3: 20*5 weights + 5 bias = 105
    # Total: 745
    assert_equal(mlp.num_parameters(), 745)


fn test_simple_mlp_get_weights() raises:
    """Test SimpleMLP.get_weights() method."""
    var mlp = SimpleMLP(10, 20, 5, num_hidden_layers=1)
    var weights = mlp.get_weights()

    # Should contain all weights and biases
    var expected_size = 10 * 20 + 20 + 20 * 5 + 5
    assert_equal(weights.numel(), expected_size)


fn test_simple_mlp_parameters() raises:
    """Test SimpleMLP.parameters() method."""
    var mlp = SimpleMLP(10, 20, 5, num_hidden_layers=1)
    var params = mlp.parameters()

    # Should have 4 parameter tensors (w1, b1, w2, b2)
    assert_equal(len(params), 4)

    # Check shapes
    assert_equal(params[0].numel(), 10 * 20)  # w1
    assert_equal(params[1].numel(), 20)  # b1
    assert_equal(params[2].numel(), 20 * 5)  # w2
    assert_equal(params[3].numel(), 5)  # b2


fn test_simple_mlp_state_dict_1_hidden() raises:
    """Test SimpleMLP.state_dict() with 1 hidden layer."""
    var mlp = SimpleMLP(10, 20, 5, num_hidden_layers=1)
    var state = mlp.state_dict()

    # Should have 4 entries: layer1_weights, layer1_bias, layer2_weights, layer2_bias
    assert_equal(len(state), 4)


fn test_simple_mlp_state_dict_2_hidden() raises:
    """Test SimpleMLP.state_dict() with 2 hidden layers."""
    var mlp = SimpleMLP(10, 20, 5, num_hidden_layers=2)
    var state = mlp.state_dict()

    # Should have 6 entries
    assert_equal(len(state), 6)


fn test_simple_mlp_zero_grad() raises:
    """Test SimpleMLP.zero_grad() method."""
    var mlp = SimpleMLP(10, 20, 5)
    # This should not raise
    mlp.zero_grad()


fn main() raises:
    """Run all tests."""
    print("Testing SimpleMLP parameters and state...")
    test_simple_mlp_num_parameters_1_hidden()
    test_simple_mlp_num_parameters_2_hidden()
    test_simple_mlp_get_weights()
    test_simple_mlp_parameters()
    test_simple_mlp_state_dict_1_hidden()
    test_simple_mlp_state_dict_2_hidden()
    test_simple_mlp_zero_grad()

    print("All tests passed!")
