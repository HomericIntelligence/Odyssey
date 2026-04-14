"""Tests for SimpleMLP2 - Sequential3-based MLP fixture.

Coverage:
    - SimpleMLP2 initialization (dimensions stored correctly)
    - SimpleMLP2 forward pass output shape
    - SimpleMLP2 parameters count and structure
    - SimpleMLP2 train/inference mode propagation via Sequential3
"""

# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under

from shared.testing import (
    SimpleMLP2,
    assert_true,
    assert_equal,
)
from shared.tensor.any_tensor import (
    AnyTensor,
    zeros,
    ones,
)


# ============================================================================
# SimpleMLP2 Initialization Tests
# ============================================================================


def test_simple_mlp2_initialization() raises:
    """Test SimpleMLP2 stores dimension fields correctly after init."""
    var mlp = SimpleMLP2(10, 20, 5)
    assert_equal(mlp.input_dim, 10)
    assert_equal(mlp.hidden_dim, 20)
    assert_equal(mlp.output_dim, 5)


# ============================================================================
# SimpleMLP2 Forward Pass Tests
# ============================================================================


def test_simple_mlp2_forward_shape_zeros() raises:
    """Test SimpleMLP2 forward produces correct output shape with zero input."""
    var mlp = SimpleMLP2(10, 20, 5)
    var input_shape = [10]
    var input = zeros(input_shape, DType.float32)

    var output = mlp.forward(input)

    assert_equal(len(output._shape), 1)
    assert_equal(output._shape[0], 5)


def test_simple_mlp2_forward_shape_ones() raises:
    """Test SimpleMLP2 forward produces correct output shape with ones input."""
    var mlp = SimpleMLP2(10, 20, 5)
    var input_shape = [10]
    var input = ones(input_shape, DType.float32)

    var output = mlp.forward(input)

    assert_equal(len(output._shape), 1)
    assert_equal(output._shape[0], 5)


def test_simple_mlp2_forward_dtype_preserved() raises:
    """Test SimpleMLP2 forward preserves float32 dtype."""
    var mlp = SimpleMLP2(4, 8, 2)
    var input_shape = [4]
    var input = zeros(input_shape, DType.float32)

    var output = mlp.forward(input)

    assert_true(
        output._dtype == DType.float32, "Output dtype should be float32"
    )


def test_simple_mlp2_forward_small_model() raises:
    """Test SimpleMLP2 forward with minimal dimensions."""
    var mlp = SimpleMLP2(2, 4, 1)
    var input_shape = [2]
    var input = ones(input_shape, DType.float32)

    var output = mlp.forward(input)

    assert_equal(output._shape[0], 1)


# ============================================================================
# SimpleMLP2 Parameter Tests
# ============================================================================


def test_simple_mlp2_parameters_count() raises:
    """Test SimpleMLP2 parameters() returns exactly 4 tensors: W1,b1,W2,b2."""
    var mlp = SimpleMLP2(10, 20, 5)
    var params = mlp.parameters()

    # Sequential3[Linear, ReLULayer, Linear]:
    # Linear(10,20): weight[10,20] + bias[20] = 2 params
    # ReLULayer: 0 params
    # Linear(20,5): weight[20,5] + bias[5] = 2 params
    # Total: 4 tensors
    assert_equal(len(params), 4)


def test_simple_mlp2_parameters_shapes() raises:
    """Test SimpleMLP2 parameter tensors have correct element counts."""
    var mlp = SimpleMLP2(10, 20, 5)
    var params = mlp.parameters()

    # W1: shape (10, 20) = 200 elements
    assert_equal(params[0].numel(), 10 * 20)
    # b1: shape (20,) = 20 elements
    assert_equal(params[1].numel(), 20)
    # W2: shape (20, 5) = 100 elements
    assert_equal(params[2].numel(), 20 * 5)
    # b2: shape (5,) = 5 elements
    assert_equal(params[3].numel(), 5)


# ============================================================================
# SimpleMLP2 Mode Switching Tests
# ============================================================================


def test_simple_mlp2_mode_switching() raises:
    """Test SimpleMLP2 train/inference mode switching, forward still works."""
    var mlp = SimpleMLP2(4, 8, 2)
    var input_shape = [4]
    var input = zeros(input_shape, DType.float32)

    # Switch to training mode and verify forward works
    mlp.train()
    var out_train = mlp.forward(input)
    assert_equal(out_train._shape[0], 2)

    # Switch to inference mode and verify forward works
    mlp.set_inference_mode()
    var out_infer = mlp.forward(input)
    assert_equal(out_infer._shape[0], 2)


def main() raises:
    """Run all SimpleMLP2 tests."""
    print("Testing SimpleMLP2 initialization...")
    test_simple_mlp2_initialization()

    print("Testing SimpleMLP2 forward pass...")
    test_simple_mlp2_forward_shape_zeros()
    test_simple_mlp2_forward_shape_ones()
    test_simple_mlp2_forward_dtype_preserved()
    test_simple_mlp2_forward_small_model()

    print("Testing SimpleMLP2 parameters...")
    test_simple_mlp2_parameters_count()
    test_simple_mlp2_parameters_shapes()

    print("Testing SimpleMLP2 mode switching...")
    test_simple_mlp2_mode_switching()

    print("All SimpleMLP2 tests passed!")
