"""Tests for Parameterized BatchNorm2dLayer[dtype].

TDD tests for Phase 4a (PR 6, epic #4998): parameterize non-Module layers.
BatchNorm2dLayer becomes BatchNorm2dLayer[dtype: DType = DType.float32] with
Tensor[dtype] forward interface.

Tests cover:
- Default dtype (float32) constructor
- Explicit float64 parameterization
- Parameter shapes (gamma, beta)
- Running statistics shapes
- Forward pass preserves dtype
- Forward pass output shape (NCHW)
- Inference mode uses running stats
- parameters() with float64 dtype (catches CRITICAL-1 bitcast bug)
"""

from std.testing import assert_true, assert_almost_equal
from projectodyssey.core.layers.batchnorm import BatchNorm2dLayer
from projectodyssey.tensor.tensor import Tensor


def test_batchnorm_default_dtype() raises:
    """BatchNorm2dLayer defaults to float32 weights."""
    var bn = BatchNorm2dLayer(num_channels=4)
    # gamma and beta should have float32 dtype
    assert_true(
        bn.gamma.get_dtype() == DType.float32, "gamma dtype should be float32"
    )
    assert_true(
        bn.beta.get_dtype() == DType.float32, "beta dtype should be float32"
    )
    print("PASS: test_batchnorm_default_dtype")


def test_batchnorm_float64() raises:
    """BatchNorm2dLayer[DType.float64] uses float64 tensors."""
    var bn = BatchNorm2dLayer[DType.float64](num_channels=4)
    assert_true(
        bn.gamma.get_dtype() == DType.float64, "gamma dtype should be float64"
    )
    assert_true(
        bn.beta.get_dtype() == DType.float64, "beta dtype should be float64"
    )
    print("PASS: test_batchnorm_float64")


def test_batchnorm_gamma_shape() raises:
    """Gamma has shape (num_channels,) and is initialized to ones."""
    var bn = BatchNorm2dLayer(num_channels=8)
    assert_true(bn.gamma.numel() == 8, "gamma should have 8 elements")
    # Gamma initialized to 1.0
    assert_almost_equal(Float32(bn.gamma[0]), Float32(1.0), atol=1e-6)
    assert_almost_equal(Float32(bn.gamma[7]), Float32(1.0), atol=1e-6)
    print("PASS: test_batchnorm_gamma_shape")


def test_batchnorm_beta_shape() raises:
    """Beta has shape (num_channels,) and is initialized to zeros."""
    var bn = BatchNorm2dLayer(num_channels=8)
    assert_true(bn.beta.numel() == 8, "beta should have 8 elements")
    # Beta initialized to 0.0
    assert_almost_equal(Float32(bn.beta[0]), Float32(0.0), atol=1e-6)
    assert_almost_equal(Float32(bn.beta[7]), Float32(0.0), atol=1e-6)
    print("PASS: test_batchnorm_beta_shape")


def test_batchnorm_running_stats_shape() raises:
    """Running mean/var have shape (num_channels,) with correct init."""
    var bn = BatchNorm2dLayer(num_channels=4)
    assert_true(
        bn.running_mean.numel() == 4, "running_mean should have 4 elements"
    )
    assert_true(
        bn.running_var.numel() == 4, "running_var should have 4 elements"
    )
    # running_mean initialized to 0.0, running_var to 1.0
    assert_almost_equal(Float32(bn.running_mean[0]), Float32(0.0), atol=1e-6)
    assert_almost_equal(Float32(bn.running_var[0]), Float32(1.0), atol=1e-6)
    print("PASS: test_batchnorm_running_stats_shape")


def test_batchnorm_running_stats_float64() raises:
    """Running stats use the parameterized dtype."""
    var bn = BatchNorm2dLayer[DType.float64](num_channels=4)
    assert_true(
        bn.running_mean.get_dtype() == DType.float64,
        "running_mean dtype should be float64",
    )
    assert_true(
        bn.running_var.get_dtype() == DType.float64,
        "running_var dtype should be float64",
    )
    print("PASS: test_batchnorm_running_stats_float64")


def test_batchnorm_forward_typed() raises:
    """Forward pass accepts and returns Tensor[dtype]."""
    var bn = BatchNorm2dLayer[DType.float32](num_channels=2)
    var input = Tensor[DType.float32]([1, 2, 3, 3])  # NCHW
    # Set some values to avoid all-zeros (batch_norm needs variance != 0)
    input._data[0] = Scalar[DType.float32](0.5)
    input._data[1] = Scalar[DType.float32](1.0)
    input._data[9] = Scalar[DType.float32](1.5)
    input._data[10] = Scalar[DType.float32](0.25)
    var output = bn.forward(input.as_any())
    assert_true(
        output.get_dtype() == DType.float32, "output dtype should be float32"
    )
    # Output should have same shape as input: [1, 2, 3, 3]
    var s = output.shape()
    assert_true(s[0] == 1, "batch dim")
    assert_true(s[1] == 2, "channel dim")
    assert_true(s[2] == 3, "height dim")
    assert_true(s[3] == 3, "width dim")
    print("PASS: test_batchnorm_forward_typed")


def test_batchnorm_forward_inference() raises:
    """Forward with training=False uses running stats, not batch stats."""
    var bn = BatchNorm2dLayer(num_channels=2)
    var input = Tensor[DType.float32]([1, 2, 3, 3])
    # Set input to all 1.0 for predictable output
    for i in range(input.numel()):
        input._data[i] = Scalar[DType.float32](1.0)
    # In inference mode with running_mean=0, running_var=1, gamma=1, beta=0:
    # output = gamma * (input - running_mean) / sqrt(running_var + eps) + beta
    # output = 1 * (1 - 0) / sqrt(1 + 1e-5) + 0 ~ 1.0
    var output = bn.forward(input.as_any(), training=False)
    assert_true(output.numel() == 18, "output should have 18 elements")
    assert_almost_equal(Float32(output[0]), Float32(1.0), atol=1e-4)
    print("PASS: test_batchnorm_forward_inference")


def test_batchnorm_parameters_typed() raises:
    """Parameters() returns List[AnyTensor] with gamma and beta."""
    var bn = BatchNorm2dLayer(num_channels=4)
    var params = bn.parameters()
    assert_true(len(params) == 2, "should have 2 parameters (gamma, beta)")
    assert_true(params[0].numel() == 4, "gamma should have 4 elements")
    assert_true(params[1].numel() == 4, "beta should have 4 elements")
    print("PASS: test_batchnorm_parameters_typed")


def test_batchnorm_parameters_float64() raises:
    """Parameters() preserves float64 dtype (catches CRITICAL-1 bitcast bug)."""
    var bn = BatchNorm2dLayer[DType.float64](num_channels=4)
    var params = bn.parameters()
    # Verify dtype is preserved (catches bitcast[Float32] bug)
    assert_true(params[0].dtype() == DType.float64, "gamma param dtype")
    assert_true(params[1].dtype() == DType.float64, "beta param dtype")
    print("PASS: test_batchnorm_parameters_float64")


def main() raises:
    test_batchnorm_default_dtype()
    test_batchnorm_float64()
    test_batchnorm_gamma_shape()
    test_batchnorm_beta_shape()
    test_batchnorm_running_stats_shape()
    test_batchnorm_running_stats_float64()
    test_batchnorm_forward_typed()
    test_batchnorm_forward_inference()
    test_batchnorm_parameters_typed()
    test_batchnorm_parameters_float64()
    print("All 10 typed batchnorm tests passed!")
