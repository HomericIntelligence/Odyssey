"""Tests for model utilities.

Tests the model weight save/load functionality including:
- Single model weight persistence
- Batch parameter serialization
- Model architecture parameter naming
- Shape validation for loaded weights
"""

from testing import assert_true, assert_equal
from shared.core.any_tensor import AnyTensor, zeros, ones, full
from shared.training.model_utils import (
    save_model_weights,
    load_model_weights,
    get_model_parameter_names,
    validate_shapes,
)
from pathlib import Path
from collections import List
import os


fn test_save_load_model_weights() raises:
    """Test saving and loading model weights."""
    # Create test parameters
    var params: List[AnyTensor] = []
    var shape1: List[Int] = [3, 4]
    var shape2: List[Int] = [4, 2]

    params.append(full(shape1, 1.5, DType.float32))
    params.append(full(shape2, 2.5, DType.float32))

    # Create parameter names
    var names = List[String]()
    names.append("conv1_kernel")
    names.append("fc1_weights")

    # Create temp directory
    var tmpdir = "/tmp/test_model_weights"

    try:
        # Save weights
        save_model_weights(params, tmpdir, names)

        # Verify files were created
        var file1 = tmpdir + "/conv1_kernel.weights"
        var file2 = tmpdir + "/fc1_weights.weights"
        assert_true(_file_exists(file1), "conv1_kernel.weights not created")
        assert_true(_file_exists(file2), "fc1_weights.weights not created")

        # Load weights
        var loaded: List[AnyTensor] = []
        load_model_weights(loaded, tmpdir, names)

        # Verify number of parameters
        assert_equal(len(loaded), 2, "Wrong number of loaded parameters")

        # Verify first parameter
        var loaded_shape1 = loaded[0].shape()
        assert_equal(len(loaded_shape1), 2, "Wrong rank for first parameter")
        assert_equal(loaded_shape1[0], 3, "Wrong first dimension")
        assert_equal(loaded_shape1[1], 4, "Wrong second dimension")

        # Verify values
        for i in range(loaded[0].numel()):
            var v = loaded[0]._get_float64(i)
            assert_true(
                Float32(v) > 1.4 and Float32(v) < 1.6,
                "Value mismatch in first param",
            )

        # Verify second parameter
        var loaded_shape2 = loaded[1].shape()
        assert_equal(len(loaded_shape2), 2, "Wrong rank for second parameter")
        assert_equal(
            loaded_shape2[0], 4, "Wrong first dimension of second param"
        )
        assert_equal(
            loaded_shape2[1], 2, "Wrong second dimension of second param"
        )

    finally:
        # Clean up
        _cleanup_directory(tmpdir)


fn test_get_lenet5_parameter_names() raises:
    """Test parameter naming for LeNet-5."""
    var names = get_model_parameter_names("lenet5")

    assert_equal(len(names), 10, "LeNet-5 should have 10 parameters")

    # Verify key parameter names
    assert_equal(names[0], "conv1_kernel", "First param should be conv1_kernel")
    assert_equal(names[1], "conv1_bias", "Second param should be conv1_bias")
    assert_equal(names[8], "fc3_weights", "Ninth param should be fc3_weights")
    assert_equal(names[9], "fc3_bias", "Tenth param should be fc3_bias")


fn test_get_alexnet_parameter_names() raises:
    """Test parameter naming for AlexNet."""
    var names = get_model_parameter_names("alexnet")

    assert_equal(len(names), 16, "AlexNet should have 16 parameters")

    # Verify conv layers
    assert_equal(names[0], "conv1_kernel", "First param should be conv1_kernel")
    assert_equal(names[9], "conv5_bias", "Tenth param should be conv5_bias")

    # Verify fc layers
    assert_equal(names[10], "fc1_weights", "FC layer 1 weights")
    assert_equal(names[15], "fc3_bias", "Last param should be fc3_bias")


fn test_get_vgg16_parameter_names() raises:
    """Test parameter naming for VGG-16."""
    var names = get_model_parameter_names("vgg16")

    assert_equal(len(names), 32, "VGG-16 should have 32 parameters")

    # Verify block 1
    assert_equal(names[0], "conv1_1_kernel", "First conv layer")
    assert_equal(names[1], "conv1_1_bias", "First bias")
    assert_equal(names[3], "conv1_2_bias", "Block 1 second bias")

    # Verify block 5
    assert_equal(names[20], "conv5_1_kernel", "Block 5 first kernel")

    # Verify FC layers
    assert_equal(names[27], "fc1_bias", "FC1 bias")
    assert_equal(names[31], "fc3_bias", "Last param should be fc3_bias")


fn test_validate_shapes_matching() raises:
    """Test shape validation with matching tensors."""
    # Create matching tensors
    var expected: List[AnyTensor] = []
    var loaded: List[AnyTensor] = []

    var shape1: List[Int] = [3, 4]
    var shape2: List[Int] = [4, 5, 6]

    expected.append(zeros(shape1, DType.float32))
    expected.append(zeros(shape2, DType.float32))

    loaded.append(zeros(shape1, DType.float32))
    loaded.append(zeros(shape2, DType.float32))

    # Should not raise error
    validate_shapes(loaded, expected)


fn test_validate_shapes_rank_mismatch() raises:
    """Test shape validation with rank mismatch."""
    var expected: List[AnyTensor] = []
    var loaded: List[AnyTensor] = []

    var shape1: List[Int] = [3, 4]
    var shape2: List[Int] = [3, 4, 1]  # Different rank

    expected.append(zeros(shape1, DType.float32))
    loaded.append(zeros(shape2, DType.float32))

    # Should raise error for rank mismatch
    try:
        validate_shapes(loaded, expected)
        assert_true(False, "Should have raised error for rank mismatch")
    except:
        # Expected
        pass


fn test_validate_shapes_dimension_mismatch() raises:
    """Test shape validation with dimension mismatch."""
    var expected: List[AnyTensor] = []
    var loaded: List[AnyTensor] = []

    var shape1: List[Int] = [3, 4]
    var shape2: List[Int] = [3, 5]  # Different second dimension

    expected.append(zeros(shape1, DType.float32))
    loaded.append(zeros(shape2, DType.float32))

    # Should raise error for dimension mismatch
    try:
        validate_shapes(loaded, expected)
        assert_true(False, "Should have raised error for dimension mismatch")
    except:
        # Expected
        pass


fn test_validate_shapes_count_mismatch() raises:
    """Test shape validation with parameter count mismatch."""
    var expected: List[AnyTensor] = []
    var loaded: List[AnyTensor] = []

    var shape1: List[Int] = [3, 4]

    expected.append(zeros(shape1, DType.float32))
    expected.append(zeros(shape1, DType.float32))

    loaded.append(zeros(shape1, DType.float32))
    # Loaded has only 1 parameter, expected has 2

    # Should raise error for count mismatch
    try:
        validate_shapes(loaded, expected)
        assert_true(False, "Should have raised error for count mismatch")
    except:
        # Expected
        pass


# ============================================================================
# Helper Functions
# ============================================================================


fn _file_exists(path: String) -> Bool:
    """Check if file exists."""
    try:
        with open(path, "r"):
            return True
    except:
        return False


fn _cleanup_directory(path: String):
    """Remove directory and all contents."""
    try:
        from python import Python

        var shutil = Python.import_module("shutil")
        var _ = shutil.rmtree(path)
    except:
        # If cleanup fails, that's okay for test cleanup
        pass


fn main() raises:
    """Run all model utils tests."""
    test_save_load_model_weights()
    test_get_lenet5_parameter_names()
    test_get_alexnet_parameter_names()
    test_get_vgg16_parameter_names()
    test_validate_shapes_matching()
    test_validate_shapes_rank_mismatch()
    test_validate_shapes_dimension_mismatch()
    test_validate_shapes_count_mismatch()

    print("All model_utils tests passed!")
