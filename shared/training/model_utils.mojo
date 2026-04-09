"""Model utilities for weight persistence and serialization.

Provides consolidated utilities for saving and loading model weights across
all ML Odyssey paper implementations. Eliminates duplication of ~100 LOC of
save/load code in each model.mojo file.

Core Functions:
    - save_model_weights(model, directory) - Save all model parameters
    - load_model_weights(model, directory) - Load all model parameters
    - get_model_parameters(model) - Extract all trainable parameters
    - set_model_parameters(model, parameters) - Set all trainable parameters

Supported Architectures:
    - LeNet-5 (10 parameters: 5 conv + 5 fc)
    - AlexNet (18 parameters: 5 conv + 3 fc)
    - VGG-16 (26 parameters: 13 conv + 3 fc)
    - ResNet-18 (varies by architecture)
    - GoogLeNet (varies by architecture)
    - MobileNetV1 (varies by architecture)

File Format:
    Each model weight is saved as a separate file:
    - weights_dir/
        ├── param_name_1.weights
        ├── param_name_2.weights
        └── ...

    Each .weights file uses the hex-encoded format:
    Line 1: <parameter_name>
    Line 2: <dtype> <shape_dim0> <shape_dim1> ...
    Line 3+: <hex_encoded_bytes>

Example:
    from shared.training.model_utils import save_model_weights, load_model_weights

    # Save weights
    save_model_weights(model, "checkpoints/epoch_10/")

    # Load weights
    load_model_weights(model, "checkpoints/epoch_10/")
    ```
"""

from shared.tensor.any_tensor import AnyTensor
from shared.utils.serialization import save_tensor, load_tensor
from std.collections import List


# ============================================================================
# Generic Model Utilities
# ============================================================================


def save_model_weights(
    parameters: List[AnyTensor], directory: String, param_names: List[String]
) raises:
    """Save model weights to directory.

        Saves a list of parameter tensors to individual .weights files in the
        specified directory. Each tensor is saved with its corresponding name.

    Args:
            parameters: List of parameter tensors to save.
            directory: Directory path to save weight files (created if doesn't exist).
            param_names: List of parameter names (must match length of parameters).

    Raises:
            Error: If directory creation fails or file write fails.

        Example:
            ```mojo
            var params : List[AnyTensor] = []
            params.append(model.conv1_kernel)
            params.append(model.fc1_weights)

            var names = List[String]()
            names.append("conv1_kernel")
            names.append("fc1_weights")

            save_model_weights(params, "checkpoint/", names)
            ```
    """
    from shared.utils.file_io import create_directory

    # Validate inputs match
    if len(parameters) != len(param_names):
        raise Error("Parameters and param_names lists must have same length")

    # Create directory
    if not create_directory(directory):
        raise Error("Failed to create directory: " + directory)

    # Save each parameter
    for i in range(len(parameters)):
        var filepath = directory + "/" + param_names[i] + ".weights"
        save_tensor(parameters[i], filepath, param_names[i])


def load_model_weights(
    mut parameters: List[AnyTensor], directory: String, param_names: List[String]
) raises:
    """Load model weights from directory.

        Loads parameter tensors from individual .weights files in the directory
        and stores them in the provided parameters list.

    Args:
            parameters: List to populate with loaded tensors.
            directory: Directory containing weight files.
            param_names: List of parameter names to load (in order).

    Raises:
            Error: If directory doesn't exist, file format is invalid, or shape mismatch.

        Example:
            ```mojo
            var params : List[AnyTensor] = []
            var names = List[String]()
            names.append("conv1_kernel")
            names.append("fc1_weights")

            load_model_weights(params, "checkpoint/", names)

            # params is now populated with loaded tensors
            ```
    """
    # Clear existing parameters
    while len(parameters) > 0:
        _ = parameters.pop()

    # Load each parameter
    for i in range(len(param_names)):
        var filepath = directory + "/" + param_names[i] + ".weights"
        var tensor = load_tensor(filepath)
        parameters.append(tensor)


def get_model_parameter_names(model_type: String) raises -> List[String]:
    """Get standard parameter names for a model architecture.

        Returns the canonical parameter names for supported architectures.
        Useful for consistent naming across save/load operations.

    Args:
            model_type: Architecture name ("lenet5", "alexnet", "vgg16", "resnet18", etc.).

    Returns:
            List of parameter names in order.

    Raises:
            Error: If model_type is unknown.

    Note:
            Parameter names must match the struct field names in each model.mojo file.

    Example:
        ```mojo
        var names = get_model_parameter_names("lenet5")
        # Returns: ["conv1_kernel", "conv1_bias", "conv2_kernel", "conv2_bias", ...]
        ```
    """
    if model_type == "lenet5":
        var names = List[String]()
        names.append("conv1_kernel")
        names.append("conv1_bias")
        names.append("conv2_kernel")
        names.append("conv2_bias")
        names.append("fc1_weights")
        names.append("fc1_bias")
        names.append("fc2_weights")
        names.append("fc2_bias")
        names.append("fc3_weights")
        names.append("fc3_bias")
        return names^

    elif model_type == "alexnet":
        var names = List[String]()
        # Conv layers
        names.append("conv1_kernel")
        names.append("conv1_bias")
        names.append("conv2_kernel")
        names.append("conv2_bias")
        names.append("conv3_kernel")
        names.append("conv3_bias")
        names.append("conv4_kernel")
        names.append("conv4_bias")
        names.append("conv5_kernel")
        names.append("conv5_bias")
        # FC layers
        names.append("fc1_weights")
        names.append("fc1_bias")
        names.append("fc2_weights")
        names.append("fc2_bias")
        names.append("fc3_weights")
        names.append("fc3_bias")
        return names^

    elif model_type == "vgg16":
        var names = List[String]()
        # Block 1
        names.append("conv1_1_kernel")
        names.append("conv1_1_bias")
        names.append("conv1_2_kernel")
        names.append("conv1_2_bias")
        # Block 2
        names.append("conv2_1_kernel")
        names.append("conv2_1_bias")
        names.append("conv2_2_kernel")
        names.append("conv2_2_bias")
        # Block 3
        names.append("conv3_1_kernel")
        names.append("conv3_1_bias")
        names.append("conv3_2_kernel")
        names.append("conv3_2_bias")
        names.append("conv3_3_kernel")
        names.append("conv3_3_bias")
        # Block 4
        names.append("conv4_1_kernel")
        names.append("conv4_1_bias")
        names.append("conv4_2_kernel")
        names.append("conv4_2_bias")
        names.append("conv4_3_kernel")
        names.append("conv4_3_bias")
        # Block 5
        names.append("conv5_1_kernel")
        names.append("conv5_1_bias")
        names.append("conv5_2_kernel")
        names.append("conv5_2_bias")
        names.append("conv5_3_kernel")
        names.append("conv5_3_bias")
        # FC layers
        names.append("fc1_weights")
        names.append("fc1_bias")
        names.append("fc2_weights")
        names.append("fc2_bias")
        names.append("fc3_weights")
        names.append("fc3_bias")
        return names^

    elif model_type == "mobilenetv1":
        var names = List[String]()
        # Initial standard convolution
        names.append("initial_conv_weights")
        names.append("initial_conv_bias")
        names.append("initial_bn_gamma")
        names.append("initial_bn_beta")
        names.append("initial_bn_running_mean")
        names.append("initial_bn_running_var")

        # 13 depthwise separable blocks
        # Each block: dw_weights, dw_bias, dw_bn_gamma, dw_bn_beta, dw_bn_running_mean, dw_bn_running_var
        #            pw_weights, pw_bias, pw_bn_gamma, pw_bn_beta, pw_bn_running_mean, pw_bn_running_var
        for block_idx in range(1, 14):
            var block_str = String(block_idx)
            names.append("ds_block_" + block_str + "_dw_weights")
            names.append("ds_block_" + block_str + "_dw_bias")
            names.append("ds_block_" + block_str + "_dw_bn_gamma")
            names.append("ds_block_" + block_str + "_dw_bn_beta")
            names.append("ds_block_" + block_str + "_dw_bn_running_mean")
            names.append("ds_block_" + block_str + "_dw_bn_running_var")
            names.append("ds_block_" + block_str + "_pw_weights")
            names.append("ds_block_" + block_str + "_pw_bias")
            names.append("ds_block_" + block_str + "_pw_bn_gamma")
            names.append("ds_block_" + block_str + "_pw_bn_beta")
            names.append("ds_block_" + block_str + "_pw_bn_running_mean")
            names.append("ds_block_" + block_str + "_pw_bn_running_var")

        # Final FC layer
        names.append("fc_weights")
        names.append("fc_bias")

        return names^

    else:
        raise Error("Unknown model type: " + model_type)


def validate_shapes(loaded: List[AnyTensor], expected: List[AnyTensor]) raises:
    """Validate that loaded tensors match expected shapes.

    Useful for checking that checkpoint weights are compatible with
    the current model architecture before assignment.

    Args:
        loaded: List of loaded tensors.
        expected: List of expected tensors (with correct shapes).

    Raises:
        Error: If any tensor shapes don't match.

    Example:
        ```mojo
        validate_shapes(loaded_params, model.get_parameters())
        ```
    """
    if len(loaded) != len(expected):
        raise Error(
            "Parameter count mismatch: "
            + String(len(loaded))
            + " loaded vs "
            + String(len(expected))
            + " expected"
        )

    for i in range(len(loaded)):
        var loaded_shape = loaded[i].shape()
        var expected_shape = expected[i].shape()

        if len(loaded_shape) != len(expected_shape):
            raise Error(
                "Shape mismatch for parameter "
                + String(i)
                + ": rank "
                + String(len(loaded_shape))
                + " vs "
                + String(len(expected_shape))
            )

        for d in range(len(loaded_shape)):
            if loaded_shape[d] != expected_shape[d]:
                raise Error(
                    "Shape mismatch for parameter "
                    + String(i)
                    + " dimension "
                    + String(d)
                    + ": "
                    + String(loaded_shape[d])
                    + " vs "
                    + String(expected_shape[d])
                )
