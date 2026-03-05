"""TOML configuration file loader using Python interop.

This module provides simple TOML loading functionality using Python's tomli library.
It's designed specifically for loading precision training configurations.

Example:
    ```mojo
    from shared.utils.toml_loader import load_toml_config

    var config = load_toml_config("configs/lenet5/emnist/fp16.toml")
    var initial_scale = config.get_float("precision.gradient_scaler.initial_scale")
    var precision_mode = config.get_string("precision.mode")
    ```
"""

from python import Python, PythonObject
from shared.utils.config import Config, ConfigValue, merge_configs


fn load_toml_file(filepath: String) raises -> PythonObject:
    """Load TOML file using Python's tomli library.

    Args:
        filepath: Path to TOML file.

    Returns:
        Python dict object with TOML contents.

    Raises:
        Error if file not found or invalid TOML.
    """
    var py = Python.import_module("builtins")
    var tomli = Python.import_module("tomllib")  # Python 3.11+ built-in

    # Read file contents
    var file_obj = py.open(filepath, "rb")
    var toml_dict = tomli.load(file_obj)
    _ = file_obj.close()

    return toml_dict


fn python_dict_to_config(
    py_dict: PythonObject, prefix: String = ""
) raises -> Config:
    """Convert Python dict to Config object recursively.

    Args:
        py_dict: Python dictionary from TOML.
        prefix: Prefix for nested keys (e.g., "precision.gradient_scaler").

    Returns:
        Config object with flattened key-value pairs.

    Example:
        Input TOML:
        ```toml
        [precision]
        mode = "fp16"
        [precision.gradient_scaler]
        initial_scale = 65536.0
        ```

        Output Config keys:
        - `precision.mode = "fp16"`
        - `precision.gradient_scaler.initial_scale = 65536.0`
    """
    var config = Config()

    # Get Python dict methods
    var items = py_dict.items()

    # Iterate through all key-value pairs
    var py = Python.import_module("builtins")
    for item in items:
        # Extract key and value from tuple
        var key_obj = item[0]
        var val_obj = item[1]

        # Convert key to string
        var key = String(key_obj)

        # Build full key with prefix
        var full_key = prefix
        if len(prefix) > 0:
            full_key += "."
        full_key += key

        # Check value type and add to config
        var val_type = py.type(val_obj).__name__

        if val_type == "dict":
            # Recursively process nested dict
            var nested_config = python_dict_to_config(val_obj, full_key)
            # Merge nested config into current config by copying all keys
            # NOTE: Config doesn't have a keys() method, so we use merge_configs
            config = merge_configs(config, nested_config)

        elif val_type == "int":
            var int_val = Int(val_obj)
            config.set(full_key, int_val)

        elif val_type == "float":
            var float_val = Float64(val_obj)
            config.set(full_key, float_val)

        elif val_type == "str":
            var str_val = String(val_obj)
            config.set(full_key, str_val)

        elif val_type == "bool":
            var bool_val = Bool(val_obj)
            config.set(full_key, bool_val)

        # NOTE: List handling could be added here if needed

    return config


fn load_toml_config(filepath: String) raises -> Config:
    """Load TOML configuration file into Config object.

    Args:
        filepath: Path to TOML file.

    Returns:
        Config object with all key-value pairs from TOML.

    Raises:
        Error if file not found or invalid TOML.

    Example:
        ```mojo
        var config = load_toml_config("configs/lenet5/emnist/fp16.toml")
        var mode = config.get_string("precision.mode")
        var scale = config.get_float("precision.gradient_scaler.initial_scale")
        ```
    """
    var py_dict = load_toml_file(filepath)
    return python_dict_to_config(py_dict, "")
