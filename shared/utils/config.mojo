"""Configuration management utilities for ML Odyssey.

This module provides configuration loading, validation, and merging
from YAML and JSON files. Supports nested configurations, environment
variable substitution, and validation rules.

Example:
    from shared.utils import Config.

    var config = Config.from_yaml("config.yaml").
    var lr = config.get_float("learning_rate").
    var batch_size = config.get_int("batch_size").
    ```
"""

from python import Python, PythonObject


# ============================================================================
# Configuration Value Union Type
# ============================================================================


struct ConfigValue(Copyable, ImplicitlyCopyable, Movable):
    """Union type to hold different configuration value types.

    Supports common types needed for ML configurations: integers, floats,
    strings, booleans, and lists.
    """

    var value_type: String
    """Type of the value: 'int', 'float', 'string', 'bool', or 'list'."""
    var int_val: Int
    """Integer value if type is 'int'."""
    var float_val: Float64
    """Float value if type is 'float'."""
    var str_val: String
    """String value if type is 'string'."""
    var bool_val: Bool
    """Boolean value if type is 'bool'."""
    var list_val: List[String]
    """List value if type is 'list'."""

    fn __init__(out self, value: Int):
        """Create ConfigValue from Int.

        Args:
            value: Integer value.

        Returns:
            None.
        """
        self.value_type = "int"
        self.int_val = value
        self.float_val = 0.0
        self.str_val = ""
        self.bool_val = False
        self.list_val = List[String]()

    fn __init__(out self, value: Float64):
        """Create ConfigValue from Float64.

        Args:
            value: Float64 value.

        Returns:
            None.
        """
        self.value_type = "float"
        self.int_val = 0
        self.float_val = value
        self.str_val = ""
        self.bool_val = False
        self.list_val = List[String]()

    fn __init__(out self, value: String):
        """Create ConfigValue from String.

        Args:
            value: String value.

        Returns:
            None.
        """
        self.value_type = "string"
        self.int_val = 0
        self.float_val = 0.0
        self.str_val = value
        self.bool_val = False
        self.list_val = List[String]()

    fn __init__(out self, value: Bool):
        """Create ConfigValue from Bool.

        Args:
            value: Boolean value.

        Returns:
            None.
        """
        self.value_type = "bool"
        self.int_val = 0
        self.float_val = 0.0
        self.str_val = ""
        self.bool_val = value
        self.list_val = List[String]()

    fn __init__(out self, var value: List[String]):
        """Create ConfigValue from List[String].

        Args:
            value: List of strings.

        Returns:
            None.
        """
        self.value_type = "list"
        self.int_val = 0
        self.float_val = 0.0
        self.str_val = ""
        self.bool_val = False
        self.list_val = value^

    fn __init__(out self, value: List[Int]):
        """Create ConfigValue from List[Int].

        Args:
            value: List of integers.

        Returns:
            None.
        """
        self.value_type = "list"
        self.int_val = 0
        self.float_val = 0.0
        self.str_val = ""
        self.bool_val = False
        self.list_val = List[String]()
        # Convert int list to string list for storage
        for i in range(len(value)):
            self.list_val.append(String(value[i]))

    fn __copyinit__(out self, existing: Self):
        """Copy constructor for ConfigValue.

        Args:
            existing: Existing ConfigValue to copy from.

        Returns:
            None.
        """
        self.value_type = existing.value_type
        self.int_val = existing.int_val
        self.float_val = existing.float_val
        self.str_val = existing.str_val
        self.bool_val = existing.bool_val
        self.list_val = existing.list_val.copy()


# ============================================================================
# Config Struct
# ============================================================================


struct Config(Copyable, ImplicitlyCopyable, Movable):
    """Configuration container with nested access and validation.

    Stores configuration as key-value pairs with support for nested
    access using dot notation (e.g., "model.learning_rate").
    """

    var data: Dict[String, ConfigValue]
    """Dictionary storing configuration key-value pairs."""

    fn __init__(out self):
        """Create empty configuration.

        Returns:
            None.
        """
        self.data = Dict[String, ConfigValue]()

    fn __copyinit__(out self, existing: Self):
        """Copy constructor for Config.

        Args:
            existing: Existing Config to copy from.

        Returns:
            None.
        """
        self.data = existing.data.copy()

    fn set(mut self, key: String, value: Int):
        """Set integer configuration value.

        Args:
            key: Configuration key.
            value: Integer value to set.
        """
        self.data[key] = ConfigValue(value)

    fn set(mut self, key: String, value: Float64):
        """Set float configuration value.

        Args:
            key: Configuration key.
            value: Float64 value to set.
        """
        self.data[key] = ConfigValue(value)

    fn set(mut self, key: String, value: String):
        """Set string configuration value.

        Args:
            key: Configuration key.
            value: String value to set.
        """
        self.data[key] = ConfigValue(value)

    fn set(mut self, key: String, value: Bool):
        """Set boolean configuration value.

        Args:
            key: Configuration key.
            value: Boolean value to set.
        """
        self.data[key] = ConfigValue(value)

    fn set(mut self, key: String, var value: List[Int]):
        """Set list of integers configuration value.

        Args:
            key: Configuration key.
            value: List of integers to set.
        """
        self.data[key] = ConfigValue(value)

    fn set(mut self, key: String, var value: List[String]):
        """Set list of strings configuration value.

        Args:
            key: Configuration key.
            value: List of strings to set.
        """
        self.data[key] = ConfigValue(value^)

    fn set(mut self, key: String, value: StringSlice) raises:
        """Set configuration value from StringSlice.

        Args:
            key: Configuration key.
            value: StringSlice value to set.

        Raises:
            Error: If value conversion fails.
        """
        self.set(key, String(value))

    fn has(self, key: String) -> Bool:
        """Check if configuration key exists.

        Args:
            key: Configuration key.

        Returns:
            True if key exists, False otherwise.
        """
        return key in self.data

    fn has_key(self, key: String) -> Bool:
        """Check if configuration key exists (comptime for has).

        Args:
            key: Configuration key.

        Returns:
            True if key exists, False otherwise.
        """
        return self.has(key)

    fn get_string(self, key: String, default: String = "") raises -> String:
        """Get string value with optional default.

        Args:
            key: Configuration key.
            default: Value to return if key not found.

        Returns:
            String value or default.

        Raises:
            Error: If key exists but type is not string.
        """
        if key not in self.data:
            return default

        var val = self.data[key]
        if val.value_type != "string":
            raise Error(
                "Type mismatch for key '"
                + key
                + "': expected string but got "
                + val.value_type
            )
        return val.str_val

    fn get_int(self, key: String, default: Int = 0) raises -> Int:
        """Get integer value with optional default.

        Args:
            key: Configuration key.
            default: Value to return if key not found.

        Returns:
            Integer value or default.

        Raises:
            Error: If key exists but type is not int.
        """
        if key not in self.data:
            return default

        var val = self.data[key]
        if val.value_type != "int":
            raise Error(
                "Type mismatch for key '"
                + key
                + "': expected int but got "
                + val.value_type
            )
        return val.int_val

    fn get_float(self, key: String, default: Float64 = 0.0) raises -> Float64:
        """Get float value with optional default.

        Args:
            key: Configuration key.
            default: Value to return if key not found.

        Returns:
            Float value or default.

        Raises:
            Error: If key exists but type is not float.
        """
        if key not in self.data:
            return default

        var val = self.data[key]
        if val.value_type != "float":
            raise Error(
                "Type mismatch for key '"
                + key
                + "': expected float but got "
                + val.value_type
            )
        return val.float_val

    fn get_bool(self, key: String, default: Bool = False) raises -> Bool:
        """Get boolean value with optional default.

        Args:
            key: Configuration key.
            default: Value to return if key not found.

        Returns:
            Boolean value or default.

        Raises:
            Error: If key exists but type is not bool.
        """
        if key not in self.data:
            return default

        var val = self.data[key]
        if val.value_type != "bool":
            raise Error(
                "Type mismatch for key '"
                + key
                + "': expected bool but got "
                + val.value_type
            )
        return val.bool_val

    fn get_list(self, key: String) raises -> List[String]:
        """Get list value.

        Args:
            key: Configuration key.

        Returns:
            List value or empty list if not found.

        Raises:
            Error: If key exists but type is not list.
        """
        if key not in self.data:
            return []

        var val = self.data[key]
        if val.value_type != "list":
            raise Error(
                "Type mismatch for key '"
                + key
                + "': expected list but got "
                + val.value_type
            )
        # Copy list_val before returning to avoid partial destruction of val
        var list_copy = List[String]()
        for i in range(len(val.list_val)):
            list_copy.append(val.list_val[i])
        return list_copy^

    fn get(self, key: String) raises -> ConfigValue:
        """Get raw configuration value by key.

        Args:
            key: Configuration key.

        Returns:
            Configuration value or default ConfigValue if not found.

        Raises:
            Error: If key access fails.
        """
        if key in self.data:
            return self.data[key]
        return ConfigValue("")  # Default to empty string

    fn get_int_with_default(self, key: String, default: Int) -> Int:
        """Get integer configuration value with default.

        Args:
            key: Configuration key.
            default: Default value to return if key not found.

        Returns:
            Integer value or default.
        """
        if not self.has(key):
            return default
        try:
            return self.get_int(key, default)
        except:
            return default

    fn get_float_with_default(self, key: String, default: Float64) -> Float64:
        """Get float configuration value with default.

        Args:
            key: Configuration key.
            default: Default value to return if key not found.

        Returns:
            Float value or default.
        """
        if not self.has(key):
            return default
        try:
            return self.get_float(key, default)
        except:
            return default

    fn get_string_with_default(self, key: String, default: String) -> String:
        """Get string configuration value with default.

        Args:
            key: Configuration key.
            default: Default value to return if key not found.

        Returns:
            String value or default.
        """
        if not self.has(key):
            return default
        try:
            return self.get_string(key, default)
        except:
            return default

    fn get_bool_with_default(self, key: String, default: Bool) -> Bool:
        """Get boolean configuration value with default.

        Args:
            key: Configuration key.
            default: Default value to return if key not found.

        Returns:
            Bool value or default.
        """
        if not self.has(key):
            return default
        try:
            return self.get_bool(key, default)
        except:
            return default

    fn merge(self, other: Config) -> Config:
        """Merge with another config, other takes precedence.

        Args:
            other: Config to merge (overrides self).

        Returns:
            New merged configuration.
        """
        var result = Config()

        # Copy all from self
        for ref item in self.data.items():
            result.data[item.key] = item.value

        # Override with other
        for ref item in other.data.items():
            result.data[item.key] = item.value

        return result

    fn validate(self, required_keys: List[String]) raises:
        """Validate that all required keys are present.

        Args:
            required_keys: List of required configuration keys.

        Raises:
            Error: If any required key is missing.
        """
        for i in range(len(required_keys)):
            var key = required_keys[i]
            if not self.has(key):
                raise Error("Missing required configuration key: " + key)

    fn validate_type(self, key: String, type_name: String) raises:
        """Validate that a key has the expected type.

        Args:
            key: Configuration key.
            type_name: Expected type name ("int", "float", "string", "bool", "list").

        Raises:
            Error: If type doesn't match.
        """
        if not self.has(key):
            raise Error("Key not found: " + key)

        var val = self.data[key]
        if val.value_type != type_name:
            raise Error(
                "Type mismatch for key '"
                + key
                + "': expected "
                + type_name
                + " but got "
                + val.value_type
            )

    fn validate_range(
        self, key: String, min_val: Float64, max_val: Float64
    ) raises:
        """Validate that a numeric value is in range.

        Args:
            key: Configuration key.
            min_val: Minimum allowed value.
            max_val: Maximum allowed value.

        Raises:
            Error: If value is out of range.
        """
        if not self.has(key):
            raise Error("Key not found: " + key)

        var val = self.data[key]
        var num_val: Float64

        if val.value_type == "float":
            num_val = val.float_val
        elif val.value_type == "int":
            num_val = Float64(val.int_val)
        else:
            raise Error("Cannot validate range for non-numeric type")

        if num_val < min_val or num_val > max_val:
            raise Error(
                "Value for key '"
                + key
                + "' is out of range ["
                + String(min_val)
                + ", "
                + String(max_val)
                + "]"
            )

    fn validate_enum(self, key: String, valid_values: List[String]) raises:
        """Validate that a value is one of allowed enum values.

        Args:
            key: Configuration key.
            valid_values: List of allowed values.

        Raises:
            Error: If value is not in valid_values.
        """
        if not self.has(key):
            raise Error("Key not found: " + key)

        var val = self.get_string(key)
        var found = False
        for i in range(len(valid_values)):
            if val == valid_values[i]:
                found = True
                break

        if not found:
            raise Error("Invalid value for key '" + key + "': " + val)

    fn validate_exclusive(self, keys: List[String]) raises:
        """Validate that at most one of the mutually exclusive keys is set.

        Args:
            keys: List of mutually exclusive keys.

        Raises:
            Error: If more than one key is present.
        """
        var count = 0
        for i in range(len(keys)):
            if self.has(keys[i]):
                count += 1

        if count > 1:
            raise Error("Mutually exclusive keys found - only one allowed")

    @staticmethod
    fn from_yaml(filepath: String) raises -> Config:
        """Load configuration from YAML file with validation.

        Uses Python's PyYAML library to parse nested YAML structures and
        flattens them into dot-notation keys for easy access.

        Example:
            ```mojo
            AML:
                optimizer:
                  name: "sgd"
                  learning_rate: 0.01

            Results in keys:
                "optimizer.name" = "sgd"
                "optimizer.learning_rate" = 0.01
            ```

        Args:
            filepath: Path to YAML file.

        Returns:
            Loaded configuration.

        Raises:
            Error: If file not found, empty, or invalid YAML.
        """
        var config = Config()

        try:
            # Use Python's PyYAML for proper nested YAML parsing
            var yaml_module = Python.import_module("yaml")

            # Read file content
            var content: String
            with open(filepath, "r") as f:
                content = f.read()

            # Validate: Check for empty file
            if len(content.strip()) == 0:
                raise Error("Config file is empty: " + filepath)

            # Parse YAML using PyYAML
            var py_data = yaml_module.safe_load(content)

            # Flatten nested dict to dot-notation and populate config
            Config._flatten_dict(config, py_data, "")

        except e:
            raise Error("Failed to load YAML file: " + String(e))

        return config

    @staticmethod
    fn _flatten_dict(
        mut config: Config, py_obj: PythonObject, prefix: String
    ) raises:
        """Recursively flatten a Python dict into dot-notation keys.

        Args:
            config: Config object to populate.
            py_obj: Python object (dict, list, or primitive).
            prefix: Current key prefix (empty for root level).
        """
        try:
            var builtins = Python.import_module("builtins")

            # Check if it's a dict
            if builtins.isinstance(py_obj, builtins.dict):
                var items = py_obj.items()
                for item in items:
                    var key = String(item[0])
                    var full_key = (
                        prefix + "." + key if len(prefix) > 0 else key
                    )
                    Config._flatten_dict(config, item[1], full_key)

            # Check if it's a list
            elif builtins.isinstance(py_obj, builtins.list):
                # Store lists as string representations for now
                var list_items = List[String]()
                for item in py_obj:
                    list_items.append(String(item))
                config.set(prefix, list_items^)

            # Check if it's a bool (must check before int, as bool is subclass of int in Python)
            elif builtins.isinstance(py_obj, builtins.bool):
                # Convert to string and check value
                var str_repr = String(py_obj)
                var bool_val = str_repr == "True"
                config.set(prefix, bool_val)

            # Check if it's an int
            elif builtins.isinstance(py_obj, builtins.int):
                # Convert to string then parse to Int
                var str_repr = String(py_obj)
                var int_val = atol(str_repr)
                config.set(prefix, int_val)

            # Check if it's a float
            elif builtins.isinstance(py_obj, builtins.float):
                # Convert to string then parse to Float64
                var str_repr = String(py_obj)
                var float_val = Float64(atof(str_repr))
                config.set(prefix, float_val)

            # Otherwise treat as string
            else:
                var str_val = String(py_obj)
                # Remove quotes if present
                if str_val.startswith('"') and str_val.endswith('"'):
                    str_val = String(str_val[1:-1])
                config.set(prefix, str_val)

        except e:
            raise Error("Failed to flatten dict: " + String(e))

    @staticmethod
    fn from_json(filepath: String) raises -> Config:
        """Load configuration from JSON file with validation.

        Current implementation only supports flat key-value pairs.
        Nested objects and arrays are not yet supported. For complex configs,
        use flattened keys (e.g., "model.learning_rate" instead of nested
        {"model": {"learning_rate": 0.001}}) or consider using Python's json
        module for parsing and converting to Config.

        Args:
            filepath: Path to JSON file.

        Returns:
            Loaded configuration.

        Raises:
            Error: If file not found, empty, or invalid JSON.
        """
        # Current implementation supports flat key-value pairs.
        # Full nested JSON parsing can be added as needed. Using basic parsing.
        var config = Config()

        try:
            with open(filepath, "r") as f:
                var content = f.read()

                # Validate: Check for empty file
                if len(content.strip()) == 0:
                    raise Error("Config file is empty: " + filepath)

                # Remove braces and parse key:value pairs
                var clean = (
                    content.replace("{", "").replace("}", "").replace('"', "")
                )
                var pairs = clean.split(",")

                for i in range(len(pairs)):
                    var pair = pairs[i].strip()
                    if ":" in pair:
                        # Split only on the FIRST colon to handle values with colons
                        var colon_idx = pair.find(":")
                        if colon_idx != -1:
                            var key = String(pair[:colon_idx].strip())
                            var value_str = String(
                                pair[colon_idx + 1 :].strip()
                            )

                            # Try to parse as number
                            if "." in value_str:
                                try:
                                    var float_val = Float64(atof(value_str))
                                    config.set(key, float_val)
                                except:
                                    config.set(key, value_str)
                            else:
                                try:
                                    var int_val = atol(value_str)
                                    config.set(key, int_val)
                                except:
                                    config.set(key, value_str)
        except e:
            raise Error("Failed to load JSON file: " + String(e))

        return config

    fn to_yaml(self, filepath: String) raises:
        """Save configuration to YAML file.

        Args:
            filepath: Output file path.

        Raises:
            Error: If file cannot be written.
        """
        try:
            with open(filepath, "w") as f:
                for ref item in self.data.items():
                    var key = item.key
                    var val = item.value

                    if val.value_type == "int":
                        _ = f.write(key + ": " + String(val.int_val) + "\n")
                    elif val.value_type == "float":
                        _ = f.write(key + ": " + String(val.float_val) + "\n")
                    elif val.value_type == "string":
                        _ = f.write(key + ': "' + val.str_val + '"\n')
                    elif val.value_type == "bool":
                        _ = f.write(key + ": " + String(val.bool_val) + "\n")
                    elif val.value_type == "list":
                        _ = f.write(key + ": [")
                        for i in range(len(val.list_val)):
                            _ = f.write('"' + val.list_val[i] + '"')
                            if i < len(val.list_val) - 1:
                                _ = f.write(", ")
                        _ = f.write("]\n")
        except e:
            raise Error("Failed to save YAML file: " + String(e))

    fn to_json(self, filepath: String) raises:
        """Save configuration to JSON file.

        Args:
            filepath: Output file path.

        Raises:
            Error: If file cannot be written.
        """
        try:
            with open(filepath, "w") as f:
                _ = f.write("{\n")

                var count = 0
                var total = len(self.data)

                for ref item in self.data.items():
                    var key = item.key
                    var val = item.value

                    _ = f.write('  "' + key + '": ')

                    if val.value_type == "int":
                        _ = f.write(String(val.int_val))
                    elif val.value_type == "float":
                        _ = f.write(String(val.float_val))
                    elif val.value_type == "string":
                        _ = f.write('"' + val.str_val + '"')
                    elif val.value_type == "bool":
                        _ = f.write(String(val.bool_val))
                    elif val.value_type == "list":
                        _ = f.write("[")
                        for i in range(len(val.list_val)):
                            _ = f.write('"' + val.list_val[i] + '"')
                            if i < len(val.list_val) - 1:
                                _ = f.write(", ")
                        _ = f.write("]")

                    count += 1
                    if count < total:
                        _ = f.write(",\n")
                    else:
                        _ = f.write("\n")

                _ = f.write("}\n")
        except e:
            raise Error("Failed to save JSON file: " + String(e))

    fn substitute_env_vars(self) -> Config:
        """Substitute environment variables in config values.

        Replaces ${VAR_NAME} with environment variable values.
        Supports default syntax: ${VAR_NAME:-default}.

        Returns:
            New configuration with substituted environment variable values.
        """
        var result = Config()

        for ref item in self.data.items():
            var key = item.key
            var val = item.value

            if val.value_type == "string":
                var str_val = val.str_val
                var new_val = self._substitute_env_in_string(str_val)
                result.set(key, new_val)
            else:
                result.data[key] = val

        return result

    fn _substitute_env_in_string(self, value: String) -> String:
        """Helper to substitute environment variables in a string.

        Replaces ${VAR} or ${VAR:-default} patterns with environment values.

        Args:
            value: String that may contain ${VAR} patterns.

        Returns:
            String with environment variable values substituted.
        """
        var result = value

        # Find all ${...} patterns and replace them
        # Simple implementation without regex - iterate and find patterns
        var start_pos = 0
        while True:
            var dollar_pos = result.find("${", start_pos)
            if dollar_pos == -1:
                break

            var close_pos = result.find("}", dollar_pos)
            if close_pos == -1:
                break  # Malformed pattern, skip.

            # Extract variable spec: VAR or VAR:-default
            var var_spec = result[dollar_pos + 2 : close_pos]

            # Check for default value syntax: VAR:-default
            var default_value = ""
            var var_name = var_spec
            var colon_pos = var_spec.find(":-")
            if colon_pos != -1:
                var_name = var_spec[:colon_pos]
                default_value = String(var_spec[colon_pos + 2 :])

            # Get environment variable value using Python
            var env_value = default_value
            try:
                var python = Python.import_module("os")
                var py_value = python.getenv(var_name, default_value)
                env_value = String(py_value)
            except:
                # If Python interop fails, use default value
                pass

            # Replace ${...} with environment value
            var before = result[:dollar_pos]
            var after = result[close_pos + 1 :]
            result = before + env_value + after

            # Move start position past the replaced value
            start_pos = len(before) + len(env_value)

        return result

    @staticmethod
    fn load_template(name: String) -> Config:
        """Load a predefined configuration template.

        Args:
            name: Template name (e.g., "training_default", "lenet5").

        Returns:
            Template configuration object.
        """
        var config = Config()

        if name == "training_default":
            config.set("learning_rate", 0.001)
            config.set("batch_size", 32)
            config.set("epochs", 10)
            config.set("optimizer", "sgd")
        elif name == "lenet5":
            config.set("layers", 3)
            config.set("activation", "relu")
            config.set("input_shape", 28)

        return config


# ============================================================================
# Configuration Loading (Legacy Functions)
# ============================================================================


fn load_config(filepath: String) raises -> Config:
    """Load configuration from YAML or JSON file.

    Automatically detects file format based on extension (.yaml/.yml
    for YAML, .json for JSON).

    Args:
        filepath: Path to configuration file.

    Returns:
        Loaded configuration.

    Raises:
        Error: If file doesn't exist or invalid format.

    Example:
        ```mojo
        var config = load_config("configs/lenet5.yaml")
        var lr = config.get_float("learning_rate")
        ```
    """
    if filepath.endswith(".yaml") or filepath.endswith(".yml"):
        return Config.from_yaml(filepath)
    elif filepath.endswith(".json"):
        return Config.from_json(filepath)
    else:
        raise Error("Unknown file format - use .yaml, .yml, or .json")


fn save_config(config: Config, filepath: String) raises:
    """Save configuration to YAML or JSON file.

    Automatically determines output format from file extension.

    Args:
        config: Configuration to save.
        filepath: Output file path.

    Raises:
        Error: If file cannot be written.

    Example:
        ```mojo
        var config = Config()
        config.set("learning_rate", 0.001)
        save_config(config, "config.yaml")
        ```
    """
    if filepath.endswith(".yaml") or filepath.endswith(".yml"):
        config.to_yaml(filepath)
    elif filepath.endswith(".json"):
        config.to_json(filepath)
    else:
        raise Error("Unknown file format - use .yaml, .yml, or .json")


# ============================================================================
# Configuration Merging
# ============================================================================


fn merge_configs(base: Config, override: Config) -> Config:
    """Merge two configurations with override taking precedence.

    Creates a new configuration that combines both inputs, with values
    from override taking precedence over base. Useful for creating
    experiment-specific configs that override defaults.

    Args:
        base: Base configuration (defaults).
        override: Override configuration (experiment-specific).

    Returns:
        Merged configuration.

    Example:
        ```mojo
        var defaults = load_config("config.yaml")
        var experiment = load_config("config.lenet5.yaml")
        var config = merge_configs(defaults, experiment)
        ```
    """
    return base.merge(override)


# ============================================================================
# Configuration Validation
# ============================================================================


struct ConfigValidator(Copyable, ImplicitlyCopyable, Movable):
    """Validator for configuration values."""

    var required_keys: List[String]
    """List of required configuration keys."""
    var allowed_keys: Dict[String, String]
    """Dictionary mapping keys to their expected type names."""

    fn __init__(out self):
        """Create empty validator.

        Returns:
            None.
        """
        self.required_keys = List[String]()
        self.allowed_keys = Dict[String, String]()

    fn __copyinit__(out self, existing: Self):
        """Copy constructor for ConfigValidator.

        Args:
            existing: Existing ConfigValidator to copy from.

        Returns:
            None.
        """
        self.required_keys = existing.required_keys.copy()
        self.allowed_keys = existing.allowed_keys.copy()

    fn require(mut self, key: String) -> Self:
        """Mark key as required.

        Args:
            key: Configuration key to mark as required.

        Returns:
            Self for method chaining.
        """
        self.required_keys.append(key)
        return self.copy()

    fn allow(mut self, key: String, type_name: String) -> Self:
        """Mark key as allowed with specific type.

        Args:
            key: Configuration key to allow.
            type_name: Expected type name for the key.

        Returns:
            Self for method chaining.
        """
        self.allowed_keys[key] = type_name
        return self.copy()

    fn validate(self, config: Config) -> Bool:
        """Validate configuration against rules.

        Args:
            config: Configuration to validate.

        Returns:
            True if valid, False otherwise.
        """
        # Check required keys
        for i in range(len(self.required_keys)):
            if not config.has(self.required_keys[i]):
                return False

        return True


fn create_validator() -> ConfigValidator:
    """Create new configuration validator.

    Returns:
        Empty validator ready for configuration.
    """
    return ConfigValidator()
