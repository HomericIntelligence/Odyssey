"""Testing utilities for ML Odyssey.

Provides tools for validating neural network implementations:
- Assertion functions for test validation
- Data generators for synthetic test datasets
- Gradient checking (numerical vs analytical)
- Consolidated test models and fixtures
- Data type utilities for comprehensive multi-dtype testing
- Fuzzing infrastructure for property-based testing
- Property-based testing for finding edge cases

Modules:
    assertions: Comprehensive assertion functions for testing
    data_generators: Generate synthetic test data (random tensors, classification datasets)
    gradient_checker: Validate backward passes using finite differences
    models: Consolidated test model implementations (SimpleCNN, LinearModel, SimpleMLP, etc.)
    fixtures: Test model factories, tensor utilities, and assertion helpers
    special_values: FP-representable test values (0.0, 0.5, 1.0, 1.5) for layerwise testing
    layer_testers: Reusable layer testing patterns (conv, linear, pooling, activation)
    dtype_utils: DType iteration utilities for testing across multiple precisions
    fuzz_core: Core fuzzing infrastructure (FuzzConfig, FuzzResult, random tensor generators)
    fuzz_shapes: Shape fuzzing utilities (random shapes, edge cases, broadcast pairs)
    fuzz_dtypes: DType fuzzing utilities (random dtypes, edge values, type classification)
    property_testing: Property-based testing framework (random shapes, property test runner)

Test Models:
    SimpleCNN: Minimal CNN for image processing tests
    LinearModel: Single fully-connected layer for basic tests
    SimpleMLP: Multi-layer perceptron (2-3 layers) for composition tests
    SimpleLinearModel: Linear model with explicit weights and bias
    MockLayer: Minimal mock layer for testing
    Parameter: Trainable parameter with gradient tracking
"""

from shared.testing.tolerance_constants import (
    TOLERANCE_DEFAULT,
    TOLERANCE_FLOAT16,
    TOLERANCE_FLOAT32,
    TOLERANCE_FLOAT64,
    TOLERANCE_GRADIENT_RTOL,
    TOLERANCE_GRADIENT_ATOL,
    GRADIENT_CHECK_EPSILON,
    GRADIENT_CHECK_EPSILON_FLOAT32,
    GRADIENT_CHECK_EPSILON_OTHER,
    TOLERANCE_CONV,
    TOLERANCE_SOFTMAX,
    TOLERANCE_CROSS_ENTROPY,
)

from shared.testing.assertions import (
    assert_true,
    assert_false,
    assert_equal,
    assert_not_equal,
    assert_not_none,
    assert_almost_equal,
    assert_dtype_equal,
    assert_equal_int,
    assert_equal_float,
    assert_close_float,
    assert_greater,
    assert_less,
    assert_greater_or_equal,
    assert_less_or_equal,
    assert_shape_equal,
    assert_not_equal_tensor,
    assert_tensor_equal,
    assert_shape,
    assert_dtype,
    assert_numel,
    assert_dim,
    assert_value_at,
    assert_all_values,
    assert_all_close,
    assert_type,
)

from shared.testing.gradient_checker import (
    check_gradients,
    check_gradients_verbose,
    relative_error,
    check_gradient,
    compute_numerical_gradient,
    assert_gradients_close,
)

from shared.testing.data_generators import (
    random_tensor,
    random_uniform,
    random_normal,
    synthetic_classification_data,
)

from shared.testing.models import (
    SimpleCNN,
    LinearModel,
    SimpleMLP,
    SimpleMLP2,
    MockLayer,
    SimpleLinearModel,
    Parameter,
)

from shared.testing.fixtures import (
    create_test_cnn,
    create_linear_model,
    create_test_input,
    create_test_targets,
    assert_tensor_shape,
    assert_tensor_dtype,
    assert_tensor_all_finite,
    assert_tensor_not_all_zeros,
)

from shared.testing.special_values import (
    SPECIAL_VALUE_ZERO,
    SPECIAL_VALUE_HALF,
    SPECIAL_VALUE_ONE,
    SPECIAL_VALUE_ONE_HALF,
    SPECIAL_VALUE_NEG_HALF,
    SPECIAL_VALUE_NEG_ONE,
    create_special_value_tensor,
    create_alternating_pattern_tensor,
    create_seeded_random_tensor,
    verify_special_value_invariants,
    create_zeros_tensor,
    create_ones_tensor,
    create_halves_tensor,
    create_one_and_half_tensor,
)

from shared.testing.layer_testers import LayerTester

from shared.testing.dtype_utils import (
    get_test_dtypes,
    get_float_dtypes,
    get_precision_dtypes,
    get_float32_only,
    dtype_to_string,
)

from shared.testing.layer_params import ConvFixture, LinearFixture

# Fuzzing infrastructure
from shared.testing.fuzz_core import (
    FuzzConfig,
    FuzzResult,
    SeededRNG,
    create_random_tensor,
    create_edge_case_tensor,
    has_nan,
    has_inf,
    is_finite,
    all_values_in_range,
    NumericInvariants,
    check_numeric_invariants,
    verify_shape_preserved,
    verify_dtype_preserved,
    verify_numel_preserved,
    format_failure_message,
)

from shared.testing.fuzz_shapes import (
    ShapeFuzzer,
    generate_broadcast_shapes,
    generate_matmul_shapes,
    generate_same_shape_pair,
    is_valid_shape,
    is_empty_shape,
    is_scalar_shape,
    compute_numel,
    shapes_equal,
    are_broadcast_compatible,
    compute_broadcast_shape,
    shape_to_string,
)

from shared.testing.fuzz_dtypes import (
    DTypeFuzzer,
    get_all_dtypes,
    get_float_dtypes as get_fuzz_float_dtypes,
    get_int_dtypes,
    get_uint_dtypes,
    get_numeric_dtypes,
    get_common_ml_dtypes,
    get_dtype_range,
    get_dtype_safe_range,
    get_edge_values,
    get_special_float_values,
    is_float_dtype,
    is_signed_int_dtype,
    is_unsigned_int_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    supports_negative,
    supports_nan_inf,
    get_dtype_size_bytes,
    get_dtype_precision_bits,
    dtype_to_string as fuzz_dtype_to_string,
)

# Property-based testing
from shared.testing.property_testing import (
    random_shape,
    random_compatible_shape,
    random_broadcastable_shapes,
    run_property_test,
    run_property_test_with_seed,
    assert_tensors_close,
    tensors_equal,
)
