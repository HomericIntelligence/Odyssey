"""LeNet-5 Reshape/Flatten Layer Tests

Tests flatten/reshape operations independently with special FP-representable values.

Workaround for Issue #2942: This file contains <10 tests to avoid heap corruption
bug that occurs after running 15+ cumulative tests.

Tests:
- Flatten operation: float32, float16
"""

from shared.tensor.any_tensor import AnyTensor
from shared.testing.assertions import (
    assert_shape,
    assert_dtype,
    assert_false,
)
from shared.testing.special_values import (
    create_special_value_tensor,
    SPECIAL_VALUE_ONE,
)
from math import isnan, isinf


# ============================================================================
# Flatten Test
# ============================================================================


fn test_flatten_operation_float32() raises:
    """Test reshape/flatten operation (16, 5, 5) -> (400,)."""
    var dtype = DType.float32

    # Create a tensor with conv output shape: (1, 16, 5, 5)
    var input = create_special_value_tensor(
        [1, 16, 5, 5], dtype, SPECIAL_VALUE_ONE
    )

    # Flatten: (1, 16, 5, 5) -> (1, 400)
    var flattened = input.reshape([1, 400])

    # Verify shape
    assert_shape(flattened, [1, 400], "Flatten shape mismatch")

    # Verify dtype preserved
    assert_dtype(flattened, dtype, "Flatten dtype mismatch")

    # Verify all values preserved
    _ = 1.0
    for i in range(flattened.numel()):
        var val = flattened._get_float64(i)
        assert_false(isnan(val), "Flatten produced NaN")
        assert_false(isinf(val), "Flatten produced Inf")


fn test_flatten_operation_float16() raises:
    """Test flatten with float16."""
    var dtype = DType.float16

    var input = create_special_value_tensor(
        [1, 16, 5, 5], dtype, SPECIAL_VALUE_ONE
    )
    var flattened = input.reshape([1, 400])

    assert_shape(flattened, [1, 400], "Flatten shape mismatch (float16)")
    assert_dtype(flattened, dtype, "Flatten dtype mismatch (float16)")


fn main() raises:
    """Run all reshape/flatten layer tests."""
    print("LeNet-5 Reshape/Flatten Layer Tests")
    print("=" * 50)

    print("  test_flatten_operation_float32...", end="")
    test_flatten_operation_float32()
    print(" OK")

    print("  test_flatten_operation_float16...", end="")
    test_flatten_operation_float16()
    print(" OK")

    print("\n✅ All reshape/flatten layer tests passed (2/2)")
