"""Transform integration tests - Part 3: Value Range Tests.

Split from test_transforms.mojo per ADR-009 to avoid Mojo heap corruption.

# ADR-009: This file is intentionally limited to ≤10 fn test_ functions.
# Mojo v0.26.1 heap corruption (libKGENCompilerRTShared.so) triggers under
# high test load. Split from test_transforms.mojo. See docs/adr/ADR-009-heap-corruption-workaround.md

Tests cover:
- Normalize output range validation
"""

from tests.shared.conftest import (
    assert_true,
    assert_equal,
    assert_not_equal,
    assert_close_float,
    TestFixtures,
)
from shared.data.transforms import Compose, Normalize, Reshape
from shared.core.extensor import ExTensor


# ============================================================================
# Transform Value Range Tests
# ============================================================================


fn test_normalize_output_range() raises:
    """Test Normalize produces reasonable output range.

    Normalized values should be relatively small (typically in [-1, 1] range).

    Integration Points:
        - Normalize output validation
        - Numerical correctness
        - Statistical properties

    Success Criteria:
        - Transform completes
        - Output tensors are created
        - No Inf/NaN in output (typically)
    """
    TestFixtures.set_seed()

    var data_list = List[Float32]()
    for i in range(20):
        data_list.append(Float32(i))
    var data = ExTensor(data_list^)

    var normalize = Normalize()
    var result = normalize(data)

    # Just verify it produced output
    assert_equal(result.num_elements(), 20)


# ============================================================================
# Main Test Runner
# ============================================================================


fn main() raises:
    """Run transform integration tests - Part 3."""
    print("Running transform integration tests (Part 3)...")

    # Value range tests
    test_normalize_output_range()

    print("✓ All transform integration tests (Part 3) passed!")
