"""Test helpers for AnyTensor test suite.

Provides utility functions, fixtures, and helpers for comprehensive testing
of AnyTensor and related components.

Exports:
- print_tensor: Pretty-print tensors for debugging
- tensor_summary: Print tensor statistics
- compare_tensors: Compare two tensors for debugging
- benchmark: Simple performance testing helper
- random_tensor: Create random tensors
- sequential_tensor: Create sequential value tensors
- nan_tensor: Create NaN-filled tensors
- inf_tensor: Create infinity-filled tensors
- ones_like: Create ones matching input shape/dtype
- zeros_like: Create zeros matching input shape/dtype
"""

from .utils import (
    print_tensor,
    tensor_summary,
    compare_tensors,
    benchmark,
)
from .fixtures import (
    random_tensor,
    sequential_tensor,
    nan_tensor,
    inf_tensor,
    ones_like,
    zeros_like,
)
