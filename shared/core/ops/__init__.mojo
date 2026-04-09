"""Mathematical Operations Module.

This module contains low-level mathematical operations optimized for performance.
All operations leverage Mojo's SIMD capabilities for parallel execution.

Components:
    - matmul: Matrix multiplication operations.
    - transpose: Matrix transposition.
    - elementwise: Element-wise operations (add, mul, div, etc.).
    - reduction: Reduction operations (sum, mean, max, min, etc.).
    - broadcast: Broadcasting utilities for tensor operations.

Example:
    ```mojo
    from shared.core.ops import matmul, transpose
    from shared.core.types import Tensor

    def forward(x: Tensor, w: Tensor) -> Tensor:
        # Efficient matrix multiplication
        return matmul(x, transpose(w))
    ```
"""

# Operation exports will be added here as components are implemented
# from .matmul import matmul
# from .elementwise import add, multiply, divide
# from .reduction import sum, mean, max, min
# from .broadcast import broadcast_to
