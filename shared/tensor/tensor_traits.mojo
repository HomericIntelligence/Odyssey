"""TensorLike trait -- shared interface for Tensor[dtype] and AnyTensor."""

from collections import List


trait TensorLike(Copyable, Hashable, Movable):
    """Common interface for all tensor types.

    Both Tensor[dtype] (compile-time typed) and AnyTensor (runtime-typed)
    conform to this trait, enabling generic code that works with either.

    Includes Hashable so tensors can be used as dictionary keys or in
    hash-based data structures.
    """

    fn numel(self) -> Int:
        """Return total number of elements."""
        ...

    fn shape(self) -> List[Int]:
        """Return shape as list of dimension sizes."""
        ...

    fn dtype(self) -> DType:
        """Return the element data type."""
        ...

    fn ndim(self) -> Int:
        """Return the number of dimensions (rank)."""
        ...
