"""TensorLike trait — shared interface for Tensor[dtype] and AnyTensor."""

from std.collections import List


trait TensorLike(Copyable, Movable):
    """Common interface for all tensor types.

    Both Tensor[dtype] (compile-time typed) and AnyTensor (runtime-typed)
    conform to this trait, enabling generic code that works with either.
    """

    def numel(self) -> Int:
        """Return total number of elements."""
        ...

    def shape(self) -> List[Int]:
        """Return shape as list of dimension sizes."""
        ...

    def get_dtype(self) -> DType:
        """Return the element data type."""
        ...

    def ndim(self) -> Int:
        """Return the number of dimensions (rank)."""
        ...
