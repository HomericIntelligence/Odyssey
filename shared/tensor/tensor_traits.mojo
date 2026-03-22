"""TensorLike trait — shared interface for Tensor[dtype] and AnyTensor."""

from collections import List


trait TensorLike(Copyable, Movable):
    """Common interface for all tensor types.

    Both Tensor[dtype] (compile-time typed) and AnyTensor (runtime-typed)
    conform to this trait, enabling generic code that works with either.
    """

    fn numel(self) -> Int:
        """Return total number of elements."""
        ...

    fn shape(self) -> List[Int]:
        """Return shape as list of dimension sizes."""
        ...

    fn get_dtype(self) -> DType:
        """Return the element data type."""
        ...

    fn ndim(self) -> Int:
        """Return the number of dimensions (rank)."""
        ...
