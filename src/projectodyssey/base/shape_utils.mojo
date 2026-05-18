"""Shape utility functions with no tensor dependencies.

Pure helper functions for shape manipulation that can be imported
by any package without creating circular dependencies.
"""

from std.collections import List


def _resolve_shape(
    new_shape: List[Int], total_elements: Int
) raises -> List[Int]:
    """Resolve -1 dimension and validate shape.

    Args:
        new_shape: Target shape (may contain -1).
        total_elements: Total elements in source tensor.

    Returns:
        Resolved shape with no -1 values.

    Raises:
        Error: If shape is invalid.
    """
    var inferred_dim = -1
    var known_product: Int = 1
    var new_len = len(new_shape)

    for i in range(new_len):
        if new_shape[i] == -1:
            if inferred_dim != -1:
                raise Error(
                    "reshape: can only specify one unknown dimension (-1)"
                )
            inferred_dim = i
        elif new_shape[i] < 0:
            raise Error("reshape: shape dimensions must be positive or -1")
        else:
            known_product *= new_shape[i]

    var final_shape = List[Int]()

    if inferred_dim != -1:
        if total_elements % known_product != 0:
            raise Error("reshape: cannot infer dimension, incompatible size")
        var inferred_size = total_elements // known_product

        for i in range(new_len):
            if i == inferred_dim:
                final_shape.append(inferred_size)
            else:
                final_shape.append(new_shape[i])
    else:
        for i in range(new_len):
            final_shape.append(new_shape[i])

    var new_total: Int = 1
    for i in range(new_len):
        new_total *= final_shape[i]

    if new_total != total_elements:
        raise Error("reshape: new shape must have same number of elements")

    return final_shape^
