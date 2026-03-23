"""Test Module trait interface.

Tests that the Module trait can be imported and used to create
a simple module implementation.
"""

from shared.core.module import Module
from shared.tensor.any_tensor import AnyTensor, zeros
from tests.shared.conftest import assert_true, assert_equal_int


struct DummyModule:
    """Simple test module implementing Module trait.

    A minimal module for testing the Module trait interface.
    This struct provides all methods required by the Module trait:
    forward(), parameters(), train(), and eval().
    """

    var output_size: Int
    var is_training: Bool

    fn __init__(out self, output_size: Int) raises:
        """Initialize dummy module.

        Args:
            output_size: Size of output tensor.
        """
        self.output_size = output_size
        self.is_training = True

    fn forward(mut self, input: AnyTensor) raises -> AnyTensor:
        """Forward pass returns zeros of specified size.

        Args:
            input: Input tensor (unused).

        Returns:
            Zeros tensor of size (1, output_size).
        """
        var shape: List[Int] = [1, self.output_size]
        return zeros(shape, DType.float32)

    fn parameters(self) raises -> List[AnyTensor]:
        """Return empty parameter list.

        Returns:
            Empty list (no trainable parameters).
        """
        return List[AnyTensor]()

    fn train(mut self):
        """Set to training mode."""
        self.is_training = True

    fn eval(mut self):
        """Set to evaluation mode."""
        self.is_training = False


fn test_module_interface() raises:
    """Test Module trait can be implemented and used."""
    var module = DummyModule(10)

    # Test forward pass
    var input = zeros([1, 5], DType.float32)
    var output = module.forward(input)
    assert_true(len(output.shape()) == 2, "Output should be 2D")
    assert_equal_int(output.shape()[1], 10, "Output size should match")

    # Test parameters
    var params = module.parameters()
    assert_equal_int(len(params), 0, "Should have no parameters")

    # Test training mode
    module.train()
    assert_true(module.is_training, "Should be in training mode")

    # Test eval mode
    module.eval()
    assert_true(not module.is_training, "Should be in eval mode")


fn main() raises:
    """Run all tests."""
    test_module_interface()
    print("All tests passed!")
