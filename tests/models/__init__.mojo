"""Model Tests.

Comprehensive testing suite for neural network models:
- Layerwise unit tests (layer-by-layer verification)
- End-to-end integration tests (complete model testing)
- Forward pass validation
- Backward pass and gradient checking
- Save/load functionality

Current Models:
- LeNet-5: Classic CNN for MNIST/EMNIST classification
  - test_lenet5_layers.mojo: Layerwise unit tests (12 layer operations)
  - test_lenet5_e2e.mojo: End-to-end integration tests
- VGG-16: Classic deep CNN for CIFAR-10
  - test_vgg16_e2e.mojo: End-to-end tests with full model forward/backward
- ResNet-18: Residual network with skip connections for CIFAR-10
  - test_resnet18_layers.mojo: Layerwise tests (residual blocks, skip connections, BatchNorm)
  - test_resnet18_e2e.mojo: End-to-end tests with full model forward/backward
"""
