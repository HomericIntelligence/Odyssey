"""MobileNetV1 CIFAR-10 example package.

Makes ``examples/mobilenetv1_cifar10/`` an importable Mojo package so its
modules can be referenced as package submodules from repo-root include paths
(e.g. ``from examples.mobilenetv1_cifar10.model import MobileNetV1``), which is
how the functional smoke test in ``tests/examples/test_mobilenetv1_train_step.mojo``
imports the real training machinery.

Modules:
- model.mojo          - MobileNetV1 architecture (depthwise-separable convs)
- train.mojo          - SGD training entrypoint
- train_autograd.mojo - Autograd-based training entrypoint
- inference.mojo      - Inference entrypoint

Run training directly:
    mojo run -I . -I examples/mobilenetv1_cifar10 examples/mobilenetv1_cifar10/train.mojo
"""
