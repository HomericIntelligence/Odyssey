"""Minimal reproducer: FM-D KGEN JIT Crash.
Segfault in libKGENCompilerRTShared.so during JIT execution.
Passes on Mojo 1.0.0b2, crashes on 1.0.0 stable.

The crash occurs during batch normalization forward pass with
training mode enabled (running statistics update).
"""
from odyssey.tensor.tensor import Tensor
from odyssey.core.any_tensor import AnyTensor, zeros, ones
from odyssey.core.normalization import BatchNorm2d

def main():
    print("=== FM-D: KGEN JIT Crash ===")
    
    # Create batch norm layer
    var bn = BatchNorm2d[DType.float32](num_features=4)
    
    # Create input: batch=2, channels=4, H=3, W=3
    var inp = ones([2, 4, 3, 3], dtype=DType.float32)
    
    # First call succeeds (JIT compiles)
    print("Calling forward (training mode)...")
    var out1 = bn.forward(inp, training=True)
    print("  out1 shape:", out1.shape())
    
    # Second call often triggers the crash (different JIT path)
    print("Calling forward again...")
    var out2 = bn.forward(inp, training=True)
    print("  out2 shape:", out2.shape())
    
    print("All done - if you see this, the crash did not occur!")
