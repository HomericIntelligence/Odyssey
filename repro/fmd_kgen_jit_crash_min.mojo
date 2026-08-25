"""Minimal reproducer for FM-D: KGEN JIT runtime crash.
Segfault in libKGENCompilerRTShared.so after successful compilation.
Passes on 1.0.0b2, crashes on 1.0.0 stable.

This minimal version uses only stdlib to avoid the full Odyssey dependency.
It creates a struct with a List field and exercises forward passes that
trigger the JIT crash path.
"""

from collections import List


struct BatchNorm:
    var gamma: List[Float32]
    var beta: List[Float32]
    var running_mean: List[Float32]
    var running_var: List[Float32]
    var num_features: Int

    def __init__(out self, num_features: Int):
        self.num_features = num_features
        self.gamma = List[Float32](num_features, 1.0)
        self.beta = List[Float32](num_features, 0.0)
        self.running_mean = List[Float32](num_features, 0.0)
        self.running_var = List[Float32](num_features, 1.0)

    def forward(self, inp: List[Float32], training: Bool) -> List[Float32]:
        var out = List[Float32](len(inp))
        for i in range(len(inp)):
            var idx = i % self.num_features
            var normalized = (inp[i] - self.running_mean[idx]) / (
                self.running_var[idx] + 1e-5
            ).__sqrt__()
            out[i] = self.gamma[idx] * normalized + self.beta[idx]
            if training:
                self.running_mean[idx] = (
                    0.9 * self.running_mean[idx] + 0.1 * inp[i]
                )
                self.running_var[idx] = (
                    0.9 * self.running_var[idx]
                    + 0.1 * (inp[i] - self.running_mean[idx]) ** 2
                )
        return out^


def main() raises:
    print("=== FM-D: KGEN JIT Crash Reproducer ===")
    var bn = BatchNorm(4)

    var inp1 = List[Float32](8, 1.0)
    for i in range(8):
        inp1[i] = Float32(i) * 0.5

    print("Forward pass 1...")
    var out1 = bn.forward(inp1, True)
    print("  out1[0] =", out1[0])

    var inp2 = List[Float32](8, 2.0)
    for i in range(8):
        inp2[i] = Float32(i) * 0.3

    print("Forward pass 2...")
    var out2 = bn.forward(inp2, True)
    print("  out2[0] =", out2[0])

    var inp3 = List[Float32](8, 0.5)
    print("Forward pass 3...")
    var out3 = bn.forward(inp3, True)
    print("  out3[0] =", out3[0])

    print("All passes completed - no crash!")
