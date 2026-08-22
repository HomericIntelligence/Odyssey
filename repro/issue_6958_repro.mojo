"""Reproducer for modular/modular#6958: KGEN JIT runtime crash.

Crashes on Mojo 1.0.0 stable, passes on 1.0.0b2.
Minimal standalone repro — uses only stdlib, no project dependencies.
"""

from collections import List
from math import sqrt

struct BatchNorm:
    var gamma: List[Float32]
    var beta: List[Float32]
    var running_mean: List[Float32]
    var running_var: List[Float32]
    var num_features: Int
    var momentum: Float32

    def __init__(out self, num_features: Int):
        self.num_features = num_features
        self.momentum = 0.1
        self.gamma = List[Float32](num_features, 1.0)
        self.beta = List[Float32](num_features, 0.0)
        self.running_mean = List[Float32](num_features, 0.0)
        self.running_var = List[Float32](num_features, 1.0)

    def forward(
        mut self, inp: List[Float32], training: Bool
    ) -> List[Float32]:
        var batch_len = len(inp)
        var out = List[Float32](batch_len)
        for i in range(batch_len):
            var c = i % self.num_features
            var mean = self.running_mean[c]
            var var_ = self.running_var[c]
            var norm = (inp[i] - mean) / sqrt(Float64(var_ + 1e-5))
            out[i] = self.gamma[c] * Float32(norm) + self.beta[c]
            if training:
                self.running_mean[c] = (
                    Float32(1.0 - self.momentum) * self.running_mean[c]
                    + self.momentum * inp[i]
                )
                self.running_var[c] = (
                    Float32(1.0 - self.momentum) * self.running_var[c]
                    + self.momentum * (inp[i] - self.running_mean[c]) ** 2
                )
        return out^


def main() raises:
    print(===
