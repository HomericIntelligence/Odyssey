"""v4 shape (many captures, borrowed x read) with and without keep-alive."""

from odyssey.tensor.tensor_creation import ones, zeros
from odyssey.tensor.any_tensor import AnyTensor
from odyssey.core.parallel_utils import parallelize


def maxpool_like_v4(x: AnyTensor) raises -> AnyTensor:
    var batch = x.shape()[0]
    var channels = 1
    var in_height = 28
    var in_width = 28
    var out_height = 14
    var out_width = 14

    var output = zeros([batch, 1, out_height, out_width], DType.float32)

    @parameter
    def maxpool_batch(b: Int) capturing:
        var acc = Float64(0)
        for i in range(out_height * out_width * channels):
            var in_idx = b * (channels * in_height * in_width) + i
            acc += x._get_float64(in_idx)
        output._set_float64(b, acc)

    parallelize[maxpool_batch](batch, 2)
    return output^


def maxpool_like_v4_keep(x: AnyTensor) raises -> AnyTensor:
    var batch = x.shape()[0]
    var channels = 1
    var in_height = 28
    var in_width = 28
    var out_height = 14
    var out_width = 14

    var output = zeros([batch, 1, out_height, out_width], DType.float32)

    @parameter
    def maxpool_batch(b: Int) capturing:
        var acc = Float64(0)
        for i in range(out_height * out_width * channels):
            var in_idx = b * (channels * in_height * in_width) + i
            acc += x._get_float64(in_idx)
        output._set_float64(b, acc)

    parallelize[maxpool_batch](batch, 2)
    # keep-alive: use x AFTER the closure-pass
    var keep = x._get_float64(0)
    if keep < 0.0:
        print("unreachable")
    return output^


def main() raises:
    var x = ones([4, 1, 28, 28], DType.float32)
    print("v4 (no keep-alive):")
    var r4 = maxpool_like_v4(x)
    print(
        "  v4 output[0]:", Float64(r4.load[DType.float32](0)), "(expect 196.0)"
    )

    print("v4-keep (keep-alive after pass):")
    var rk = maxpool_like_v4_keep(x)
    print(
        "  v4-keep output[0]:",
        Float64(rk.load[DType.float32](0)),
        "(expect 196.0)",
    )
    print("done")
