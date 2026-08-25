"""v6: capture raw _data pointer of BORROWED x (safe escape, Class C) + many
scalars. Does the read work now?"""

from odyssey.tensor.tensor_creation import ones, zeros
from odyssey.tensor.any_tensor import AnyTensor
from odyssey.core.parallel_utils import parallelize


def maxpool_like_v6(x: AnyTensor) raises -> AnyTensor:
    var batch = x.shape()[0]
    var channels = 1
    var in_height = 28
    var in_width = 28
    var out_height = 14
    var out_width = 14

    var output = zeros([batch, 1, out_height, out_width], DType.float32)

    # x is borrowed -> escaping its raw pointer into a local is Class-C safe
    # (the owner lives in the caller frame; this frame cannot destroy x).
    var xd = x._data

    @parameter
    def maxpool_batch(b: Int) capturing:
        var acc = Float64(0)
        for i in range(out_height * out_width * channels):
            var in_idx = b * (channels * in_height * in_width) + i
            var ptr = xd.unsafe_offset(in_idx).unsafe_bitcast[Float32]()
            acc += Float64(ptr[])
        output._set_float64(b, acc)

    parallelize[maxpool_batch](batch, 2)
    return output^


def main() raises:
    var x = ones([4, 1, 28, 28], DType.float32)
    print("v6 (raw ptr capture of borrowed x, many scalars):")
    var r6 = maxpool_like_v6(x)
    print(
        "  v6 output[0]:", Float64(r6.load[DType.float32](0)), "(expect 196.0)"
    )
    print("done")
