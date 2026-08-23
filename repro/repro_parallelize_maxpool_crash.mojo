"""Standalone copy of the real maxpool_batch closure (pooling.mojo ~line 120)
with the same captures and index arithmetic, to isolate the crash."""

from odyssey.tensor.tensor_creation import ones, zeros
from odyssey.tensor.any_tensor import AnyTensor
from odyssey.core.parallel_utils import parallelize, should_parallelize


def maxpool_like(x: AnyTensor) raises -> AnyTensor:
    var x_shape = x.shape()
    var batch = x_shape[0]
    var channels = x_shape[1]
    var in_height = x_shape[2]
    var in_width = x_shape[3]

    var kernel_size = 2
    var actual_stride = 2
    var padding = 0
    var out_height = (
        in_height + 2 * padding - kernel_size
    ) // actual_stride + 1
    var out_width = (in_width + 2 * padding - kernel_size) // actual_stride + 1

    var out_shape = List[Int](capacity=4)
    out_shape.append(batch)
    out_shape.append(channels)
    out_shape.append(out_height)
    out_shape.append(out_width)
    var output = zeros(out_shape, x.dtype())

    if should_parallelize(batch):

        @parameter
        def maxpool_batch(b: Int) capturing:
            for c in range(channels):
                for oh in range(out_height):
                    for ow in range(out_width):
                        var in_h_start = oh * actual_stride - padding
                        var in_w_start = ow * actual_stride - padding
                        var max_val = Float64(-65504.0)
                        for kh in range(kernel_size):
                            for kw in range(kernel_size):
                                var in_h = in_h_start + kh
                                var in_w = in_w_start + kw
                                if (
                                    in_h >= 0
                                    and in_h < in_height
                                    and in_w >= 0
                                    and in_w < in_width
                                ):
                                    var in_idx = (
                                        b * (channels * in_height * in_width)
                                        + c * (in_height * in_width)
                                        + in_h * in_width
                                        + in_w
                                    )
                                    var val = x._get_float64(in_idx)
                                    if val > max_val:
                                        max_val = val
                        var out_idx = (
                            b * (channels * out_height * out_width)
                            + c * (out_height * out_width)
                            + oh * out_width
                            + ow
                        )
                        output._set_float64(out_idx, max_val)

        parallelize[maxpool_batch](batch)
    else:
        for b in range(batch):
            for c in range(channels):
                for oh in range(out_height):
                    for ow in range(out_width):
                        var max_val = Float64(-65504.0)
                        for kh in range(kernel_size):
                            for kw in range(kernel_size):
                                var in_h = oh * actual_stride - padding + kh
                                var in_w = ow * actual_stride - padding + kw
                                if (
                                    in_h >= 0
                                    and in_h < in_height
                                    and in_w >= 0
                                    and in_w < in_width
                                ):
                                    var in_idx = (
                                        b * (channels * in_height * in_width)
                                        + c * (in_height * in_width)
                                        + in_h * in_width
                                        + in_w
                                    )
                                    var val = x._get_float64(in_idx)
                                    if val > max_val:
                                        max_val = val
                        var out_idx = (
                            b * (channels * out_height * out_width)
                            + c * (out_height * out_width)
                            + oh * out_width
                            + ow
                        )
                        output._set_float64(out_idx, max_val)

    return output^


def main() raises:
    var x = ones([4, 1, 28, 28], DType.float32)
    print("calling maxpool_like (batch=4, parallel)")
    var res = maxpool_like(x)
    print("sum:", Float64(res.load[DType.float32](0)))
    var total = Float64(0)
    for i in range(res.numel()):
        total += Float64(res.load[DType.float32](i))
    print("total sum:", total)
    print("done")
