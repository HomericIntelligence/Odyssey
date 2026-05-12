"""Standalone repro candidate for modular/modular#6413 (Backtrace A).

Captured backtrace (CI run 25649843238, Mojo 1.0.0b2.dev2026050805):

    Thread 1 "mojo" received signal SIGILL, Illegal instruction.
    assert_almost_equal () at shared/testing/assertions.mojo:170
    170     var diff = abs(a - b)
    #1  test_tensor_dataset_negative_indexing () at tests/.../test_tensor_dataset.mojo:166
    #2  main ()

This file mirrors test_tensor_dataset_negative_indexing exactly (same imports,
same construction order) so the captured frame matches pixel-for-pixel. It
removes the testing harness (no conftest, no SimpleMLP, no perf_counter) so the
Mojo file is the smallest possible reproducer for Modular.

Run under the gdb wrapper:

    bash scripts/mojo-under-gdb.sh /tmp/cores repro/repro_6413_assert_almost_equal.mojo

Loops the failing block 50× per process so a single process probabilistically
exercises the crash window without needing thousands of process restarts.
"""

from shared.data.datasets import TensorDataset
from shared.tensor.any_tensor import AnyTensor
from shared.testing.assertions import assert_almost_equal, assert_equal


def trigger() raises:
    # Identical to test_tensor_dataset_negative_indexing in
    # tests/shared/data/datasets/test_tensor_dataset.mojo:153-171
    var data_list: List[Float32] = [Float32(1.0), Float32(2.0), Float32(3.0)]
    var data = AnyTensor(data_list^)
    var labels_list: List[Int] = [0, 1, 2]
    var labels = AnyTensor(labels_list^)
    var dataset = TensorDataset(data^, labels^)

    var last_sample = dataset[-1]
    # Crash site in captured core: `abs(a - b)` inside assert_almost_equal.
    assert_almost_equal(last_sample[0][0], Float32(3.0))
    assert_equal(last_sample[1][0], 2)

    var second_last_sample = dataset[-2]
    assert_almost_equal(second_last_sample[0][0], Float32(2.0))
    assert_equal(second_last_sample[1][0], 1)


def main() raises:
    for i in range(50):
        try:
            trigger()
        except e:
            print("iter", i, "raised:", String(e))
            raise e.copy()
    print("repro_6413_assert_almost_equal: completed 50 iterations cleanly")
