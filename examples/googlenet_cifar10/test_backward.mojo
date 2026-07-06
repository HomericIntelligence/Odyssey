"""Execution test: GoogLeNet backward pass reduces loss over training steps.

Builds a small GoogLeNet, feeds a fixed synthetic batch whose samples span all
10 classes (interleaved, so no single-class-batch degeneracy), and runs several
SGD-momentum steps. Asserts the loss strictly decreases from first to last step
— the proof that compute_gradients' backward pass and updates actually learn.
"""

from projectodyssey.tensor.any_tensor import AnyTensor
from projectodyssey.tensor.tensor_creation import zeros
from projectodyssey.data import one_hot_encode
from model import GoogLeNet
from train import compute_gradients, initialize_velocities


def main() raises:
    var num_classes = 10
    var batch = 10  # one sample per class, interleaved
    var model = GoogLeNet(num_classes=num_classes)
    var velocities = initialize_velocities(model)

    # Synthetic images (batch, 3, 32, 32): class-correlated signal so the task
    # is learnable but not trivial.
    var images = zeros([batch, 3, 32, 32], DType.float32)
    var img_d = images._data.bitcast[Float32]()
    for s in range(batch):
        var cls = s  # sample s has class s -> every batch sees all classes
        for i in range(3 * 32 * 32):
            img_d[s * (3 * 32 * 32) + i] = (
                Float32(cls) * 0.05 + Float32(i % 5) * 0.01
            )

    # Raw class-index labels -> one-hot
    var labels_raw = zeros([batch], DType.uint8)
    var lbl_d = labels_raw._data.bitcast[UInt8]()
    for s in range(batch):
        lbl_d[s] = UInt8(s)
    var labels = one_hot_encode(labels_raw, num_classes)

    var lr = Float32(0.01)
    var momentum = Float32(0.9)

    var first_loss = Float32(0.0)
    var last_loss = Float32(0.0)
    var steps = 15
    for step in range(steps):
        var loss = compute_gradients(
            model, images, labels, lr, momentum, velocities
        )
        if step == 0:
            first_loss = loss
        last_loss = loss
        print(
            "  Step "
            + String(step + 1)
            + "/"
            + String(steps)
            + ", Loss: "
            + String(loss)
        )

    print("first=" + String(first_loss) + " last=" + String(last_loss))
    if last_loss < first_loss:
        print("GOOGLENET_BWD_CONVERGES: PASS")
    else:
        print("GOOGLENET_BWD_CONVERGES: FAIL (loss did not decrease)")
        raise Error(
            "Loss did not decrease: first="
            + String(first_loss)
            + " last="
            + String(last_loss)
        )
