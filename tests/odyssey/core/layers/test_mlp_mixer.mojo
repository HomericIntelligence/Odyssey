"""Unit tests for the 1-layer MLP-Mixer block (MLPMixerBlock).

Tests cover:
- Construction + shape preservation ([batch, seq, dim] -> [batch, seq, dim])
- Default hidden dims (token_hidden -> 4*seq_len, channel_hidden -> 4*dim)
- Parameter collection (12 tensors: 2 LayerNorm x 2 + 2 FeedForward x 4)
- seq_len / dim / rank validation (raises)
- dtype-guard: a scalar loaded with the wrong dtype misreads the buffer (the
  test uses the tensor's own dtype throughout)
- Numerical parity with a PyTorch reference (LayerNorm -> transpose -> token MLP
  -> residual -> LayerNorm -> channel MLP -> residual) on ramp weights + input,
  tolerance 1e-5. Reference: parity_refs/mlp_mixer_parity_reference.py.
- Package-path import (from odyssey.core.layers import MLPMixerBlock) so the
  export line, not just the docstring Components entry, is exercised.
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros
from odyssey.core.layers.mlp_mixer import MLPMixerBlock

# Package-path import: guards that __init__.mojo actually EXPORTS the symbol
# (a docstring-only Components line would not satisfy this import).
from odyssey.core.layers import MLPMixerBlock as MLPMixerBlockPkg


def test_shape_preserved() raises:
    """Mixer maps [batch, seq, dim] -> [batch, seq, dim]."""
    print("Running test_shape_preserved...")
    var mixer = MLPMixerBlock[DType.float32](seq_len=4, dim=8)
    var x = zeros([2, 4, 8], DType.float32)
    var y = mixer.forward(x)
    if y.shape()[0] != 2 or y.shape()[1] != 4 or y.shape()[2] != 8:
        raise Error("Mixer must preserve [batch, seq, dim] shape")
    print("  ok shape preserved (2, 4, 8)")
    print("test_shape_preserved PASSED")


def test_default_hidden_dims() raises:
    """Default token_hidden is 4*seq_len, channel_hidden is 4*dim."""
    print("Running test_default_hidden_dims...")
    var mixer = MLPMixerBlock[DType.float32](seq_len=5, dim=6)
    if mixer.token_hidden != 20:
        raise Error("default token_hidden should be 4 * seq_len = 20")
    if mixer.channel_hidden != 24:
        raise Error("default channel_hidden should be 4 * dim = 24")
    print("  ok defaults token_hidden=20, channel_hidden=24")
    print("test_default_hidden_dims PASSED")


def test_parameter_count() raises:
    """`parameters()` returns 12 tensors (2 LayerNorm + 2 FeedForward)."""
    print("Running test_parameter_count...")
    var mixer = MLPMixerBlock[DType.float32](
        seq_len=4, dim=8, token_hidden=6, channel_hidden=16
    )
    var params = mixer.parameters()
    if len(params) != 12:
        raise Error("Mixer should expose 12 parameter tensors")
    print("  ok 12 parameter tensors")
    print("test_parameter_count PASSED")


def test_reject_bad_dims() raises:
    """Non-positive seq_len/dim must raise."""
    print("Running test_reject_bad_dims...")
    try:
        var _ = MLPMixerBlock[DType.float32](seq_len=0, dim=8)
        raise Error("Should have rejected seq_len = 0")
    except _:
        print("  ok rejected seq_len = 0")
    try:
        var _ = MLPMixerBlock[DType.float32](seq_len=4, dim=0)
        raise Error("Should have rejected dim = 0")
    except _:
        print("  ok rejected dim = 0")
    print("test_reject_bad_dims PASSED")


def test_reject_shape_mismatch() raises:
    """Wrong rank / seq / dim on forward must raise."""
    print("Running test_reject_shape_mismatch...")
    var mixer = MLPMixerBlock[DType.float32](seq_len=4, dim=8)
    # Wrong rank (2D)
    try:
        var bad = zeros([4, 8], DType.float32)
        var _ = mixer.forward(bad)
        raise Error("Should have rejected rank-2 input")
    except _:
        print("  ok rejected rank-2 input")
    # Wrong seq
    try:
        var bad = zeros([2, 5, 8], DType.float32)
        var _ = mixer.forward(bad)
        raise Error("Should have rejected seq mismatch")
    except _:
        print("  ok rejected seq mismatch")
    # Wrong dim
    try:
        var bad = zeros([2, 4, 7], DType.float32)
        var _ = mixer.forward(bad)
        raise Error("Should have rejected dim mismatch")
    except _:
        print("  ok rejected dim mismatch")
    print("test_reject_shape_mismatch PASSED")


def test_package_import() raises:
    """The package-path export builds and constructs (guards __init__.mojo)."""
    print("Running test_package_import...")
    var mixer = MLPMixerBlockPkg[DType.float32](seq_len=4, dim=8)
    var x = zeros([1, 4, 8], DType.float32)
    var y = mixer.forward(x)
    if y.shape()[2] != 8:
        raise Error("package-path MLPMixerBlock forward failed")
    print("  ok package-path import + forward")
    print("test_package_import PASSED")


def test_parity_with_pytorch() raises:
    """Forward must match the PyTorch MLP-Mixer reference to 1e-5.

    Ramp weights/input are set with the SAME float arithmetic as the generator
    (parity_refs/mlp_mixer_parity_reference.py) — verified bit-identical to the
    reference's stored weight arrays. The 64 expected outputs are transcribed
    literals produced by that generator (re-run + diff before committing).

    Config: batch=2, seq=4, dim=8, token_hidden=6, channel_hidden=16, exact GELU,
    LayerNorm eps=1e-5. Odyssey Linear W is (in, out); the reference sets torch's
    (out, in) weight to the transpose so the flat ramp order matches directly.
    """
    print("Running test_parity_with_pytorch...")

    # Expected output (flattened [2, 4, 8]) from the torch reference.
    var out = List[Float64]()
    out.append(0.4630923633773012)
    out.append(0.515304118067486)
    out.append(0.5676896096626505)
    out.append(0.620251016093345)
    out.append(0.6729888502803849)
    out.append(0.7259019502812196)
    out.append(0.778987503404756)
    out.append(0.8322411035516328)
    out.append(0.5721702475104775)
    out.append(0.6246940237890714)
    out.append(0.6774280780679043)
    out.append(0.7303749220173466)
    out.append(0.7835351512648738)
    out.append(0.8369074341213943)
    out.append(0.89048853893572)
    out.append(0.944273399237553)
    out.append(0.6812170290940615)
    out.append(0.7340518246314501)
    out.append(0.7871334392643378)
    out.append(0.8404647184029136)
    out.append(0.8940463403813144)
    out.append(0.9478768037639067)
    out.append(1.0019524579394081)
    out.append(1.0562675760665834)
    out.append(0.7902366807997545)
    out.append(0.8433816212735346)
    out.append(0.8968099219380731)
    out.append(0.9505247619433791)
    out.append(1.0045269023302499)
    out.append(1.0588146719165101)
    out.append(1.1133840011307836)
    out.append(1.1682285027608976)
    out.append(0.7830923633773011)
    out.append(0.8353041180674861)
    out.append(0.8876896096626505)
    out.append(0.940251016093345)
    out.append(0.9929888502803851)
    out.append(1.0459019502812197)
    out.append(1.098987503404756)
    out.append(1.152241103551633)
    out.append(0.8921702475104774)
    out.append(0.9446940237890713)
    out.append(0.9974280780679041)
    out.append(1.0503749220173466)
    out.append(1.103535151264874)
    out.append(1.156907434121394)
    out.append(1.2104885389357198)
    out.append(1.264273399237553)
    out.append(1.0012170290940614)
    out.append(1.05405182463145)
    out.append(1.1071334392643377)
    out.append(1.1604647184029135)
    out.append(1.2140463403813144)
    out.append(1.2678768037639068)
    out.append(1.321952457939408)
    out.append(1.3762675760665835)
    out.append(1.1102366807997548)
    out.append(1.1633816212735348)
    out.append(1.2168099219380735)
    out.append(1.2705247619433795)
    out.append(1.3245269023302502)
    out.append(1.3788146719165106)
    out.append(1.4333840011307841)
    out.append(1.4882285027608981)

    var mixer = MLPMixerBlock[DType.float64](
        seq_len=4, dim=8, token_hidden=6, channel_hidden=16
    )

    # Token MLP: fc1 (S=4, HT=6) ramp i*0.01 - 0.1; bias (HT=6) i*0.02 - 0.05.
    for i in range(4 * 6):
        mixer.token_mlp.fc1.weight.store[DType.float64](
            i, Float64(i) * 0.01 - 0.1
        )
    for i in range(6):
        mixer.token_mlp.fc1.bias.store[DType.float64](
            i, Float64(i) * 0.02 - 0.05
        )
    # fc2 (HT=6, S=4) ramp i*0.005 - 0.06; bias (S=4) i*0.03 - 0.04.
    for i in range(6 * 4):
        mixer.token_mlp.fc2.weight.store[DType.float64](
            i, Float64(i) * 0.005 - 0.06
        )
    for i in range(4):
        mixer.token_mlp.fc2.bias.store[DType.float64](
            i, Float64(i) * 0.03 - 0.04
        )

    # Channel MLP: fc1 (D=8, HC=16) ramp i*0.002 - 0.12; bias (HC=16) i*0.01 - 0.07.
    for i in range(8 * 16):
        mixer.channel_mlp.fc1.weight.store[DType.float64](
            i, Float64(i) * 0.002 - 0.12
        )
    for i in range(16):
        mixer.channel_mlp.fc1.bias.store[DType.float64](
            i, Float64(i) * 0.01 - 0.07
        )
    # fc2 (HC=16, D=8) ramp i*0.003 - 0.09; bias (D=8) i*0.02 - 0.03.
    for i in range(16 * 8):
        mixer.channel_mlp.fc2.weight.store[DType.float64](
            i, Float64(i) * 0.003 - 0.09
        )
    for i in range(8):
        mixer.channel_mlp.fc2.bias.store[DType.float64](
            i, Float64(i) * 0.02 - 0.03
        )

    # LayerNorm params are gamma=1, beta=0 by construction (matching the
    # reference's default nn.LayerNorm) — left untouched.

    # Input [B=2, S=4, D=8] ramp i*0.01 - 0.15.
    var x = zeros([2, 4, 8], DType.float64)
    for i in range(2 * 4 * 8):
        x.store[DType.float64](i, Float64(i) * 0.01 - 0.15)

    var y = mixer.forward(x)
    for i in range(64):
        var d = y.load[DType.float64](i) - out[i]
        if d < 0:
            d = -d
        if d > 1e-5:
            raise Error("Mixer parity mismatch at index " + String(i))
    print("  ok matches PyTorch MLP-Mixer block to 1e-5 (64 elements)")
    print("test_parity_with_pytorch PASSED")


def main() raises:
    """Run all MLPMixerBlock tests."""
    print("=" * 60)
    print("MLP-Mixer Block Test Suite")
    print("=" * 60)
    test_shape_preserved()
    test_default_hidden_dims()
    test_parameter_count()
    test_reject_bad_dims()
    test_reject_shape_mismatch()
    test_package_import()
    test_parity_with_pytorch()
    print("=" * 60)
    print("All MLPMixerBlock tests PASSED")
    print("=" * 60)
