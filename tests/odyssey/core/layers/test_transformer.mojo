"""Unit tests for the pre-LN Transformer encoder block.

Tests cover:
- Construction + validation (positive d_model; num_heads divisibility inherited
  from the attention sub-layer).
- forward() shape preservation ([batch, seq, d_model] -> same).
- 3D-input requirement (rejects non-3D input) and d_model mismatch.
- parameter collection (16 tensors: norm1 x2, attn x8, norm2 x2, ffn x4).
- Numerical parity with an explicit PyTorch pre-LN Transformer block reference on
  fixed ramp weights + a non-affine ramp input (tolerance 1e-5), for BOTH:
    * no_mask (num_heads=1, causal=False)
    * causal  (num_heads=2, causal=True)
- The two parity fixtures produce DIFFERENT outputs (proves the causal mask is
  actually exercised, not collapsed to a no-op by the LayerNorm).
- Package-path export via core/layers/__init__.mojo.

Reference values are transcribed from the committed JSON fixture
parity_refs/transformer_parity_reference.json (produced by
parity_refs/transformer_parity_reference.py; d_model=4, seq=3, batch=2, d_ff=8,
float64). The Mojo weights and input are seeded here by the SAME ramp / non-affine
formulas the generator uses, in the SAME flat order (norm1, q,k,v,out, norm2,
ffn, X), so only the reference OUTPUTS are transcribed — never the weights.
Odyssey Linear uses W (in, out); the reference sets torch's (out, in) weight to
the transpose. The two LayerNorms are identity-affine (gamma=1, beta=0) as
constructed, so no gamma/beta seeding is required.
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros
from odyssey.core.layers.transformer import TransformerEncoderBlock

# Package-path import (via core/layers/__init__.mojo) — asserts the public export
# exists; if the __init__ export line is dropped, this fails to COMPILE.
from odyssey.core.layers import TransformerEncoderBlock as TransformerBlockPkg
from odyssey.core.layers.linear import Linear
from odyssey.core.layers.attention import MultiHeadAttention

comptime D_MODEL = 4
comptime SEQ = 3
comptime BATCH = 2
comptime D_FF = 8


def _seed_ramp(
    mut t: AnyTensor, count: Int, scale: Float64, off: Float64
) raises:
    """Seed a tensor's flat buffer with value[i] = i*scale + off (float64)."""
    for i in range(count):
        t.store[DType.float64](i, Float64(i) * scale + off)


def _seed_projection(
    mut lin: Linear[DType.float64],
    w_scale: Float64,
    w_off: Float64,
    b_scale: Float64,
    b_off: Float64,
) raises:
    """Seed one attention projection: ramp weight (d_model, d_model) + bias."""
    _seed_ramp(lin.weight, D_MODEL * D_MODEL, w_scale, w_off)
    _seed_ramp(lin.bias, D_MODEL, b_scale, b_off)


def _seed_block(mut block: TransformerEncoderBlock[DType.float64]) raises:
    """Seed q/k/v/out attention projections + FFN fc1/fc2 with the generator's
    per-sub-layer ramp specs. LayerNorm gamma/beta keep their identity init."""
    _seed_projection(block.attn.q_proj, 0.11, -0.7, 0.05, -0.2)
    _seed_projection(block.attn.k_proj, 0.13, -0.6, 0.04, -0.15)
    _seed_projection(block.attn.v_proj, 0.09, -0.5, 0.06, -0.25)
    _seed_projection(block.attn.out_proj, 0.07, -0.4, 0.05, -0.1)
    # FFN: fc1 (d_model, d_ff), fc2 (d_ff, d_model).
    _seed_ramp(block.ffn.fc1.weight, D_MODEL * D_FF, 0.007, -0.09)
    _seed_ramp(block.ffn.fc1.bias, D_FF, 0.01, -0.04)
    _seed_ramp(block.ffn.fc2.weight, D_FF * D_MODEL, 0.004, -0.07)
    _seed_ramp(block.ffn.fc2.bias, D_MODEL, 0.02, -0.03)


def _seed_input(mut x: AnyTensor) raises:
    """Seed X[b, s, c] with the generator's non-affine formula so positions are
    NOT affine-equivalent (else LayerNorm collapses them and the causal mask is a
    no-op):  base*0.1 - 0.3 + (s+1)*(c-1.5)^2 * 0.05, base = (b*SEQ+s)*D_MODEL+c.
    """
    for b in range(BATCH):
        for s in range(SEQ):
            for c in range(D_MODEL):
                var base = (b * SEQ + s) * D_MODEL + c
                var curve = Float64(s + 1) * (Float64(c) - 1.5) ** 2 * 0.05
                var val = Float64(base) * 0.1 - 0.3 + curve
                x.store[DType.float64](base, val)


def test_shape_preserved() raises:
    """The block maps [batch, seq, d_model] -> [batch, seq, d_model]."""
    print("Running test_shape_preserved...")
    var block = TransformerEncoderBlock[DType.float32](8, num_heads=2)
    var x = zeros([2, 5, 8], DType.float32)
    var y = block.forward(x)
    if (
        y.dim() != 3
        or y.shape()[0] != 2
        or y.shape()[1] != 5
        or y.shape()[2] != 8
    ):
        raise Error("block must preserve [batch, seq, d_model] shape")
    print("  ok shape preserved (2, 5, 8)")
    print("test_shape_preserved PASSED")


def test_defaults() raises:
    """Defaults: num_heads=1, d_ff=4*d_model, non-causal."""
    print("Running test_defaults...")
    var block = TransformerEncoderBlock[DType.float32](16)
    if block.num_heads != 1 or block.d_ff != 64 or block.causal:
        raise Error("defaults should be num_heads=1, d_ff=64, causal=False")
    print("  ok defaults num_heads=1, d_ff=64, non-causal")
    print("test_defaults PASSED")


def test_reject_bad_config() raises:
    """Non-positive d_model and indivisible d_model/num_heads must raise."""
    print("Running test_reject_bad_config...")
    try:
        var _ = TransformerEncoderBlock[DType.float32](0)
        raise Error("Should have rejected d_model = 0")
    except _:
        print("  ok rejected d_model = 0")
    try:
        var _ = TransformerEncoderBlock[DType.float32](8, num_heads=3)
        raise Error("Should have rejected indivisible d_model/num_heads")
    except _:
        print("  ok rejected d_model % num_heads != 0")
    print("test_reject_bad_config PASSED")


def test_reject_non_3d() raises:
    """A 2D input (missing the sequence axis) must raise."""
    print("Running test_reject_non_3d...")
    var block = TransformerEncoderBlock[DType.float32](8)
    var x = zeros([2, 8], DType.float32)
    try:
        var _ = block.forward(x)
        raise Error("Should have rejected 2D input")
    except _:
        print("  ok rejected non-3D input")
    print("test_reject_non_3d PASSED")


def test_reject_dmodel_mismatch() raises:
    """A 3D input whose last axis != d_model must raise."""
    print("Running test_reject_dmodel_mismatch...")
    var block = TransformerEncoderBlock[DType.float32](8)
    var x = zeros([2, 3, 6], DType.float32)
    try:
        var _ = block.forward(x)
        raise Error("Should have rejected last-dim != d_model")
    except _:
        print("  ok rejected last-dim != d_model")
    print("test_reject_dmodel_mismatch PASSED")


def test_parameter_count() raises:
    """`parameters()` returns 16 tensors: norm1 x2, attn x8, norm2 x2, ffn x4.
    """
    print("Running test_parameter_count...")
    var block = TransformerEncoderBlock[DType.float32](8, num_heads=2)
    if len(block.parameters()) != 16:
        raise Error("block should expose 16 parameter tensors")
    print("  ok 16 parameter tensors")
    print("test_parameter_count PASSED")


def _check_parity(y: AnyTensor, ref_vals: List[Float64], label: String) raises:
    """Assert every flat entry of `y` matches `ref_vals` to 1e-5."""
    for i in range(len(ref_vals)):
        var d = y.load[DType.float64](i) - ref_vals[i]
        if d < 0:
            d = -d
        if d > 1e-5:
            raise Error(
                "transformer parity mismatch ("
                + label
                + ") at index "
                + String(i)
            )


def _forward_no_mask() raises -> AnyTensor:
    """Run the no-mask (num_heads=1) block on the seeded input; return output.
    """
    var block = TransformerEncoderBlock[DType.float64](
        D_MODEL, num_heads=1, d_ff=D_FF, causal=False
    )
    _seed_block(block)
    var x = zeros([BATCH, SEQ, D_MODEL], DType.float64)
    _seed_input(x)
    return block.forward(x)


def _forward_causal() raises -> AnyTensor:
    """Run the causal (num_heads=2) block on the seeded input; return output."""
    var block = TransformerEncoderBlock[DType.float64](
        D_MODEL, num_heads=2, d_ff=D_FF, causal=True
    )
    _seed_block(block)
    var x = zeros([BATCH, SEQ, D_MODEL], DType.float64)
    _seed_input(x)
    return block.forward(x)


def test_parity_no_mask() raises:
    """No-mask forward matches the PyTorch pre-LN block reference to 1e-5.

    Reference: parity_refs/transformer_parity_reference.json["no_mask"]["out"]
    (num_heads=1, causal=False, d_model=4, seq=3, batch=2, d_ff=8).
    """
    print("Running test_parity_no_mask...")
    var ref_vals = List[Float64]()
    ref_vals.append(-0.147374)
    ref_vals.append(0.269714)
    ref_vals.append(0.786802)
    ref_vals.append(1.403890)
    ref_vals.append(0.363359)
    ref_vals.append(0.673406)
    ref_vals.append(1.183452)
    ref_vals.append(1.893499)
    ref_vals.append(0.874124)
    ref_vals.append(1.076687)
    ref_vals.append(1.579249)
    ref_vals.append(2.381812)
    ref_vals.append(1.052626)
    ref_vals.append(1.469714)
    ref_vals.append(1.986802)
    ref_vals.append(2.603890)
    ref_vals.append(1.563359)
    ref_vals.append(1.873406)
    ref_vals.append(2.383452)
    ref_vals.append(3.093499)
    ref_vals.append(2.074124)
    ref_vals.append(2.276687)
    ref_vals.append(2.779249)
    ref_vals.append(3.581812)

    var y = _forward_no_mask()
    _check_parity(y, ref_vals, "no_mask")
    print("  ok no-mask matches PyTorch pre-LN block to 1e-5")
    print("test_parity_no_mask PASSED")


def test_parity_causal() raises:
    """Causal forward matches the PyTorch pre-LN block reference to 1e-5.

    Reference: parity_refs/transformer_parity_reference.json["causal"]["out"]
    (num_heads=2, causal=True, d_model=4, seq=3, batch=2, d_ff=8). Exercises head
    split AND the upper-triangular −∞ causal mask.
    """
    print("Running test_parity_causal...")
    var ref_vals = List[Float64]()
    ref_vals.append(-0.140478)
    ref_vals.append(0.300773)
    ref_vals.append(0.842024)
    ref_vals.append(1.483276)
    ref_vals.append(0.367092)
    ref_vals.append(0.683874)
    ref_vals.append(1.200656)
    ref_vals.append(1.917438)
    ref_vals.append(0.875962)
    ref_vals.append(1.069517)
    ref_vals.append(1.563073)
    ref_vals.append(2.356629)
    ref_vals.append(1.059522)
    ref_vals.append(1.500773)
    ref_vals.append(2.042024)
    ref_vals.append(2.683276)
    ref_vals.append(1.567092)
    ref_vals.append(1.883874)
    ref_vals.append(2.400656)
    ref_vals.append(3.117438)
    ref_vals.append(2.075962)
    ref_vals.append(2.269517)
    ref_vals.append(2.763073)
    ref_vals.append(3.556629)

    var y = _forward_causal()
    _check_parity(y, ref_vals, "causal")
    print("  ok causal matches PyTorch pre-LN block to 1e-5")
    print("test_parity_causal PASSED")


def test_causal_mask_changes_output() raises:
    """The causal fixture must differ from the no-mask fixture.

    If the LayerNorm collapsed all positions to one vector (as a pure-ramp input
    would), the causal mask would be a no-op and both fixtures would coincide.
    The non-affine input (see `_seed_input`) guarantees an observable difference,
    proving the causal path is genuinely exercised by test_parity_causal.
    """
    print("Running test_causal_mask_changes_output...")
    var y_nm = _forward_no_mask()
    var y_ca = _forward_causal()
    var max_diff = 0.0
    for i in range(BATCH * SEQ * D_MODEL):
        var d = y_nm.load[DType.float64](i) - y_ca.load[DType.float64](i)
        if d < 0:
            d = -d
        if d > max_diff:
            max_diff = d
    # Reference max |no_mask - causal| ~ 0.0794; require clearly above tolerance.
    if max_diff < 1e-3:
        raise Error(
            "causal mask has no observable effect (max diff "
            + String(max_diff)
            + "); the fixture no longer exercises masking"
        )
    print("  ok causal mask changes output (max diff > 1e-3)")
    print("test_causal_mask_changes_output PASSED")


def test_package_path_export() raises:
    """`from odyssey.core.layers import TransformerEncoderBlock` must resolve.

    Constructing through the package alias proves the public export line in
    `core/layers/__init__.mojo` is present — if that line is dropped, this test
    file fails to COMPILE, so the export can never silently vanish.
    """
    print("Running test_package_path_export...")
    var block = TransformerBlockPkg[DType.float32](8, num_heads=2)
    if block.d_model != 8 or block.num_heads != 2:
        raise Error("package-path export constructed the wrong layer")
    print("  ok imported via odyssey.core.layers package path")
    print("test_package_path_export PASSED")


def main() raises:
    """Run all TransformerEncoderBlock tests."""
    print("=" * 60)
    print("TransformerEncoderBlock (pre-LN attention + FFN) Test Suite")
    print("=" * 60)
    test_shape_preserved()
    test_defaults()
    test_reject_bad_config()
    test_reject_non_3d()
    test_reject_dmodel_mismatch()
    test_parameter_count()
    test_parity_no_mask()
    test_parity_causal()
    test_causal_mask_changes_output()
    test_package_path_export()
    print("=" * 60)
    print("All TransformerEncoderBlock tests PASSED")
    print("=" * 60)
