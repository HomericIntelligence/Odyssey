"""Unit tests for the standalone MultiHeadAttention self-attention block.

Tests cover:
- Construction + validation (positive d_model/num_heads, divisibility)
- forward() shape preservation ([batch, seq, d_model] -> [batch, seq, d_model])
- 3D-input requirement (rejects non-3D and wrong last-dim inputs)
- parameter collection (8 tensors: q/k/v/out weight+bias)
- Numerical parity with an explicit PyTorch scaled-dot-product self-attention
  reference on fixed ramp weights (tolerance 1e-5), for BOTH:
    * single-head, non-causal (num_heads=1, causal=False)
    * multi-head, causal      (num_heads=2, causal=True)

Reference values are transcribed from the committed JSON fixture
parity_refs/attention_parity_reference.json (produced by
parity_refs/attention_parity_reference.py; d_model=4, seq=3, batch=2, float64).
The Mojo weights are seeded here by the SAME ramp formulas the generator uses
(value[i] = i*scale + off), in the SAME flat order (q,k,v,out weight+bias, X),
so only the reference OUTPUTS are transcribed — never the weights.
Odyssey Linear uses W (in, out); the reference sets torch's (out, in) weight to
the transpose.
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros
from odyssey.core.layers.attention import MultiHeadAttention

# Package-path import (via core/layers/__init__.mojo) — asserts the public
# export exists; if the __init__ export line is dropped, this fails to compile.
from odyssey.core.layers import MultiHeadAttention as MultiHeadAttentionPkg
from odyssey.core.layers.linear import Linear

# Functional attention core — used only by the raises-regression tripwire.
from odyssey.core.attention import (
    scaled_dot_product_attention_masked,
    multi_head_attention_masked,
    MultiHeadAttentionWeights,
)


def _seed_ramp(
    mut t: AnyTensor, count: Int, scale: Float64, off: Float64
) raises:
    """Seed a tensor's flat buffer with value[i] = i*scale + off (float64)."""
    for i in range(count):
        t.store[DType.float64](i, Float64(i) * scale + off)


def _seed_projection(
    mut lin: Linear[DType.float64],
    d_model: Int,
    w_scale: Float64,
    w_off: Float64,
    b_scale: Float64,
    b_off: Float64,
) raises:
    """Seed one projection's ramp weight (d_model, d_model) + bias (d_model,).
    """
    _seed_ramp(lin.weight, d_model * d_model, w_scale, w_off)
    _seed_ramp(lin.bias, d_model, b_scale, b_off)


def _seed_all(mut attn: MultiHeadAttention[DType.float64], d_model: Int) raises:
    """Seed q/k/v/out projections with the generator's per-proj ramp specs."""
    _seed_projection(attn.q_proj, d_model, 0.01, -0.15, 0.02, -0.08)
    _seed_projection(attn.k_proj, d_model, 0.013, -0.12, 0.015, -0.05)
    _seed_projection(attn.v_proj, d_model, 0.008, -0.10, 0.011, -0.06)
    _seed_projection(attn.out_proj, d_model, 0.006, -0.08, 0.03, -0.05)


def test_shape_preserved() raises:
    """Attention maps [batch, seq, d_model] -> [batch, seq, d_model]."""
    print("Running test_shape_preserved...")
    var attn = MultiHeadAttention[DType.float32](8, num_heads=2)
    var x = zeros([2, 5, 8], DType.float32)
    var y = attn.forward(x)
    if (
        y.dim() != 3
        or y.shape()[0] != 2
        or y.shape()[1] != 5
        or y.shape()[2] != 8
    ):
        raise Error("attention must preserve [batch, seq, d_model] shape")
    print("  ok shape preserved (2, 5, 8)")
    print("test_shape_preserved PASSED")


def test_single_head_default() raises:
    """`num_heads` defaults to 1 (single-head), d_k == d_model."""
    print("Running test_single_head_default...")
    var attn = MultiHeadAttention[DType.float32](16)
    if attn.num_heads != 1 or attn.d_k != 16:
        raise Error("default num_heads should be 1 with d_k == d_model")
    print("  ok default single head, d_k = 16")
    print("test_single_head_default PASSED")


def test_reject_bad_config() raises:
    """Non-positive d_model / num_heads and non-divisible splits must raise."""
    print("Running test_reject_bad_config...")
    try:
        var _ = MultiHeadAttention[DType.float32](0)
        raise Error("Should have rejected d_model = 0")
    except _:
        print("  ok rejected d_model = 0")
    try:
        var _ = MultiHeadAttention[DType.float32](8, num_heads=0)
        raise Error("Should have rejected num_heads = 0")
    except _:
        print("  ok rejected num_heads = 0")
    try:
        var _ = MultiHeadAttention[DType.float32](8, num_heads=3)
        raise Error("Should have rejected indivisible d_model/num_heads")
    except _:
        print("  ok rejected d_model % num_heads != 0")
    print("test_reject_bad_config PASSED")


def test_reject_non_3d() raises:
    """A 2D input (missing the sequence axis) must raise."""
    print("Running test_reject_non_3d...")
    var attn = MultiHeadAttention[DType.float32](8)
    var x = zeros([2, 8], DType.float32)
    try:
        var _ = attn.forward(x)
        raise Error("Should have rejected 2D input")
    except _:
        print("  ok rejected non-3D input")
    print("test_reject_non_3d PASSED")


def test_parameter_count() raises:
    """`parameters()` returns 8 tensors (q/k/v/out weight+bias)."""
    print("Running test_parameter_count...")
    var attn = MultiHeadAttention[DType.float32](8, num_heads=2)
    if len(attn.parameters()) != 8:
        raise Error("attention should expose 8 parameter tensors")
    print("  ok 8 parameter tensors")
    print("test_parameter_count PASSED")


def _check_parity(y: AnyTensor, ref_vals: List[Float64], label: String) raises:
    """Assert every flat entry of `y` matches `ref_vals` to 1e-5."""
    for i in range(len(ref_vals)):
        var d = y.load[DType.float64](i) - ref_vals[i]
        if d < 0:
            d = -d
        if d > 1e-5:
            raise Error(
                "attention parity mismatch ("
                + label
                + ") at index "
                + String(i)
            )


def test_parity_single_head() raises:
    """Single-head non-causal forward matches the PyTorch reference to 1e-5.

    Reference: parity_refs/attention_parity_reference.json["single_head"]["out"]
    (num_heads=1, causal=False, d_model=4, seq=3, batch=2).
    """
    print("Running test_parity_single_head...")
    var ref_vals = List[Float64]()
    ref_vals.append(-0.035843)
    ref_vals.append(-0.007463)
    ref_vals.append(0.020918)
    ref_vals.append(0.049298)
    ref_vals.append(-0.035762)
    ref_vals.append(-0.007392)
    ref_vals.append(0.020979)
    ref_vals.append(0.049350)
    ref_vals.append(-0.035681)
    ref_vals.append(-0.007320)
    ref_vals.append(0.021041)
    ref_vals.append(0.049402)
    ref_vals.append(0.002799)
    ref_vals.append(0.026543)
    ref_vals.append(0.050286)
    ref_vals.append(0.074029)
    ref_vals.append(0.002880)
    ref_vals.append(0.026614)
    ref_vals.append(0.050347)
    ref_vals.append(0.074081)
    ref_vals.append(0.002961)
    ref_vals.append(0.026685)
    ref_vals.append(0.050409)
    ref_vals.append(0.074133)

    var attn = MultiHeadAttention[DType.float64](4, num_heads=1, causal=False)
    _seed_all(attn, 4)
    var x = zeros([2, 3, 4], DType.float64)
    _seed_ramp(x, 24, 0.1, -0.3)

    var y = attn.forward(x)
    _check_parity(y, ref_vals, "single_head")
    print("  ok single-head matches PyTorch SDPA to 1e-5")
    print("test_parity_single_head PASSED")


def test_parity_multi_head_causal() raises:
    """Multi-head causal forward matches the PyTorch reference to 1e-5.

    Reference: parity_refs/attention_parity_reference.json["multi_head"]["out"]
    (num_heads=2, causal=True, d_model=4, seq=3, batch=2). Exercises head split
    and the upper-triangular −∞ mask.
    """
    print("Running test_parity_multi_head_causal...")
    var ref_vals = List[Float64]()
    ref_vals.append(-0.048640)
    ref_vals.append(-0.018724)
    ref_vals.append(0.011192)
    ref_vals.append(0.041108)
    ref_vals.append(-0.042210)
    ref_vals.append(-0.013065)
    ref_vals.append(0.016081)
    ref_vals.append(0.045226)
    ref_vals.append(-0.035678)
    ref_vals.append(-0.007313)
    ref_vals.append(0.021052)
    ref_vals.append(0.049416)
    ref_vals.append(-0.010240)
    ref_vals.append(0.015068)
    ref_vals.append(0.040376)
    ref_vals.append(0.065684)
    ref_vals.append(-0.003718)
    ref_vals.append(0.020810)
    ref_vals.append(0.045339)
    ref_vals.append(0.069867)
    ref_vals.append(0.002967)
    ref_vals.append(0.026700)
    ref_vals.append(0.050434)
    ref_vals.append(0.074167)

    var attn = MultiHeadAttention[DType.float64](4, num_heads=2, causal=True)
    _seed_all(attn, 4)
    var x = zeros([2, 3, 4], DType.float64)
    _seed_ramp(x, 24, 0.1, -0.3)

    var y = attn.forward(x)
    _check_parity(y, ref_vals, "multi_head_causal")
    print("  ok multi-head causal matches PyTorch SDPA to 1e-5")
    print("test_parity_multi_head_causal PASSED")


def test_package_path_export() raises:
    """`from odyssey.core.layers import MultiHeadAttention` must resolve.

    The alias `MultiHeadAttentionPkg` is imported at module top via the package
    `__init__` (not the submodule). Constructing through it proves the public
    export line in `core/layers/__init__.mojo` is present — if that line is
    dropped, this test file fails to COMPILE, so the export can never silently
    vanish again (PR #5640 review MAJOR-1).
    """
    print("Running test_package_path_export...")
    var attn = MultiHeadAttentionPkg[DType.float32](8, num_heads=2)
    if attn.d_model != 8 or attn.num_heads != 2:
        raise Error("package-path export constructed the wrong layer")
    print("  ok imported via odyssey.core.layers package path")
    print("test_package_path_export PASSED")


def test_functional_core_still_broken_tripwire() raises:
    """Tripwire: the functional attention core still RAISES on all inputs.

    `MultiHeadAttention` reimplements the forward instead of delegating to the
    functional core (`core/attention.mojo`) because that core's `transpose(key)`
    reverses ALL axes rather than swapping only the last two, so its QKᵀ matmul
    raises on every documented input (Odyssey#5648, PR #5640 review MAJOR-2).

    This test asserts BOTH functional paths still raise. When Odyssey#5648 is
    fixed, this test starts FAILING — that is intentional: its failure message
    is the actionable TODO to collapse the twin into a real cross-parity test
    asserting `MultiHeadAttention.forward == multi_head_attention_masked` on a
    shared input.
    """
    print("Running test_functional_core_still_broken_tripwire...")
    var x = zeros([2, 3, 4], DType.float64)  # batch=2, seq=3, d_model=4
    var empty = zeros(List[Int](), DType.float64)

    # 3D path: scaled_dot_product_attention_masked reverses [B,S,d_k] fully.
    var sdpa_raised = False
    try:
        var _ = scaled_dot_product_attention_masked(x, x, x, empty)
    except _:
        sdpa_raised = True
    if not sdpa_raised:
        raise Error(
            "functional attention core no longer raises (3D path): the"
            " transpose bug (Odyssey#5648) appears fixed — replace this"
            " tripwire with a true cross-parity test asserting"
            " MultiHeadAttention == multi_head_attention_masked (see PR #5640"
            " review MAJOR-2)"
        )

    # 4D / multi-head path: multi_head_attention_masked, num_heads=2.
    var wq = zeros([4, 4], DType.float64)
    var wk = zeros([4, 4], DType.float64)
    var wv = zeros([4, 4], DType.float64)
    var wo = zeros([4, 4], DType.float64)
    var w = MultiHeadAttentionWeights(wq, wk, wv, wo)
    var mha_raised = False
    try:
        var _ = multi_head_attention_masked(x, x, x, w, 2, empty)
    except _:
        mha_raised = True
    if not mha_raised:
        raise Error(
            "functional attention core no longer raises (multi-head path): the"
            " transpose bug (Odyssey#5648) appears fixed — replace this"
            " tripwire with a true cross-parity test asserting"
            " MultiHeadAttention == multi_head_attention_masked (see PR #5640"
            " review MAJOR-2)"
        )
    print("  ok both functional paths still raise (Odyssey#5648 open)")
    print("test_functional_core_still_broken_tripwire PASSED")


def main() raises:
    """Run all MultiHeadAttention tests."""
    print("=" * 60)
    print("MultiHeadAttention (scaled dot-product self-attention) Test Suite")
    print("=" * 60)
    test_shape_preserved()
    test_single_head_default()
    test_reject_bad_config()
    test_reject_non_3d()
    test_parameter_count()
    test_parity_single_head()
    test_parity_multi_head_causal()
    test_package_path_export()
    test_functional_core_still_broken_tripwire()
    print("=" * 60)
    print("All MultiHeadAttention tests PASSED")
    print("=" * 60)
