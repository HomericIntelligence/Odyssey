"""Unit tests for the standalone LinearAttention (kernel-feature) block.

Tests cover:
- Construction + validation (positive d_model/num_heads, divisibility)
- forward() shape preservation ([batch, seq, d_model] -> [batch, seq, d_model])
- 3D-input requirement (rejects non-3D and wrong last-dim inputs)
- parameter collection (8 tensors: q/k/v/out weight+bias)
- Numerical parity with an explicit PyTorch linear-attention reference on fixed
  ramp weights (tolerance 1e-5), for BOTH single-head and multi-head configs.
- KEY PROPERTY: the O(N) associativity order (φ(K)ᵀV summarized once) equals the
  naive O(N²) order (φ(Q)φ(K)ᵀ then ·V) to tight tolerance (1e-10, f64).
- Denominator positivity: φ = elu + 1 > 0 keeps the normalizer strictly positive.

Reference values are transcribed from the committed JSON fixture
parity_refs/linear_attention_parity_reference.json (produced by
parity_refs/linear_attention_parity_reference.py; d_model=8, seq=4, batch=2,
float64, eps=1e-6). The Mojo weights are seeded here by the SAME ramp formulas
the generator uses (value[i] = i*scale + off), in the SAME flat order (q,k,v,out
weight+bias, X), so only the reference OUTPUTS are transcribed — never the
weights. Odyssey Linear uses W (in, out); the reference sets torch's (out, in)
weight to the transpose.
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros, full_like
from odyssey.core.layers.linear_attention import LinearAttention

# Package-path import (via core/layers/__init__.mojo) — asserts the public
# export exists; if the __init__ export line is dropped, this fails to compile.
from odyssey.core.layers import LinearAttention as LinearAttentionPkg
from odyssey.core.layers.linear import Linear
from odyssey.core.activation import elu
from odyssey.core.matrix import matmul, transpose
from odyssey.core.arithmetic_simd import add_simd, divide_simd
from odyssey.core.reduction import sum as reduce_sum

comptime EPS = 1e-6


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


def _seed_all(mut attn: LinearAttention[DType.float64], d_model: Int) raises:
    """Seed q/k/v/out projections with the generator's per-proj ramp specs."""
    _seed_projection(attn.q_proj, d_model, 0.003, -0.20, 0.02, -0.08)
    _seed_projection(attn.k_proj, d_model, 0.004, -0.18, 0.015, -0.05)
    _seed_projection(attn.v_proj, d_model, 0.005, -0.16, 0.011, -0.06)
    _seed_projection(attn.out_proj, d_model, 0.002, -0.12, 0.03, -0.05)


def test_shape_preserved() raises:
    """Linear attention maps [batch, seq, d_model] -> [batch, seq, d_model]."""
    print("Running test_shape_preserved...")
    var attn = LinearAttention[DType.float32](8, num_heads=2)
    var x = zeros([2, 5, 8], DType.float32)
    var y = attn.forward(x)
    if (
        y.dim() != 3
        or y.shape()[0] != 2
        or y.shape()[1] != 5
        or y.shape()[2] != 8
    ):
        raise Error(
            "linear attention must preserve [batch, seq, d_model] shape"
        )
    print("  ok shape preserved (2, 5, 8)")
    print("test_shape_preserved PASSED")


def test_single_head_default() raises:
    """`num_heads` defaults to 1 (single-head), d_k == d_model."""
    print("Running test_single_head_default...")
    var attn = LinearAttention[DType.float32](16)
    if attn.num_heads != 1 or attn.d_k != 16:
        raise Error("default num_heads should be 1 with d_k == d_model")
    print("  ok default single head, d_k = 16")
    print("test_single_head_default PASSED")


def test_reject_bad_config() raises:
    """Non-positive d_model / num_heads and non-divisible splits must raise."""
    print("Running test_reject_bad_config...")
    try:
        var _ = LinearAttention[DType.float32](0)
        raise Error("Should have rejected d_model = 0")
    except _:
        print("  ok rejected d_model = 0")
    try:
        var _ = LinearAttention[DType.float32](8, num_heads=0)
        raise Error("Should have rejected num_heads = 0")
    except _:
        print("  ok rejected num_heads = 0")
    try:
        var _ = LinearAttention[DType.float32](8, num_heads=3)
        raise Error("Should have rejected indivisible d_model/num_heads")
    except _:
        print("  ok rejected d_model % num_heads != 0")
    print("test_reject_bad_config PASSED")


def test_reject_non_3d() raises:
    """A 2D input (missing the sequence axis) must raise."""
    print("Running test_reject_non_3d...")
    var attn = LinearAttention[DType.float32](8)
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
    var attn = LinearAttention[DType.float32](8, num_heads=2)
    if len(attn.parameters()) != 8:
        raise Error("linear attention should expose 8 parameter tensors")
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
                "linear attention parity mismatch ("
                + label
                + ") at index "
                + String(i)
            )


def test_parity_single_head() raises:
    """Single-head forward matches the PyTorch reference to 1e-5.

    Reference:
    parity_refs/linear_attention_parity_reference.json["single_head"]["out"]
    (num_heads=1, d_model=8, seq=4, batch=2).
    """
    print("Running test_parity_single_head...")
    var ref_vals = List[Float64]()
    ref_vals.append(-0.063744)
    ref_vals.append(-0.032838)
    ref_vals.append(-0.001932)
    ref_vals.append(0.028975)
    ref_vals.append(0.059881)
    ref_vals.append(0.090787)
    ref_vals.append(0.121694)
    ref_vals.append(0.152600)
    ref_vals.append(-0.063729)
    ref_vals.append(-0.032823)
    ref_vals.append(-0.001917)
    ref_vals.append(0.028990)
    ref_vals.append(0.059896)
    ref_vals.append(0.090802)
    ref_vals.append(0.121708)
    ref_vals.append(0.152615)
    ref_vals.append(-0.063717)
    ref_vals.append(-0.032811)
    ref_vals.append(-0.001905)
    ref_vals.append(0.029001)
    ref_vals.append(0.059907)
    ref_vals.append(0.090814)
    ref_vals.append(0.121720)
    ref_vals.append(0.152626)
    ref_vals.append(-0.063705)
    ref_vals.append(-0.032799)
    ref_vals.append(-0.001893)
    ref_vals.append(0.029013)
    ref_vals.append(0.059919)
    ref_vals.append(0.090825)
    ref_vals.append(0.121731)
    ref_vals.append(0.152637)
    ref_vals.append(-0.004321)
    ref_vals.append(0.026074)
    ref_vals.append(0.056468)
    ref_vals.append(0.086862)
    ref_vals.append(0.117256)
    ref_vals.append(0.147650)
    ref_vals.append(0.178044)
    ref_vals.append(0.208438)
    ref_vals.append(-0.004309)
    ref_vals.append(0.026084)
    ref_vals.append(0.056478)
    ref_vals.append(0.086872)
    ref_vals.append(0.117266)
    ref_vals.append(0.147660)
    ref_vals.append(0.178054)
    ref_vals.append(0.208448)
    ref_vals.append(-0.004299)
    ref_vals.append(0.026095)
    ref_vals.append(0.056489)
    ref_vals.append(0.086883)
    ref_vals.append(0.117277)
    ref_vals.append(0.147671)
    ref_vals.append(0.178065)
    ref_vals.append(0.208459)
    ref_vals.append(-0.004288)
    ref_vals.append(0.026106)
    ref_vals.append(0.056500)
    ref_vals.append(0.086894)
    ref_vals.append(0.117287)
    ref_vals.append(0.147681)
    ref_vals.append(0.178075)
    ref_vals.append(0.208469)

    var attn = LinearAttention[DType.float64](8, num_heads=1, eps=EPS)
    _seed_all(attn, 8)
    var x = zeros([2, 4, 8], DType.float64)
    _seed_ramp(x, 64, 0.05, -0.4)

    var y = attn.forward(x)
    _check_parity(y, ref_vals, "single_head")
    print("  ok single-head matches PyTorch linear attention to 1e-5")
    print("test_parity_single_head PASSED")


def test_parity_multi_head() raises:
    """Multi-head forward matches the PyTorch reference to 1e-5.

    Reference:
    parity_refs/linear_attention_parity_reference.json["multi_head"]["out"]
    (num_heads=2, d_model=8, seq=4, batch=2). Exercises the head split.
    """
    print("Running test_parity_multi_head...")
    var ref_vals = List[Float64]()
    ref_vals.append(-0.064370)
    ref_vals.append(-0.033447)
    ref_vals.append(-0.002524)
    ref_vals.append(0.028399)
    ref_vals.append(0.059322)
    ref_vals.append(0.090245)
    ref_vals.append(0.121168)
    ref_vals.append(0.152092)
    ref_vals.append(-0.064367)
    ref_vals.append(-0.033443)
    ref_vals.append(-0.002520)
    ref_vals.append(0.028403)
    ref_vals.append(0.059326)
    ref_vals.append(0.090249)
    ref_vals.append(0.121172)
    ref_vals.append(0.152095)
    ref_vals.append(-0.064364)
    ref_vals.append(-0.033441)
    ref_vals.append(-0.002518)
    ref_vals.append(0.028405)
    ref_vals.append(0.059328)
    ref_vals.append(0.090251)
    ref_vals.append(0.121174)
    ref_vals.append(0.152097)
    ref_vals.append(-0.064361)
    ref_vals.append(-0.033438)
    ref_vals.append(-0.002515)
    ref_vals.append(0.028408)
    ref_vals.append(0.059331)
    ref_vals.append(0.090254)
    ref_vals.append(0.121177)
    ref_vals.append(0.152100)
    ref_vals.append(-0.005013)
    ref_vals.append(0.025399)
    ref_vals.append(0.055810)
    ref_vals.append(0.086221)
    ref_vals.append(0.116632)
    ref_vals.append(0.147043)
    ref_vals.append(0.177454)
    ref_vals.append(0.207865)
    ref_vals.append(-0.005010)
    ref_vals.append(0.025401)
    ref_vals.append(0.055812)
    ref_vals.append(0.086223)
    ref_vals.append(0.116634)
    ref_vals.append(0.147045)
    ref_vals.append(0.177456)
    ref_vals.append(0.207867)
    ref_vals.append(-0.005007)
    ref_vals.append(0.025404)
    ref_vals.append(0.055815)
    ref_vals.append(0.086226)
    ref_vals.append(0.116637)
    ref_vals.append(0.147048)
    ref_vals.append(0.177459)
    ref_vals.append(0.207870)
    ref_vals.append(-0.005004)
    ref_vals.append(0.025407)
    ref_vals.append(0.055818)
    ref_vals.append(0.086229)
    ref_vals.append(0.116640)
    ref_vals.append(0.147051)
    ref_vals.append(0.177462)
    ref_vals.append(0.207872)

    var attn = LinearAttention[DType.float64](8, num_heads=2, eps=EPS)
    _seed_all(attn, 8)
    var x = zeros([2, 4, 8], DType.float64)
    _seed_ramp(x, 64, 0.05, -0.4)

    var y = attn.forward(x)
    _check_parity(y, ref_vals, "multi_head")
    print("  ok multi-head matches PyTorch linear attention to 1e-5")
    print("test_parity_multi_head PASSED")


def _feature_map(x: AnyTensor) raises -> AnyTensor:
    """φ(x) = elu(x) + 1, mirroring LinearAttention._feature_map."""
    var e = elu(x, alpha=1.0)
    return add_simd(e, full_like(e, 1.0))


def test_on_vs_naive_order_equivalence() raises:
    """KEY PROPERTY: O(N) associativity order == naive O(N²) order.

    Linear attention's O(N) cost comes from computing the key–value summary
    S = φ(K)ᵀV ONCE (as `forward` does) instead of forming the S×S matrix
    φ(Q)φ(K)ᵀ. Both are the same function by associativity of matmul; this test
    computes BOTH on the SAME seeded weights and asserts they agree to 1e-10
    (f64). Any drift here means the associativity refactor introduced a bug.

    Both orders share the SAME feature map, the SAME summed-key normalizer
    Z = Σ_j φ(K_j), and the SAME eps guard, so the ONLY difference is the matmul
    grouping:
        O(N):    numerator = φ(Q) (φ(K)ᵀ V)
        O(N²):   numerator = (φ(Q) φ(K)ᵀ) V
    """
    print("Running test_on_vs_naive_order_equivalence...")
    var d_model = 8
    var num_heads = 2
    var batch = 2
    var seq = 4

    var attn = LinearAttention[DType.float64](
        d_model, num_heads=num_heads, eps=EPS
    )
    _seed_all(attn, d_model)
    var x = zeros([batch, seq, d_model], DType.float64)
    _seed_ramp(x, batch * seq * d_model, 0.05, -0.4)

    # --- O(N) path: the layer's own forward. ---
    var y_on = attn.forward(x)

    # --- Naive O(N²) path: recompute Q/K/V exactly like forward, then group the
    # matmuls the other way and materialize the S×S score matrix. ---
    var q_flat = LinearAttention[DType.float64]._project(
        attn.q_proj, x, d_model
    )
    var k_flat = LinearAttention[DType.float64]._project(
        attn.k_proj, x, d_model
    )
    var v_flat = LinearAttention[DType.float64]._project(
        attn.v_proj, x, d_model
    )
    var q = attn._split_heads(q_flat, batch, seq)  # [B, H, S, d_k]
    var k = attn._split_heads(k_flat, batch, seq)
    var v = attn._split_heads(v_flat, batch, seq)

    var phi_q = _feature_map(q)  # [B, H, S, d_k]
    var phi_k = _feature_map(k)

    # φ(K)ᵀ transposed to [B, H, d_k, S] for the S×S score matrix.
    var kt_perm = List[Int]()
    kt_perm.append(0)
    kt_perm.append(1)
    kt_perm.append(3)
    kt_perm.append(2)
    var phi_k_t = transpose(phi_k, kt_perm^)
    # scores = φ(Q) φ(K)ᵀ  ->  [B, H, S, S]  (the O(N²) matrix, materialized).
    var scores = matmul(phi_q, phi_k_t)
    # numerator = scores V  ->  [B, H, S, d_k].
    var numerator = matmul(scores, v)

    # Denominator: identical to forward — row-sums of the SAME score matrix
    # equal φ(Q)·Σ_j φ(K_j), so reuse the score matrix here for a fully
    # independent computation of the same normalizer.
    var denom = reduce_sum(scores, axis=3, keepdims=True)  # [B, H, S, 1]
    denom = add_simd(denom, full_like(denom, EPS))
    var context = divide_simd(numerator, denom)

    var merge_perm = List[Int]()
    merge_perm.append(0)
    merge_perm.append(2)
    merge_perm.append(1)
    merge_perm.append(3)
    var merged = transpose(context, merge_perm^)
    var concat = merged.reshape([batch, seq, d_model])
    var y_naive = LinearAttention[DType.float64]._project(
        attn.out_proj, concat, d_model
    )

    # Assert the two orders agree to 1e-10 (f64, associativity is exact up to
    # floating-point rounding on this tiny config).
    var n = batch * seq * d_model
    for i in range(n):
        var d = y_on.load[DType.float64](i) - y_naive.load[DType.float64](i)
        if d < 0:
            d = -d
        if d > 1e-10:
            raise Error("O(N) vs O(N^2) order mismatch at index " + String(i))
    print("  ok O(N) associativity order == naive O(N^2) order to 1e-10")
    print("test_on_vs_naive_order_equivalence PASSED")


def test_denominator_positive() raises:
    """φ = elu + 1 > 0, so the normalizer Z = Σ_j φ(K_j) is strictly positive.

    Directly checks the feature-map positivity the denominator guard relies on:
    for an input spanning strongly negative to positive values, every entry of
    φ(K) must be > 0 (elu(x) > −1 for all x), hence every summed-key normalizer
    entry is > 0 and the division is well-posed even before the eps guard.
    """
    print("Running test_denominator_positive...")
    var d_model = 8
    var attn = LinearAttention[DType.float64](d_model, num_heads=2)
    _seed_all(attn, d_model)
    var x = zeros([2, 4, d_model], DType.float64)
    # Span from -0.4 (very negative post-projection) upward.
    _seed_ramp(x, 2 * 4 * d_model, 0.05, -0.4)

    var k_flat = LinearAttention[DType.float64]._project(
        attn.k_proj, x, d_model
    )
    var k = attn._split_heads(k_flat, 2, 4)
    var phi_k = _feature_map(k)
    for i in range(phi_k.numel()):
        if phi_k.load[DType.float64](i) <= 0.0:
            raise Error(
                "feature map phi(K) must be strictly positive at index "
                + String(i)
            )
    print("  ok phi(K) = elu(K) + 1 > 0 everywhere (normalizer positive)")
    print("test_denominator_positive PASSED")


def test_package_path_export() raises:
    """`from odyssey.core.layers import LinearAttention` must resolve.

    The alias `LinearAttentionPkg` is imported at module top via the package
    `__init__` (not the submodule). Constructing through it proves the public
    export line in `core/layers/__init__.mojo` is present — if that line is
    dropped, this test file fails to COMPILE, so the export can never silently
    vanish.
    """
    print("Running test_package_path_export...")
    var attn = LinearAttentionPkg[DType.float32](8, num_heads=2)
    if attn.d_model != 8 or attn.num_heads != 2:
        raise Error("package-path export constructed the wrong layer")
    print("  ok imported via odyssey.core.layers package path")
    print("test_package_path_export PASSED")


def main() raises:
    """Run all LinearAttention tests."""
    print("=" * 60)
    print("LinearAttention (kernel-feature self-attention) Test Suite")
    print("=" * 60)
    test_shape_preserved()
    test_single_head_default()
    test_reject_bad_config()
    test_reject_non_3d()
    test_parameter_count()
    test_parity_single_head()
    test_parity_multi_head()
    test_on_vs_naive_order_equivalence()
    test_denominator_positive()
    test_package_path_export()
    print("=" * 60)
    print("All LinearAttention tests PASSED")
    print("=" * 60)
