"""Unit tests for the 1-layer SparseAttention (strided) self-attention block.

Tests cover:
- Construction + validation (positive d_model/num_heads, divisibility,
  window >= 1, stride >= 1)
- forward() shape preservation ([batch, seq, d_model] -> [batch, seq, d_model])
- 3D-input requirement (rejects non-3D and wrong last-dim inputs)
- parameter collection (8 tensors: q/k/v/out weight+bias)
- **Sparsity enforcement**: attention weights are EXACTLY zero (post-softmax) at
  every masked (i, j), and each query row is a proper distribution summing to 1.
- **No all-masked row**: every query attends to at least its self-position, so
  softmax never sees an all−∞ row.
- **Dense equivalence**: with window >= seq (and stride == 1) the sparse mask
  collapses to plain lower-triangular causal, so the output EQUALS dense causal
  `MultiHeadAttention` on the same seeded weights (to 1e-5).
- Numerical parity with an explicit PyTorch sparse (strided) self-attention
  reference on fixed ramp weights (tolerance 1e-5), for BOTH:
    * single-head (num_heads=1, window=2, stride=2)
    * multi-head  (num_heads=2, window=2, stride=3)

Reference values are transcribed from the committed JSON fixture
parity_refs/sparse_attention_parity_reference.json (produced by
parity_refs/sparse_attention_parity_reference.py; d_model=4, seq=4, batch=2,
float64). The Mojo weights are seeded here by the SAME ramp formulas the
generator uses (value[i] = i*scale + off), in the SAME flat order (q,k,v,out
weight+bias, X), so only the reference OUTPUTS are transcribed — never weights.
Odyssey Linear uses W (in, out); the reference sets torch's (out, in) weight to
the transpose.
"""

from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros
from odyssey.core.layers.sparse_attention import SparseAttention

# Package-path import (via core/layers/__init__.mojo) — asserts the public
# export exists; if the __init__ export line is dropped, this fails to compile.
from odyssey.core.layers import SparseAttention as SparseAttentionPkg
from odyssey.core.layers.attention import MultiHeadAttention
from odyssey.core.layers.linear import Linear


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


def _seed_sparse(mut attn: SparseAttention[DType.float64], d_model: Int) raises:
    """Seed q/k/v/out projections with the generator's per-proj ramp specs."""
    _seed_projection(attn.q_proj, d_model, 0.01, -0.15, 0.02, -0.08)
    _seed_projection(attn.k_proj, d_model, 0.013, -0.12, 0.015, -0.05)
    _seed_projection(attn.v_proj, d_model, 0.008, -0.10, 0.011, -0.06)
    _seed_projection(attn.out_proj, d_model, 0.006, -0.08, 0.03, -0.05)


def _seed_dense(
    mut attn: MultiHeadAttention[DType.float64], d_model: Int
) raises:
    """Seed a dense MultiHeadAttention with the SAME per-proj ramp specs.

    Used by the dense-equivalence test so the sparse block (window >= seq) and
    the dense causal block share identical weights.
    """
    _seed_projection(attn.q_proj, d_model, 0.01, -0.15, 0.02, -0.08)
    _seed_projection(attn.k_proj, d_model, 0.013, -0.12, 0.015, -0.05)
    _seed_projection(attn.v_proj, d_model, 0.008, -0.10, 0.011, -0.06)
    _seed_projection(attn.out_proj, d_model, 0.006, -0.08, 0.03, -0.05)


def test_shape_preserved() raises:
    """Sparse attention maps [batch, seq, d_model] -> same shape."""
    print("Running test_shape_preserved...")
    var attn = SparseAttention[DType.float32](8, num_heads=2)
    var x = zeros([2, 5, 8], DType.float32)
    var y = attn.forward(x)
    if (
        y.dim() != 3
        or y.shape()[0] != 2
        or y.shape()[1] != 5
        or y.shape()[2] != 8
    ):
        raise Error("sparse attention must preserve [batch, seq, d_model]")
    print("  ok shape preserved (2, 5, 8)")
    print("test_shape_preserved PASSED")


def test_single_head_default() raises:
    """`num_heads` defaults to 1 (single-head), d_k == d_model."""
    print("Running test_single_head_default...")
    var attn = SparseAttention[DType.float32](16)
    if attn.num_heads != 1 or attn.d_k != 16:
        raise Error("default num_heads should be 1 with d_k == d_model")
    print("  ok default single head, d_k = 16")
    print("test_single_head_default PASSED")


def test_reject_bad_config() raises:
    """Bad d_model/num_heads/window/stride must raise."""
    print("Running test_reject_bad_config...")
    try:
        var _ = SparseAttention[DType.float32](0)
        raise Error("Should have rejected d_model = 0")
    except _:
        print("  ok rejected d_model = 0")
    try:
        var _ = SparseAttention[DType.float32](8, num_heads=0)
        raise Error("Should have rejected num_heads = 0")
    except _:
        print("  ok rejected num_heads = 0")
    try:
        var _ = SparseAttention[DType.float32](8, num_heads=3)
        raise Error("Should have rejected indivisible d_model/num_heads")
    except _:
        print("  ok rejected d_model % num_heads != 0")
    try:
        var _ = SparseAttention[DType.float32](8, window=0)
        raise Error("Should have rejected window = 0")
    except _:
        print("  ok rejected window = 0")
    try:
        var _ = SparseAttention[DType.float32](8, stride=0)
        raise Error("Should have rejected stride = 0")
    except _:
        print("  ok rejected stride = 0")
    print("test_reject_bad_config PASSED")


def test_reject_non_3d() raises:
    """A 2D input (missing the sequence axis) must raise."""
    print("Running test_reject_non_3d...")
    var attn = SparseAttention[DType.float32](8)
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
    var attn = SparseAttention[DType.float32](8, num_heads=2)
    if len(attn.parameters()) != 8:
        raise Error("sparse attention should expose 8 parameter tensors")
    print("  ok 8 parameter tensors")
    print("test_parameter_count PASSED")


def test_is_attended_pattern() raises:
    """`is_attended` matches Child et al. §4.2 strided union set exactly.

    window=2, stride=2, seq=4 — every (i, j) keep/mask decision is checked
    against the reference pattern computed independently here.
    """
    print("Running test_is_attended_pattern...")
    var W = 2
    var ST = 2
    for i in range(4):
        for j in range(4):
            var got = SparseAttention[DType.float32].is_attended(i, j, W, ST)
            var want = (j <= i) and ((i - j) < W or (i - j) % ST == 0)
            if got != want:
                raise Error(
                    "is_attended mismatch at i=" + String(i) + " j=" + String(j)
                )
    # Spot-check the documented cases: i=3 masks j=0 (delta 3, not <2, 3%2=1).
    if SparseAttention[DType.float32].is_attended(3, 0, W, ST):
        raise Error("i=3 j=0 should be masked (window=2, stride=2)")
    if not SparseAttention[DType.float32].is_attended(3, 3, W, ST):
        raise Error("self-position i=3 j=3 must always be attended")
    print("  ok is_attended matches the strided union pattern")
    print("test_is_attended_pattern PASSED")


def test_sparsity_enforced_zero() raises:
    """Post-softmax attention weights are EXACTLY 0 at masked (i, j).

    Single-head window=2 stride=2 seq=4. `attention_weights` returns the
    [B, H, S, S] matrix; every masked (i, j) MUST be 0.0 (not merely small),
    and every query row must sum to 1 (a valid distribution over kept keys).
    """
    print("Running test_sparsity_enforced_zero...")
    var W = 2
    var ST = 2
    var attn = SparseAttention[DType.float64](
        4, num_heads=1, window=W, stride=ST
    )
    _seed_sparse(attn, 4)
    var x = zeros([2, 4, 4], DType.float64)
    _seed_ramp(x, 32, 0.1, -0.3)

    var w = attn.attention_weights(x)  # [B=2, H=1, S=4, S=4]
    if (
        w.dim() != 4
        or w.shape()[0] != 2
        or w.shape()[1] != 1
        or w.shape()[2] != 4
        or w.shape()[3] != 4
    ):
        raise Error("attention_weights must be [batch, heads, seq, seq]")

    var seq = 4
    var head_stride = seq * seq
    for bh in range(2):  # B*H = 2
        var base = bh * head_stride
        for i in range(seq):
            var row_sum = 0.0
            for j in range(seq):
                var val = w.load[DType.float64](base + i * seq + j)
                var attended = SparseAttention[DType.float64].is_attended(
                    i, j, W, ST
                )
                if not attended:
                    # Masked position: must be EXACTLY zero (post-softmax).
                    if val != 0.0:
                        raise Error(
                            "masked weight not exactly 0 at i="
                            + String(i)
                            + " j="
                            + String(j)
                        )
                else:
                    if val < 0.0:
                        raise Error("attended weight is negative")
                row_sum += val
            # Each query row is a valid distribution over kept keys.
            var d = row_sum - 1.0
            if d < 0:
                d = -d
            if d > 1e-9:
                raise Error("attention row does not sum to 1 at i=" + String(i))
    print("  ok masked weights exactly 0, rows sum to 1")
    print("test_sparsity_enforced_zero PASSED")


def test_no_all_masked_row() raises:
    """Every query attends to at least its self-position (no all−∞ row).

    A degenerate config (window=1, a large stride that skips everything but the
    diagonal) still leaves the self-position attended, so softmax never sees an
    all−∞ row (which would produce NaN). The diagonal weight for the first row
    must be exactly 1 and no output entry may be NaN.
    """
    print("Running test_no_all_masked_row...")
    var attn = SparseAttention[DType.float64](
        4, num_heads=1, window=1, stride=100
    )
    _seed_sparse(attn, 4)
    var x = zeros([1, 4, 4], DType.float64)
    _seed_ramp(x, 16, 0.1, -0.3)
    var w = attn.attention_weights(x)  # [1, 1, 4, 4]
    # Row 0 (query i=0) attends only to j=0 -> weight exactly 1.
    var w00 = w.load[DType.float64](0)
    var d = w00 - 1.0
    if d < 0:
        d = -d
    if d > 1e-12:
        raise Error("query 0 self-weight should be exactly 1 (window=1)")
    # No NaN anywhere in the forward output.
    var y = attn.forward(x)
    for i in range(y.shape()[0] * y.shape()[1] * y.shape()[2]):
        var v = y.load[DType.float64](i)
        # NaN != NaN.
        if v != v:
            raise Error("forward produced NaN (all-masked row?)")
    print("  ok self-position always attended; no NaN")
    print("test_no_all_masked_row PASSED")


def _check_parity(y: AnyTensor, ref_vals: List[Float64], label: String) raises:
    """Assert every flat entry of `y` matches `ref_vals` to 1e-5."""
    for i in range(len(ref_vals)):
        var d = y.load[DType.float64](i) - ref_vals[i]
        if d < 0:
            d = -d
        if d > 1e-5:
            raise Error(
                "sparse attention parity mismatch ("
                + label
                + ") at index "
                + String(i)
            )


def test_parity_single_head() raises:
    """Single-head sparse forward matches the PyTorch reference to 1e-5.

    Reference:
    parity_refs/sparse_attention_parity_reference.json["single_head"]["out"]
    (num_heads=1, window=2, stride=2, d_model=4, seq=4, batch=2).
    """
    print("Running test_parity_single_head...")
    var ref_vals = List[Float64]()
    ref_vals.append(-0.048640)
    ref_vals.append(-0.018724)
    ref_vals.append(0.011192)
    ref_vals.append(0.041108)
    ref_vals.append(-0.042211)
    ref_vals.append(-0.013066)
    ref_vals.append(0.016078)
    ref_vals.append(0.045223)
    ref_vals.append(-0.035681)
    ref_vals.append(-0.007320)
    ref_vals.append(0.021041)
    ref_vals.append(0.049402)
    ref_vals.append(-0.022801)
    ref_vals.append(0.004015)
    ref_vals.append(0.030830)
    ref_vals.append(0.057645)
    ref_vals.append(0.002560)
    ref_vals.append(0.026332)
    ref_vals.append(0.050104)
    ref_vals.append(0.073876)
    ref_vals.append(0.009110)
    ref_vals.append(0.032096)
    ref_vals.append(0.055082)
    ref_vals.append(0.078068)
    ref_vals.append(0.015842)
    ref_vals.append(0.038020)
    ref_vals.append(0.060198)
    ref_vals.append(0.082376)
    ref_vals.append(0.028722)
    ref_vals.append(0.049355)
    ref_vals.append(0.069987)
    ref_vals.append(0.090620)

    var attn = SparseAttention[DType.float64](
        4, num_heads=1, window=2, stride=2
    )
    _seed_sparse(attn, 4)
    var x = zeros([2, 4, 4], DType.float64)
    _seed_ramp(x, 32, 0.1, -0.3)

    var y = attn.forward(x)
    _check_parity(y, ref_vals, "single_head")
    print("  ok single-head matches PyTorch sparse attention to 1e-5")
    print("test_parity_single_head PASSED")


def test_parity_multi_head() raises:
    """Multi-head sparse forward matches the PyTorch reference to 1e-5.

    Reference:
    parity_refs/sparse_attention_parity_reference.json["multi_head"]["out"]
    (num_heads=2, window=2, stride=3, d_model=4, seq=4, batch=2). Exercises head
    split AND the strided-union sparse mask.
    """
    print("Running test_parity_multi_head...")
    var ref_vals = List[Float64]()
    ref_vals.append(-0.048640)
    ref_vals.append(-0.018724)
    ref_vals.append(0.011192)
    ref_vals.append(0.041108)
    ref_vals.append(-0.042210)
    ref_vals.append(-0.013065)
    ref_vals.append(0.016081)
    ref_vals.append(0.045226)
    ref_vals.append(-0.029379)
    ref_vals.append(-0.001773)
    ref_vals.append(0.025833)
    ref_vals.append(0.053440)
    ref_vals.append(-0.026742)
    ref_vals.append(0.000560)
    ref_vals.append(0.027863)
    ref_vals.append(0.055166)
    ref_vals.append(0.002560)
    ref_vals.append(0.026332)
    ref_vals.append(0.050104)
    ref_vals.append(0.073876)
    ref_vals.append(0.009112)
    ref_vals.append(0.032102)
    ref_vals.append(0.055092)
    ref_vals.append(0.078081)
    ref_vals.append(0.021943)
    ref_vals.append(0.043394)
    ref_vals.append(0.064844)
    ref_vals.append(0.086295)
    ref_vals.append(0.025196)
    ref_vals.append(0.046284)
    ref_vals.append(0.067373)
    ref_vals.append(0.088461)

    var attn = SparseAttention[DType.float64](
        4, num_heads=2, window=2, stride=3
    )
    _seed_sparse(attn, 4)
    var x = zeros([2, 4, 4], DType.float64)
    _seed_ramp(x, 32, 0.1, -0.3)

    var y = attn.forward(x)
    _check_parity(y, ref_vals, "multi_head")
    print("  ok multi-head matches PyTorch sparse attention to 1e-5")
    print("test_parity_multi_head PASSED")


def test_dense_equivalence() raises:
    """`window >= seq` (stride=1) EQUALS dense causal MultiHeadAttention.

    When the local window already covers every causal key, the strided-union
    mask degenerates to plain lower-triangular causal, so the sparse block must
    produce byte-comparable output (to 1e-5) to `MultiHeadAttention(causal=True)`
    seeded with the SAME weights. Both are also checked against the shared
    PyTorch `dense_equiv` fixture (num_heads=1, window=4, stride=1, seq=4).
    """
    print("Running test_dense_equivalence...")
    # Reference fixture: dense_equiv["out"].
    var ref_vals = List[Float64]()
    ref_vals.append(-0.048640)
    ref_vals.append(-0.018724)
    ref_vals.append(0.011192)
    ref_vals.append(0.041108)
    ref_vals.append(-0.042211)
    ref_vals.append(-0.013066)
    ref_vals.append(0.016078)
    ref_vals.append(0.045223)
    ref_vals.append(-0.035681)
    ref_vals.append(-0.007320)
    ref_vals.append(0.021041)
    ref_vals.append(0.049402)
    ref_vals.append(-0.028991)
    ref_vals.append(-0.001433)
    ref_vals.append(0.026125)
    ref_vals.append(0.053683)
    ref_vals.append(0.002560)
    ref_vals.append(0.026332)
    ref_vals.append(0.050104)
    ref_vals.append(0.073876)
    ref_vals.append(0.009110)
    ref_vals.append(0.032096)
    ref_vals.append(0.055082)
    ref_vals.append(0.078068)
    ref_vals.append(0.015842)
    ref_vals.append(0.038020)
    ref_vals.append(0.060198)
    ref_vals.append(0.082376)
    ref_vals.append(0.022814)
    ref_vals.append(0.044155)
    ref_vals.append(0.065497)
    ref_vals.append(0.086839)

    var x = zeros([2, 4, 4], DType.float64)
    _seed_ramp(x, 32, 0.1, -0.3)

    # Sparse block with window = seq = 4, stride = 1 (mask -> plain causal).
    var sparse = SparseAttention[DType.float64](
        4, num_heads=1, window=4, stride=1
    )
    _seed_sparse(sparse, 4)
    var y_sparse = sparse.forward(x)

    # Dense causal MultiHeadAttention with identical seeded weights.
    var dense = MultiHeadAttention[DType.float64](4, num_heads=1, causal=True)
    _seed_dense(dense, 4)
    var y_dense = dense.forward(x)

    # 1) sparse (dense config) matches the PyTorch dense-causal fixture.
    _check_parity(y_sparse, ref_vals, "dense_equiv_fixture")
    # 2) sparse (dense config) EQUALS the Odyssey dense causal layer.
    for i in range(32):
        var d = y_sparse.load[DType.float64](i) - y_dense.load[DType.float64](i)
        if d < 0:
            d = -d
        if d > 1e-5:
            raise Error(
                "dense-equivalence mismatch vs MultiHeadAttention at index "
                + String(i)
            )
    print("  ok window>=seq equals dense causal MultiHeadAttention (1e-5)")
    print("test_dense_equivalence PASSED")


def test_package_path_export() raises:
    """`from odyssey.core.layers import SparseAttention` must resolve.

    Constructing through the package-path alias proves the public export line in
    `core/layers/__init__.mojo` is present — if that line is dropped, this test
    file fails to COMPILE, so the export can never silently vanish.
    """
    print("Running test_package_path_export...")
    var attn = SparseAttentionPkg[DType.float32](8, num_heads=2)
    if attn.d_model != 8 or attn.num_heads != 2:
        raise Error("package-path export constructed the wrong layer")
    print("  ok imported via odyssey.core.layers package path")
    print("test_package_path_export PASSED")


def main() raises:
    """Run all SparseAttention tests."""
    print("=" * 60)
    print("SparseAttention (strided factorized self-attention) Test Suite")
    print("=" * 60)
    test_shape_preserved()
    test_single_head_default()
    test_reject_bad_config()
    test_reject_non_3d()
    test_parameter_count()
    test_is_attended_pattern()
    test_sparsity_enforced_zero()
    test_no_all_masked_row()
    test_parity_single_head()
    test_parity_multi_head()
    test_dense_equivalence()
    test_package_path_export()
    print("=" * 60)
    print("All SparseAttention tests PASSED")
    print("=" * 60)
