"""Forward-mode Jacobian-vector product tests.

The fixtures are intentionally vector-valued and asymmetric.  They exercise
both terms of the matrix-product rule and a deterministic two-layer model
rather than accepting a scalar directional-derivative shortcut.
"""

from odyssey.autograd import (
    DualTensor,
    JVPFunction,
    dual_add,
    dual_constant,
    dual_linear,
    dual_matmul,
    dual_multiply,
    dual_relu,
    dual_subtract,
    jvp,
    jvp_only,
)
from odyssey.tensor.any_tensor import AnyTensor
from odyssey.tensor.tensor_creation import zeros


def _tensor(shape: List[Int], values: List[Float64]) raises -> AnyTensor:
    if len(values) == 0:
        raise Error("test fixture values must not be empty")
    var result = zeros(shape, DType.float32)
    if result.numel() != len(values):
        raise Error("test fixture shape/value count mismatch")
    for i in range(len(values)):
        result._set_float64(i, values[i])
    return result^


def _abs(value: Float64) -> Float64:
    return value if value >= 0.0 else -value


def _assert_tensor_close(
    label: String,
    actual: AnyTensor,
    expected: List[Float64],
    tolerance: Float64 = 1e-5,
) raises:
    if actual.numel() != len(expected):
        raise Error(label + ": element-count mismatch")
    for i in range(len(expected)):
        var got = actual._get_float64(i)
        if _abs(got - expected[i]) > tolerance:
            raise Error(
                label
                + "["
                + String(i)
                + "]: expected "
                + String(expected[i])
                + ", got "
                + String(got)
            )


struct IdentityFn(JVPFunction):
    def __init__(out self):
        pass

    def __call__(self, input: DualTensor) raises -> DualTensor:
        return input.copy()


struct PolynomialFn(JVPFunction):
    """Elementwise polynomial f(x) = x² + x."""

    def __init__(out self):
        pass

    def __call__(self, input: DualTensor) raises -> DualTensor:
        var squared = dual_multiply(input, input)
        return dual_add(squared, input)


struct CompositionFn(JVPFunction):
    """Composition f(g(x)) with g(x) = x² and f(g) = g²."""

    def __init__(out self):
        pass

    def __call__(self, input: DualTensor) raises -> DualTensor:
        var squared = dual_multiply(input, input)
        return dual_multiply(squared, squared)


@fieldwise_init
struct BilinearMatmulFn(JVPFunction):
    var rhs: DualTensor

    def __call__(self, lhs: DualTensor) raises -> DualTensor:
        return dual_matmul(lhs, self.rhs)


struct ReluFn(JVPFunction):
    def __init__(out self):
        pass

    def __call__(self, input: DualTensor) raises -> DualTensor:
        return dual_relu(input)


@fieldwise_init
struct TwoLayerMLP(JVPFunction):
    var weight1: DualTensor
    var bias1: DualTensor
    var weight2: DualTensor
    var bias2: DualTensor

    def __call__(self, input: DualTensor) raises -> DualTensor:
        var hidden_pre = dual_linear(input, self.weight1, self.bias1)
        var hidden = dual_relu(hidden_pre)
        return dual_linear(hidden, self.weight2, self.bias2)


def test_public_import_identity_polynomial_and_composition() raises:
    var shape: List[Int] = [3]
    var x_values: List[Float64] = [-2.0, 0.5, 3.0]
    var v_values: List[Float64] = [0.25, -2.0, 1.5]

    var identity = jvp(
        IdentityFn(), _tensor(shape, x_values), _tensor(shape, v_values)
    )
    _assert_tensor_close("identity primal", identity.primal, x_values)
    _assert_tensor_close("identity tangent", identity.tangent, v_values)

    var polynomial = jvp(
        PolynomialFn(), _tensor(shape, x_values), _tensor(shape, v_values)
    )
    var polynomial_primal: List[Float64] = [2.0, 0.75, 12.0]
    var polynomial_tangent: List[Float64] = [-0.75, -4.0, 10.5]
    _assert_tensor_close(
        "polynomial primal", polynomial.primal, polynomial_primal
    )
    _assert_tensor_close(
        "polynomial tangent", polynomial.tangent, polynomial_tangent
    )

    var composed = jvp(
        CompositionFn(), _tensor(shape, x_values), _tensor(shape, v_values)
    )
    var composed_primal: List[Float64] = [16.0, 0.0625, 81.0]
    var composed_tangent: List[Float64] = [-8.0, -1.0, 162.0]
    _assert_tensor_close("composition primal", composed.primal, composed_primal)
    _assert_tensor_close(
        "composition tangent", composed.tangent, composed_tangent
    )

    # Exercise the public constant/subtraction rules as part of the import test.
    var constant = dual_constant(_tensor(shape, x_values))
    var difference = dual_subtract(constant, constant)
    var zero_values: List[Float64] = [0.0, 0.0, 0.0]
    _assert_tensor_close("constant subtraction", difference.primal, zero_values)
    _assert_tensor_close("constant tangent", difference.tangent, zero_values)


def test_matmul_uses_both_product_rule_terms() raises:
    var matrix_shape: List[Int] = [2, 2]
    var a_values: List[Float64] = [1.0, 2.0, 3.0, 4.0]
    var a_dot_values: List[Float64] = [0.5, -1.0, 2.0, 0.25]
    var b_values: List[Float64] = [2.0, -1.0, 0.5, 3.0]
    var b_dot_values: List[Float64] = [-2.0, 1.0, 4.0, -0.5]
    var rhs = DualTensor(
        _tensor(matrix_shape, b_values), _tensor(matrix_shape, b_dot_values)
    )
    var result = jvp(
        BilinearMatmulFn(rhs^),
        _tensor(matrix_shape, a_values),
        _tensor(matrix_shape, a_dot_values),
    )

    var expected_primal: List[Float64] = [3.0, 5.0, 8.0, 9.0]
    # A_dot @ B + A @ B_dot.  Neither term alone equals this fixture.
    var expected_tangent: List[Float64] = [6.5, -3.5, 14.125, -0.25]
    _assert_tensor_close("matmul primal", result.primal, expected_primal)
    _assert_tensor_close("matmul tangent", result.tangent, expected_tangent)


def test_relu_tangent_mask_away_from_kink() raises:
    var shape: List[Int] = [4]
    var primal_values: List[Float64] = [-2.0, -0.5, 0.25, 3.0]
    var tangent_values: List[Float64] = [1.0, 2.0, -3.0, 4.0]
    var result = jvp(
        ReluFn(),
        _tensor(shape, primal_values),
        _tensor(shape, tangent_values),
    )
    var expected_primal: List[Float64] = [0.0, 0.0, 0.25, 3.0]
    var expected_tangent: List[Float64] = [0.0, 0.0, -3.0, 4.0]
    _assert_tensor_close("relu primal", result.primal, expected_primal)
    _assert_tensor_close("relu tangent", result.tangent, expected_tangent)


def test_relu_at_zero_uses_zero_tangent() raises:
    var shape: List[Int] = [1]
    var primal_values: List[Float64] = [0.0]
    var tangent_values: List[Float64] = [7.5]
    var result = jvp(
        ReluFn(),
        _tensor(shape, primal_values),
        _tensor(shape, tangent_values),
    )
    var expected: List[Float64] = [0.0]
    _assert_tensor_close("relu-at-zero primal", result.primal, expected)
    _assert_tensor_close("relu-at-zero tangent", result.tangent, expected)


def test_asymmetric_seed_42_two_layer_mlp_formula_parity() raises:
    # Fixed seed-42 fixture, recorded explicitly so model state cannot vary with
    # global RNG state.  Shapes are x:[1,2], W1:[2,3], W2:[3,2].
    var input_shape: List[Int] = [1, 2]
    var hidden_weight_shape: List[Int] = [2, 3]
    var hidden_bias_shape: List[Int] = [3]
    var output_weight_shape: List[Int] = [3, 2]
    var output_bias_shape: List[Int] = [2]
    var x_values: List[Float64] = [0.75, -1.25]
    var v_values: List[Float64] = [0.2, -0.4]
    var w1_values: List[Float64] = [0.8, -0.3, 0.5, -0.2, 0.7, -0.6]
    var b1_values: List[Float64] = [0.1, -0.2, 0.05]
    var w2_values: List[Float64] = [-0.4, 0.9, 0.3, -0.5, 0.8, 0.2]
    var b2_values: List[Float64] = [-0.15, 0.4]

    var model = TwoLayerMLP(
        dual_constant(_tensor(hidden_weight_shape, w1_values)),
        dual_constant(_tensor(hidden_bias_shape, b1_values)),
        dual_constant(_tensor(output_weight_shape, w2_values)),
        dual_constant(_tensor(output_bias_shape, b2_values)),
    )
    var result = jvp(
        model, _tensor(input_shape, x_values), _tensor(input_shape, v_values)
    )

    # Analytic formula:
    # h = relu(x @ W1 + b1), h_dot = (v @ W1) * 1[h_pre > 0]
    # y = h @ W2 + b2,          y_dot = h_dot @ W2
    var expected_primal: List[Float64] = [0.41, 1.49]
    var expected_tangent: List[Float64] = [0.176, 0.284]
    _assert_tensor_close("two-layer MLP primal", result.primal, expected_primal)
    _assert_tensor_close(
        "two-layer MLP tangent", result.tangent, expected_tangent
    )


def test_jvp_only_parity_and_lower_retained_payload() raises:
    var shape: List[Int] = [3]
    var x_values: List[Float64] = [-1.5, 0.25, 2.0]
    var v_values: List[Float64] = [0.5, -3.0, 1.25]
    var pair = jvp(
        PolynomialFn(), _tensor(shape, x_values), _tensor(shape, v_values)
    )
    var tangent_only = jvp_only(
        PolynomialFn(), _tensor(shape, x_values), _tensor(shape, v_values)
    )
    for i in range(tangent_only.numel()):
        if (
            _abs(tangent_only._get_float64(i) - pair.tangent._get_float64(i))
            > 1e-5
        ):
            raise Error("jvp_only tangent differs from jvp")

    # This checks numerical parity and the smaller retained return payload.
    # jvp_only still computes primal intermediates; this is not a claim about
    # peak memory or allocation count.
    if tangent_only.numel() >= pair.primal.numel() + pair.tangent.numel():
        raise Error("jvp_only did not reduce the retained output payload")


def _expect_invalid_pair(
    label: String, var primal: AnyTensor, var tangent: AnyTensor
) raises:
    var did_raise = False
    try:
        _ = DualTensor(primal^, tangent^)
    except:
        did_raise = True
    if not did_raise:
        raise Error(label + ": expected DualTensor validation to fail")


def test_shape_dtype_integer_and_bfloat16_validation() raises:
    var vector_shape: List[Int] = [2]
    var matrix_shape: List[Int] = [1, 2]
    var pair_values: List[Float64] = [1.0, 2.0]
    _expect_invalid_pair(
        "shape mismatch",
        _tensor(vector_shape, pair_values),
        _tensor(matrix_shape, pair_values),
    )

    var float_seed = _tensor(vector_shape, pair_values)
    var float64_tangent = zeros(vector_shape, DType.float64)
    _expect_invalid_pair("dtype mismatch", float_seed^, float64_tangent^)

    var integer_primal = zeros(vector_shape, DType.int32)
    var integer_tangent = zeros(vector_shape, DType.int32)
    _expect_invalid_pair("integer pair", integer_primal^, integer_tangent^)

    # BF16 is floating-point, but the arithmetic/matmul dispatchers used by
    # these forward rules do not all support it yet.  Reject it at the seed
    # boundary instead of allowing a later, operation-dependent failure.
    var bf16_primal = zeros(vector_shape, DType.bfloat16)
    var bf16_tangent = zeros(vector_shape, DType.bfloat16)
    var bf16_did_raise = False
    try:
        _ = DualTensor(bf16_primal^, bf16_tangent^)
    except error:
        bf16_did_raise = True
        var message = String(error)
        if (
            "supports only float16, float32, and float64" not in message
            or "bfloat16" not in message
        ):
            raise Error("bfloat16 rejection diagnostic is unclear: " + message)
    if not bf16_did_raise:
        raise Error("bfloat16 pair: expected DualTensor validation to fail")


def main() raises:
    test_public_import_identity_polynomial_and_composition()
    test_matmul_uses_both_product_rule_terms()
    test_relu_tangent_mask_away_from_kink()
    test_relu_at_zero_uses_zero_tangent()
    test_asymmetric_seed_42_two_layer_mlp_formula_parity()
    test_jvp_only_parity_and_lower_retained_payload()
    test_shape_dtype_integer_and_bfloat16_validation()
    print("All vector JVP tests passed")
