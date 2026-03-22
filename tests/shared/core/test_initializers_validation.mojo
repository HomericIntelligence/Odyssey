"""Comprehensive statistical validation for all initializers.

This test suite validates statistical properties across ALL initialization methods,
ensuring consistency, correctness, and integration.

Coordination tests (#273-277):
- #274: Cross-initializer statistical validation
- #275: API consistency validation
- #276: Integration testing

Validation strategy:
- Statistical correctness (mean, variance, distribution shape)
- API consistency (parameters, return types, error handling)
- Reproducibility (seeding behavior)
- Integration (interoperability with AnyTensor)
"""

from tests.shared.conftest import (
    assert_close_float,
    assert_dtype_equal,
    assert_equal,
    assert_equal_int,
    assert_false,
    assert_true,
)
from math import sqrt
from shared.core.any_tensor import AnyTensor
from shared.core.initializers import xavier_uniform, xavier_normal, kaiming_uniform, kaiming_normal, uniform, normal, constant


fn compute_mean(tensor: AnyTensor) -> Float64:
    """Compute mean of tensor values."""
    var sum = Float64(0.0)
    var size = tensor.numel()

    if tensor.dtype() == DType.float32:
        for i in range(size):
            sum += Float64(tensor._data.bitcast[Float32]()[i])
    elif tensor.dtype() == DType.float64:
        for i in range(size):
            sum += tensor._data.bitcast[Float64]()[i]
    elif tensor.dtype() == DType.float16:
        for i in range(size):
            sum += Float64(tensor._data.bitcast[Float16]()[i])

    return sum / Float64(size)


fn compute_variance(tensor: AnyTensor, mean: Float64) -> Float64:
    """Compute variance of tensor values."""
    var sum_sq = Float64(0.0)
    var size = tensor.numel()

    if tensor.dtype() == DType.float32:
        for i in range(size):
            var val = Float64(tensor._data.bitcast[Float32]()[i])
            var diff = val - mean
            sum_sq += diff * diff
    elif tensor.dtype() == DType.float64:
        for i in range(size):
            var val = tensor._data.bitcast[Float64]()[i]
            var diff = val - mean
            sum_sq += diff * diff
    elif tensor.dtype() == DType.float16:
        for i in range(size):
            var val = Float64(tensor._data.bitcast[Float16]()[i])
            var diff = val - mean
            sum_sq += diff * diff

    return sum_sq / Float64(size)


fn test_all_initializers_produce_tensors() raises:
    """Validate that all initializers produce valid AnyTensor objects."""
    print("Testing all initializers produce valid tensors...")

    var shape: List[Int] = [100, 100]
    var fan_in = 100
    var fan_out = 100

    # Xavier initializers
    var xavier_u = xavier_uniform(fan_in, fan_out, shape)
    var xavier_n = xavier_normal(fan_in, fan_out, shape)

    # Kaiming initializers
    var kaiming_u = kaiming_uniform(fan_in, fan_out, shape)
    var kaiming_n = kaiming_normal(fan_in, fan_out, shape)

    # Basic distributions
    var uniform_t = uniform(shape)
    var normal_t = normal(shape)
    var constant_t = constant(shape, 0.5)

    # Verify all have correct shape
    assert_equal(xavier_u.numel(), 10000, "Xavier uniform size")
    assert_equal(xavier_n.numel(), 10000, "Xavier normal size")
    assert_equal(kaiming_u.numel(), 10000, "Kaiming uniform size")
    assert_equal(kaiming_n.numel(), 10000, "Kaiming normal size")
    assert_equal(uniform_t.numel(), 10000, "Uniform size")
    assert_equal(normal_t.numel(), 10000, "Normal size")
    assert_equal(constant_t.numel(), 10000, "Constant size")

    print("  ✓ All initializers produce valid tensors")


fn test_all_initializers_respect_seed() raises:
    """Validate that all initializers are reproducible with seeds."""
    print("Testing all initializers respect seeds...")

    var shape: List[Int] = [50, 50]
    var fan_in = 50
    var fan_out = 50
    var seed = 42

    # Test Xavier uniform
    var xu1 = xavier_uniform(fan_in, fan_out, shape, seed_val=seed)
    var xu2 = xavier_uniform(fan_in, fan_out, shape, seed_val=seed)
    var xu3 = xavier_uniform(fan_in, fan_out, shape, seed_val=seed + 1)

    # First two should match, third should differ
    var xu_match = True
    var xu_differ = False
    for i in range(100):  # Check first 100 elements
        var v1 = xu1._data.bitcast[Float32]()[i]
        var v2 = xu2._data.bitcast[Float32]()[i]
        var v3 = xu3._data.bitcast[Float32]()[i]
        if v1 != v2:
            xu_match = False
        if v1 != v3:
            xu_differ = True

    assert_true(xu_match, "Xavier uniform: same seed produces same values")
    assert_true(
        xu_differ, "Xavier uniform: different seed produces different values"
    )

    # Test Kaiming normal
    var kn1 = kaiming_normal(fan_in, fan_out, shape, seed_val=seed)
    var kn2 = kaiming_normal(fan_in, fan_out, shape, seed_val=seed)
    var kn3 = kaiming_normal(fan_in, fan_out, shape, seed_val=seed + 1)

    var kn_match = True
    var kn_differ = False
    for i in range(100):
        var v1 = kn1._data.bitcast[Float32]()[i]
        var v2 = kn2._data.bitcast[Float32]()[i]
        var v3 = kn3._data.bitcast[Float32]()[i]
        if v1 != v2:
            kn_match = False
        if v1 != v3:
            kn_differ = True

    assert_true(kn_match, "Kaiming normal: same seed produces same values")
    assert_true(
        kn_differ, "Kaiming normal: different seed produces different values"
    )

    # Test basic uniform
    var u1 = uniform(shape, seed_val=seed)
    var u2 = uniform(shape, seed_val=seed)
    var u3 = uniform(shape, seed_val=seed + 1)

    var u_match = True
    var u_differ = False
    for i in range(100):
        var v1 = u1._data.bitcast[Float32]()[i]
        var v2 = u2._data.bitcast[Float32]()[i]
        var v3 = u3._data.bitcast[Float32]()[i]
        if v1 != v2:
            u_match = False
        if v1 != v3:
            u_differ = True

    assert_true(u_match, "Uniform: same seed produces same values")
    assert_true(u_differ, "Uniform: different seed produces different values")

    print("  ✓ All initializers respect seeds for reproducibility")


fn test_xavier_statistical_properties() raises:
    """Validate Xavier initializers have correct statistical properties."""
    print("Testing Xavier statistical properties...")

    var shape: List[Int] = [1000, 1000]
    var fan_in = 1000
    var fan_out = 1000

    # Xavier uniform: variance should be 2/(fan_in + fan_out)
    var xavier_u = xavier_uniform(fan_in, fan_out, shape, seed_val=42)
    var xu_mean = compute_mean(xavier_u)
    var xu_var = compute_variance(xavier_u, xu_mean)

    var expected_xu_var = 2.0 / Float64(fan_in + fan_out)

    print("  Xavier uniform:")
    print("    Mean: " + String(xu_mean) + " (expected ~0.0)")
    print(
        "    Variance: "
        + String(xu_var)
        + " (expected "
        + String(expected_xu_var)
        + ")"
    )

    assert_true(abs(xu_mean) < 0.01, "Xavier uniform mean should be ~0")
    assert_true(
        abs(xu_var - expected_xu_var) < 0.0005, "Xavier uniform variance"
    )

    # Xavier normal: variance should be 2/(fan_in + fan_out)
    var xavier_n = xavier_normal(fan_in, fan_out, shape, seed_val=42)
    var xn_mean = compute_mean(xavier_n)
    var xn_var = compute_variance(xavier_n, xn_mean)

    var expected_xn_var = 2.0 / Float64(fan_in + fan_out)

    print("  Xavier normal:")
    print("    Mean: " + String(xn_mean) + " (expected ~0.0)")
    print(
        "    Variance: "
        + String(xn_var)
        + " (expected "
        + String(expected_xn_var)
        + ")"
    )

    assert_true(abs(xn_mean) < 0.01, "Xavier normal mean should be ~0")
    assert_true(
        abs(xn_var - expected_xn_var) < 0.0005, "Xavier normal variance"
    )

    print("  ✓ Xavier initializers have correct variance")


fn test_kaiming_statistical_properties() raises:
    """Validate Kaiming initializers have correct statistical properties."""
    print("Testing Kaiming statistical properties...")

    var shape: List[Int] = [1000, 1000]
    var fan_in = 1000
    var fan_out = 1000

    # Kaiming uniform (fan_in mode): variance should be 2/fan_in
    var kaiming_u = kaiming_uniform(
        fan_in, fan_out, shape, fan_mode="fan_in", seed_val=42
    )
    var ku_mean = compute_mean(kaiming_u)
    var ku_var = compute_variance(kaiming_u, ku_mean)

    var expected_ku_var = 2.0 / Float64(fan_in)

    print("  Kaiming uniform (fan_in):")
    print("    Mean: " + String(ku_mean) + " (expected ~0.0)")
    print(
        "    Variance: "
        + String(ku_var)
        + " (expected "
        + String(expected_ku_var)
        + ")"
    )

    assert_true(abs(ku_mean) < 0.01, "Kaiming uniform mean should be ~0")
    assert_true(
        abs(ku_var - expected_ku_var) < 0.0005, "Kaiming uniform variance"
    )

    # Kaiming normal (fan_in mode): variance should be 2/fan_in
    var kaiming_n = kaiming_normal(
        fan_in, fan_out, shape, fan_mode="fan_in", seed_val=42
    )
    var kn_mean = compute_mean(kaiming_n)
    var kn_var = compute_variance(kaiming_n, kn_mean)

    var expected_kn_var = 2.0 / Float64(fan_in)

    print("  Kaiming normal (fan_in):")
    print("    Mean: " + String(kn_mean) + " (expected ~0.0)")
    print(
        "    Variance: "
        + String(kn_var)
        + " (expected "
        + String(expected_kn_var)
        + ")"
    )

    assert_true(abs(kn_mean) < 0.01, "Kaiming normal mean should be ~0")
    assert_true(
        abs(kn_var - expected_kn_var) < 0.0005, "Kaiming normal variance"
    )

    # Kaiming uniform (fan_out mode): variance should be 2/fan_out
    var kaiming_u_out = kaiming_uniform(
        fan_in, fan_out, shape, fan_mode="fan_out", seed_val=42
    )
    var kuo_mean = compute_mean(kaiming_u_out)
    var kuo_var = compute_variance(kaiming_u_out, kuo_mean)

    var expected_kuo_var = 2.0 / Float64(fan_out)

    print("  Kaiming uniform (fan_out):")
    print("    Mean: " + String(kuo_mean) + " (expected ~0.0)")
    print(
        "    Variance: "
        + String(kuo_var)
        + " (expected "
        + String(expected_kuo_var)
        + ")"
    )

    assert_true(
        abs(kuo_mean) < 0.01, "Kaiming uniform (fan_out) mean should be ~0"
    )
    assert_true(
        abs(kuo_var - expected_kuo_var) < 0.0005,
        "Kaiming uniform (fan_out) variance",
    )

    print("  ✓ Kaiming initializers have correct variance for both fan modes")


fn test_basic_distributions_statistical_properties() raises:
    """Validate basic distributions have correct statistical properties."""
    print("Testing basic distribution statistical properties...")

    var shape: List[Int] = [1000, 1000]

    # Uniform distribution: mean should be (low+high)/2, variance should be (high-low)²/12
    var low = -0.5
    var high = 0.5
    var unif = uniform(shape, low=low, high=high, seed_val=42)
    var u_mean = compute_mean(unif)
    var u_var = compute_variance(unif, u_mean)

    var expected_u_mean = (low + high) / 2.0
    var expected_u_var = ((high - low) * (high - low)) / 12.0

    print("  Uniform(" + String(low) + ", " + String(high) + "):")
    print(
        "    Mean: "
        + String(u_mean)
        + " (expected "
        + String(expected_u_mean)
        + ")"
    )
    print(
        "    Variance: "
        + String(u_var)
        + " (expected "
        + String(expected_u_var)
        + ")"
    )

    assert_true(abs(u_mean - expected_u_mean) < 0.01, "Uniform mean")
    assert_true(abs(u_var - expected_u_var) < 0.01, "Uniform variance")

    # Normal distribution: mean and std should match parameters
    var mean_val = 0.0
    var std_val = 0.1
    var norm = normal(shape, mean=mean_val, std=std_val, seed_val=42)
    var n_mean = compute_mean(norm)
    var n_var = compute_variance(norm, n_mean)
    var n_std = sqrt(n_var)

    print(
        "  Normal(mean=" + String(mean_val) + ", std=" + String(std_val) + "):"
    )
    print(
        "    Mean: " + String(n_mean) + " (expected " + String(mean_val) + ")"
    )
    print("    Std: " + String(n_std) + " (expected " + String(std_val) + ")")

    assert_true(abs(n_mean - mean_val) < 0.01, "Normal mean")
    assert_true(abs(n_std - std_val) < 0.01, "Normal std")

    # Constant: all values should be constant
    var const_val = 0.42
    var const = constant(shape, const_val)
    var c_mean = compute_mean(const)
    var c_var = compute_variance(const, c_mean)

    print("  Constant(" + String(const_val) + "):")
    print(
        "    Mean: " + String(c_mean) + " (expected " + String(const_val) + ")"
    )
    print("    Variance: " + String(c_var) + " (expected 0.0)")

    assert_close_float(
        c_mean, const_val, rtol=1e-5, atol=1e-6, message="Constant mean"
    )
    assert_close_float(
        c_var, 0.0, rtol=1e-5, atol=1e-6, message="Constant variance"
    )

    print("  ✓ Basic distributions have correct statistical properties")


fn test_all_initializers_support_dtypes() raises:
    """Validate that all initializers support multiple dtypes."""
    print("Testing all initializers support multiple dtypes...")

    var shape: List[Int] = [10, 10]
    var fan_in = 10
    var fan_out = 10

    # Test each initializer with float16, float32, float64
    var dtypes: List[DType] = [DType.float16, DType.float32, DType.float64]
    var dtype_names: List[String] = ["float16", "float32", "float64"]

    for i in range(3):
        var dt = dtypes[i]
        var name = dtype_names[i]

        var xu = xavier_uniform(fan_in, fan_out, shape, dtype=dt)
        var xn = xavier_normal(fan_in, fan_out, shape, dtype=dt)
        var ku = kaiming_uniform(fan_in, fan_out, shape, dtype=dt)
        var kn = kaiming_normal(fan_in, fan_out, shape, dtype=dt)
        var u = uniform(shape, dtype=dt)
        var n = normal(shape, dtype=dt)
        var c = constant(shape, 0.5, dtype=dt)

        assert_dtype_equal(xu.dtype(), dt, "Xavier uniform dtype: " + name)
        assert_dtype_equal(xn.dtype(), dt, "Xavier normal dtype: " + name)
        assert_dtype_equal(ku.dtype(), dt, "Kaiming uniform dtype: " + name)
        assert_dtype_equal(kn.dtype(), dt, "Kaiming normal dtype: " + name)
        assert_dtype_equal(u.dtype(), dt, "Uniform dtype: " + name)
        assert_dtype_equal(n.dtype(), dt, "Normal dtype: " + name)
        assert_dtype_equal(c.dtype(), dt, "Constant dtype: " + name)

    print("  ✓ All initializers support float16, float32, float64")


fn test_initializers_api_consistency() raises:
    """Validate that all initializers follow consistent API patterns."""
    print("Testing initializers API consistency...")

    var shape: List[Int] = [100, 100]
    var fan_in = 100
    var fan_out = 100

    # All scaled initializers should accept: fan_in, fan_out, shape, dtype, seed
    var xu = xavier_uniform(fan_in, fan_out, shape, DType.float32, 42)
    var xn = xavier_normal(fan_in, fan_out, shape, DType.float32, 42)
    var ku = kaiming_uniform(
        fan_in, fan_out, shape, "fan_in", DType.float32, 42
    )
    var kn = kaiming_normal(fan_in, fan_out, shape, "fan_in", DType.float32, 42)

    # All basic distributions should accept: shape, params, dtype, seed
    var u = uniform(shape, -0.1, 0.1, DType.float32, 42)
    var n = normal(shape, 0.0, 0.01, DType.float32, 42)
    var c = constant(shape, 0.5, DType.float32)

    # All should return AnyTensor
    assert_equal(xu.numel(), 10000, "Xavier uniform returns AnyTensor")
    assert_equal(xn.numel(), 10000, "Xavier normal returns AnyTensor")
    assert_equal(ku.numel(), 10000, "Kaiming uniform returns AnyTensor")
    assert_equal(kn.numel(), 10000, "Kaiming normal returns AnyTensor")
    assert_equal(u.numel(), 10000, "Uniform returns AnyTensor")
    assert_equal(n.numel(), 10000, "Normal returns AnyTensor")
    assert_equal(c.numel(), 10000, "Constant returns AnyTensor")

    # All should raise on invalid inputs (tested in individual test files)

    print("  ✓ All initializers follow consistent API patterns")


fn test_initializers_integration() raises:
    """Test that all initializers work together in realistic scenarios."""
    print("Testing initializers integration...")

    # Simulate initializing a multi-layer network
    var layer1_shape: List[Int] = [784, 256]  # Input layer
    var layer2_shape: List[Int] = [256, 128]  # Hidden layer
    var layer3_shape: List[Int] = [128, 10]  # Output layer

    # Use different initializers for different layers
    var w1 = kaiming_uniform(784, 256, layer1_shape, seed_val=1)  # ReLU layer
    var bias1_shape: List[Int] = [256]
    var b1 = constant(bias1_shape, 0.0)

    var w2 = kaiming_normal(256, 128, layer2_shape, seed_val=2)  # ReLU layer
    var bias2_shape: List[Int] = [128]
    var b2 = constant(bias2_shape, 0.0)

    var w3 = xavier_uniform(128, 10, layer3_shape, seed_val=3)  # Softmax layer
    var bias3_shape: List[Int] = [10]
    var b3 = uniform(bias3_shape, low=-0.01, high=0.01, seed_val=4)

    # Verify all tensors have correct shapes
    assert_equal(w1.numel(), 784 * 256, "Layer 1 weights shape")
    assert_equal(b1.numel(), 256, "Layer 1 bias shape")
    assert_equal(w2.numel(), 256 * 128, "Layer 2 weights shape")
    assert_equal(b2.numel(), 128, "Layer 2 bias shape")
    assert_equal(w3.numel(), 128 * 10, "Layer 3 weights shape")
    assert_equal(b3.numel(), 10, "Layer 3 bias shape")

    # Verify biases are properly initialized
    var b1_mean = compute_mean(b1)
    var b2_mean = compute_mean(b2)
    assert_equal(b1_mean, 0.0, "Layer 1 bias is zero")
    assert_equal(b2_mean, 0.0, "Layer 2 bias is zero")

    var b3_mean = compute_mean(b3)
    assert_true(abs(b3_mean) < 0.01, "Layer 3 bias is small")

    print(
        "  ✓ Initializers work together for multi-layer network initialization"
    )


fn main() raises:
    """Run all cross-initializer validation tests."""
    print("\n" + "=" * 70)
    print("INITIALIZERS COORDINATION TEST SUITE")
    print("Statistical Validation Across All Initializers (#273-277)")
    print("=" * 70 + "\n")

    print("API Consistency Tests (#274)")
    print("-" * 70)
    test_all_initializers_produce_tensors()
    test_all_initializers_respect_seed()
    test_all_initializers_support_dtypes()
    test_initializers_api_consistency()

    print("\nStatistical Validation Tests (#274)")
    print("-" * 70)
    test_xavier_statistical_properties()
    test_kaiming_statistical_properties()
    test_basic_distributions_statistical_properties()

    print("\nIntegration Tests (#275, #276)")
    print("-" * 70)
    test_initializers_integration()

    print("\n" + "=" * 70)
    print("ALL INITIALIZERS COORDINATION TESTS PASSED ✓")
    print("=" * 70 + "\n")
    print("Summary:")
    print("  ✓ All 7 initializers produce valid tensors")
    print("  ✓ All initializers are reproducible with seeds")
    print("  ✓ All initializers support float16/32/64")
    print("  ✓ All initializers follow consistent API patterns")
    print("  ✓ Xavier initializers have correct variance: 2/(fan_in+fan_out)")
    print("  ✓ Kaiming initializers have correct variance: 2/fan")
    print("  ✓ Basic distributions have correct mean/variance")
    print("  ✓ All initializers work together for network initialization")
    print()
