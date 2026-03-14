"""Example: Mixed Precision Training with FP16.

Demonstrates how to use mixed precision training for faster training
with minimal accuracy loss. This example shows:

1. Creating FP16 model parameters
2. Using GradientScaler for loss/gradient scaling
3. Master weights in FP32 for optimizer
4. Gradient clipping for numerical stability
5. Handling NaN/Inf detection

Performance Benefits:
- 2-3x faster training on modern GPUs
- 50% reduction in memory usage
- Minimal accuracy loss (< 0.1% in most cases)

Usage:
    mojo examples/mixed_precision_training.mojo
"""

from shared.core import ExTensor, full
from shared.training.mixed_precision import (
    GradientScaler,
    convert_to_fp32_master,
    update_model_from_master,
    check_gradients_finite,
    clip_gradients_by_norm,
)
from shared.training.trainer_interface import TrainerConfig


fn simulate_forward_pass(params: ExTensor, input: ExTensor) raises -> ExTensor:
    """Simulate a simple forward pass: output = params * input."""
    return params * input


fn simulate_backward_pass(
    output: ExTensor, target: ExTensor
) raises -> ExTensor:
    """Simulate backward pass: compute gradients."""
    # In real training, this would compute actual gradients
    # For demo, just return the error
    return output - target


fn simple_optimizer_step(
    mut master_params: ExTensor, gradients: ExTensor, learning_rate: Float64
) raises:
    """Simple SGD update: params = params - lr * grads."""
    var lr_tensor = full(
        master_params.shape(), learning_rate, master_params.dtype()
    )
    var update = gradients * lr_tensor
    master_params = master_params - update


fn main() raises:
    print("\n" + "=" * 70)
    print("MIXED PRECISION TRAINING EXAMPLE")
    print("=" * 70)
    print()

    # ========================================================================
    # Configuration
    # ========================================================================
    print("Configuration:")
    print("-" * 70)

    var use_fp16 = True
    var model_dtype = DType.float16 if use_fp16 else DType.float32
    var learning_rate = 0.01
    var num_steps = 10
    var gradient_clip_norm = 1.0

    print("  Model Precision: " + ("FP16" if use_fp16 else "FP32"))
    print("  Learning Rate: " + String(learning_rate))
    print("  Training Steps: " + String(num_steps))
    print("  Gradient Clip Norm: " + String(gradient_clip_norm))
    print()

    # ========================================================================
    # Initialize Model Parameters
    # ========================================================================
    print("Initializing model parameters...")
    print("-" * 70)

    var param_shape = List[Int]()

    # Model parameters in FP16
    var model_params = full(param_shape, 1.0, model_dtype)
    print("  Model params shape: [100]")
    print(
        "  Model params dtype: "
        + ("float16" if model_dtype == DType.float16 else "float32")
    )

    # Master weights in FP32 (for optimizer)
    var master_params = convert_to_fp32_master(model_params)
    print("  Master params dtype: float32")
    print("  Master params are used for optimizer updates")
    print()

    # ========================================================================
    # Initialize Gradient Scaler (for FP16 training)
    # ========================================================================
    var scaler = GradientScaler(
        initial_scale=65536.0,  # Start with 2^16
        growth_factor=2.0,  # Double scale on success
        backoff_factor=0.5,  # Halve scale on overflow
        growth_interval=2000,  # Grow every 2000 successful steps
    )

    if use_fp16:
        print("Gradient Scaler initialized:")
        print("-" * 70)
        print("  Initial scale: " + String(scaler.get_scale()))
        print("  Growth factor: 2.0x")
        print("  Backoff factor: 0.5x")
        print("  Growth interval: 2000 steps")
        print()

    # ========================================================================
    # Training Loop
    # ========================================================================
    print("Starting training...")
    print("-" * 70)
    print()

    for step in range(num_steps):
        print("Step " + String(step + 1) + "/" + String(num_steps))

        # Create dummy input and target
        var input = full(param_shape, 0.5, model_dtype)
        var target = full(param_shape, 0.6, model_dtype)

        # ----------------------------------------------------------------
        # Forward Pass
        # ----------------------------------------------------------------
        var output = simulate_forward_pass(model_params, input)

        # Compute loss (simple MSE)
        var error = output - target
        var loss = error * error

        var loss_val = loss.item()
        print("  Loss: " + String(loss_val))

        # ----------------------------------------------------------------
        # Scale Loss (FP16 only)
        # ----------------------------------------------------------------
        var scaled_loss = scaler.scale_loss(loss) if use_fp16 else loss
        if use_fp16:
            var scaled_val = scaled_loss.item()
            print(
                "  Scaled Loss: "
                + String(scaled_val)
                + " (scale: "
                + String(scaler.get_scale())
                + ")"
            )

        # ----------------------------------------------------------------
        # Backward Pass (compute gradients)
        # ----------------------------------------------------------------
        var gradients = simulate_backward_pass(output, target)

        # ----------------------------------------------------------------
        # Unscale Gradients (FP16 only)
        # ----------------------------------------------------------------
        if use_fp16:
            gradients = scaler.unscale_gradients(gradients)

        # ----------------------------------------------------------------
        # Check for NaN/Inf
        # ----------------------------------------------------------------
        if not check_gradients_finite(gradients):
            print("  ⚠ NaN/Inf detected! Skipping step and reducing scale...")
            if use_fp16:
                scaler.backoff()
                print("  New scale: " + String(scaler.get_scale()))
            continue

        # ----------------------------------------------------------------
        # Clip Gradients (optional, for stability)
        # ----------------------------------------------------------------
        if gradient_clip_norm > 0.0:
            gradients = clip_gradients_by_norm(
                gradients, Float32(gradient_clip_norm)
            )

        # ----------------------------------------------------------------
        # Convert gradients to FP32 for optimizer
        # ----------------------------------------------------------------
        var fp32_gradients = convert_to_fp32_master(gradients)

        # ----------------------------------------------------------------
        # Optimizer Step (update master weights in FP32)
        # ----------------------------------------------------------------
        simple_optimizer_step(master_params, fp32_gradients, learning_rate)

        # ----------------------------------------------------------------
        # Copy master weights back to model parameters
        # ----------------------------------------------------------------
        update_model_from_master(model_params, master_params)

        # ----------------------------------------------------------------
        # Update Scaler (FP16 only)
        # ----------------------------------------------------------------
        if use_fp16:
            scaler.step()

        print("  ✓ Step completed")
        print()

    # ========================================================================
    # Training Summary
    # ========================================================================
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print()

    print("Final Statistics:")
    print("-" * 70)
    print("  Total steps: " + String(num_steps))

    if use_fp16:
        print("  Final gradient scale: " + String(scaler.get_scale()))
        print("  Total scaler steps: " + String(scaler.get_num_steps()))

    var final_param_val = model_params.item()
    print("  Final parameter value: " + String(final_param_val))
    print()

    print("Mixed Precision Training Benefits:")
    print("-" * 70)
    print("  ✓ 2-3x faster training on modern hardware")
    print("  ✓ 50% reduction in memory usage")
    print("  ✓ Automatic gradient scaling prevents underflow")
    print("  ✓ Master weights in FP32 maintain precision")
    print("  ✓ Dynamic scale adjustment handles overflow")
    print()

    print("=" * 70)
    print("Example completed successfully!")
    print("=" * 70)
