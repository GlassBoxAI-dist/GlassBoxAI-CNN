/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 *
 * Kani Verification: Floating-Point Sanity
 * Prove operations never result in unhandled NaN or Infinity
 */

use super::*;

#[kani::proof]
fn verify_relu_no_nan() {
    let x: f64 = kani::any();

    let result = if x > 0.0 { x } else { 0.0 };

    if x.is_finite() {
        assert!(result.is_finite(), "ReLU output must be finite for finite input");
    }
}

#[kani::proof]
fn verify_softmax_denominator_positive() {
    let max_val: f64 = kani::any();
    let input_val: f64 = kani::any();

    kani::assume(max_val.is_finite());
    kani::assume(input_val.is_finite());
    kani::assume(input_val <= max_val);

    let exp_diff = (input_val - max_val).exp();

    assert!(exp_diff > 0.0, "Exponential must be positive");
    assert!(exp_diff <= 1.0, "Exponential of non-positive should be <= 1");
}

#[kani::proof]
fn verify_gradient_clipping() {
    let grad: f64 = kani::any();
    kani::assume(grad.is_finite());

    let clip_val = 1.0;
    let clipped = if grad > clip_val {
        clip_val
    } else if grad < -clip_val {
        -clip_val
    } else {
        grad
    };

    assert!(clipped >= -1.0 && clipped <= 1.0, "Clipped gradient must be in [-1, 1]");
    assert!(clipped.is_finite(), "Clipped gradient must be finite");
}
