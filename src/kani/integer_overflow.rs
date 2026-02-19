//! @file
//! @ingroup CNN_Core_Verified
/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 *
 * Kani Verification: Integer Overflow Prevention
 * Prove arithmetic operations are safe from overflow
 */

use super::*;

#[kani::proof]
fn verify_weight_size_no_overflow() {
    let num_filters: i32 = kani::any();
    let input_channels: i32 = kani::any();
    let kernel_size: i32 = kani::any();

    kani::assume(num_filters > 0 && num_filters <= 128);
    kani::assume(input_channels > 0 && input_channels <= 64);
    kani::assume(kernel_size > 0 && kernel_size <= 11);

    let step1 = num_filters.checked_mul(input_channels);
    kani::assume(step1.is_some());

    let step2 = step1.unwrap().checked_mul(kernel_size);
    kani::assume(step2.is_some());

    let weight_size = step2.unwrap().checked_mul(kernel_size);
    assert!(weight_size.is_some(), "Weight size calculation must not overflow");
}

#[kani::proof]
fn verify_output_dimension_no_overflow() {
    let input_dim: i32 = kani::any();
    let padding: i32 = kani::any();
    let kernel_size: i32 = kani::any();
    let stride: i32 = kani::any();

    kani::assume(input_dim > 0 && input_dim <= 256);
    kani::assume(padding >= 0 && padding <= 10);
    kani::assume(kernel_size > 0 && kernel_size <= 11);
    kani::assume(stride > 0 && stride <= 4);

    let step1 = padding.checked_mul(2);
    kani::assume(step1.is_some());

    let step2 = input_dim.checked_add(step1.unwrap());
    kani::assume(step2.is_some());

    let step3 = step2.unwrap().checked_sub(kernel_size);
    kani::assume(step3.is_some() && step3.unwrap() >= 0);

    let output_dim = step3.unwrap() / stride + 1;
    assert!(output_dim > 0, "Output dimension must be positive");
}

#[kani::proof]
fn verify_adam_timestep_no_overflow() {
    let adam_t: i32 = kani::any();
    kani::assume(adam_t >= 0 && adam_t < i32::MAX);

    let new_t = adam_t.checked_add(1);
    assert!(new_t.is_some(), "Adam timestep increment must not overflow");
}

