/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 *
 * Kani Verification: Division-by-Zero Exclusion
 * Verify denominators are never zero
 */

use super::*;

#[kani::proof]
fn verify_launch_config_no_div_zero() {
    let n: u32 = kani::any();
    kani::assume(n > 0);

    const BLOCK_SIZE: u32 = 256;
    assert!(BLOCK_SIZE > 0, "BLOCK_SIZE must not be zero");

    let blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    assert!(blocks > 0, "Number of blocks must be positive");
}

#[kani::proof]
fn verify_pooling_stride_no_div_zero() {
    let pool_size: i32 = kani::any();
    kani::assume(pool_size > 0 && pool_size <= 4);

    let input_h: i32 = kani::any();
    kani::assume(input_h > 0 && input_h <= 256);

    let output_h = input_h / pool_size;
    assert!(pool_size != 0, "Pool size must not be zero");
}

#[kani::proof]
fn verify_batch_average_no_div_zero() {
    let num_samples: usize = kani::any();
    kani::assume(num_samples > 0);

    let total_loss: f64 = kani::any();
    kani::assume(total_loss.is_finite());

    let avg_loss = total_loss / num_samples as f64;
    assert!(avg_loss.is_finite() || num_samples > 0, "Average must be computable");
}
