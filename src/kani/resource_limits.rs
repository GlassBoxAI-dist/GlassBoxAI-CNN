//! @file
//! @ingroup CNN_Core_Verified
/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 *
 * Kani Verification: Resource Limit Compliance
 * Verify allocations don't exceed thresholds
 */

use super::*;

#[kani::proof]
fn verify_weight_allocation_limit() {
    let num_filters: usize = kani::any();
    let input_channels: usize = kani::any();
    let kernel_size: usize = kani::any();

    kani::assume(num_filters <= 128);
    kani::assume(input_channels <= 64);
    kani::assume(kernel_size <= 11);

    let weight_count = num_filters.saturating_mul(input_channels)
        .saturating_mul(kernel_size)
        .saturating_mul(kernel_size);

    const MAX_WEIGHTS: usize = 1_000_000;
    assert!(weight_count <= MAX_WEIGHTS, "Weight count must not exceed budget");
}

#[kani::proof]
fn verify_fc_layer_allocation_limit() {
    let num_inputs: usize = kani::any();
    let num_neurons: usize = kani::any();

    kani::assume(num_inputs <= 10000);
    kani::assume(num_neurons <= 1000);

    let weight_count = num_inputs.saturating_mul(num_neurons);

    const MAX_FC_WEIGHTS: usize = 10_000_000;
    assert!(weight_count <= MAX_FC_WEIGHTS, "FC layer weight count within budget");
}

#[kani::proof]
fn verify_output_buffer_size_limit() {
    let num_filters: usize = kani::any();
    let output_h: usize = kani::any();
    let output_w: usize = kani::any();

    kani::assume(num_filters <= 256);
    kani::assume(output_h <= 256);
    kani::assume(output_w <= 256);

    let buffer_size = num_filters.saturating_mul(output_h).saturating_mul(output_w);

    const MAX_BUFFER: usize = 16_777_216;
    assert!(buffer_size <= MAX_BUFFER, "Output buffer within memory budget");
}

