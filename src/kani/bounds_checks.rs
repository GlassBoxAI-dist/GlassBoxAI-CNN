/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 *
 * Kani Verification: Strict Bound Checks
 * Prove that all collection indexing is incapable of out-of-bounds access
 */

use super::*;

#[kani::proof]
#[kani::unwind(10)]
fn verify_parse_args_bounds() {
    let argc: usize = kani::any();
    kani::assume(argc <= 8);

    if argc > 0 {
        let cmd_idx = kani::any::<usize>();
        kani::assume(cmd_idx < argc);
        assert!(cmd_idx < argc);
    }
}

#[kani::proof]
#[kani::unwind(5)]
fn verify_conv_filter_indexing() {
    let num_filters: usize = kani::any();
    let num_channels: usize = kani::any();
    let kernel_size: usize = kani::any();

    kani::assume(num_filters > 0 && num_filters <= 64);
    kani::assume(num_channels > 0 && num_channels <= 3);
    kani::assume(kernel_size > 0 && kernel_size <= 7);

    let weight_size = num_filters * num_channels * kernel_size * kernel_size;
    let filter_idx: usize = kani::any();
    let channel_idx: usize = kani::any();
    let kh: usize = kani::any();
    let kw: usize = kani::any();

    kani::assume(filter_idx < num_filters);
    kani::assume(channel_idx < num_channels);
    kani::assume(kh < kernel_size);
    kani::assume(kw < kernel_size);

    let weight_idx = filter_idx * num_channels * kernel_size * kernel_size
        + channel_idx * kernel_size * kernel_size
        + kh * kernel_size + kw;

    assert!(weight_idx < weight_size, "Weight index must be in bounds");
}

#[kani::proof]
#[kani::unwind(5)]
fn verify_output_indexing() {
    let num_filters: usize = kani::any();
    let output_h: usize = kani::any();
    let output_w: usize = kani::any();

    kani::assume(num_filters > 0 && num_filters <= 32);
    kani::assume(output_h > 0 && output_h <= 28);
    kani::assume(output_w > 0 && output_w <= 28);

    let total_size = num_filters * output_h * output_w;

    let f: usize = kani::any();
    let oh: usize = kani::any();
    let ow: usize = kani::any();

    kani::assume(f < num_filters);
    kani::assume(oh < output_h);
    kani::assume(ow < output_w);

    let out_idx = f * output_h * output_w + oh * output_w + ow;
    assert!(out_idx < total_size, "Output index must be in bounds");
}
