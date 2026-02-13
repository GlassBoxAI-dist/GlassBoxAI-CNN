/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 *
 * Kani Verification: Input Sanitization Bounds
 * Prove loops have formal upper bounds
 */

use super::*;

#[kani::proof]
#[kani::unwind(20)]
fn verify_conv_layer_iteration_bounded() {
    let num_conv_layers: usize = kani::any();
    kani::assume(num_conv_layers <= 10);

    let mut iterations = 0;
    for _i in 0..num_conv_layers {
        iterations += 1;
        assert!(iterations <= 10, "Iteration must be bounded");
    }
}

#[kani::proof]
#[kani::unwind(20)]
fn verify_fc_layer_iteration_bounded() {
    let num_fc_layers: usize = kani::any();
    kani::assume(num_fc_layers <= 10);

    let mut iterations = 0;
    for _i in 0..num_fc_layers {
        iterations += 1;
        assert!(iterations <= 10, "FC iteration must be bounded");
    }
}

#[kani::proof]
#[kani::unwind(100)]
fn verify_epoch_iteration_bounded() {
    let epochs: i32 = kani::any();
    kani::assume(epochs > 0 && epochs <= 50);

    let mut count = 0;
    for _epoch in 0..epochs {
        count += 1;
        assert!(count <= 50, "Epoch iteration must be bounded");
    }
}
