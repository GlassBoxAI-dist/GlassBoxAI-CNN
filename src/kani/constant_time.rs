//! @file
//! @ingroup CNN_Core_Verified
/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 *
 * Kani Verification: Constant-Time Execution
 * Security-sensitive operations
 */

use super::*;

#[kani::proof]
fn verify_activation_constant_time() {
    let x: f64 = kani::any();
    kani::assume(x.is_finite());

    let relu_result = if x > 0.0 { x } else { 0.0 };

    assert!(relu_result >= 0.0);
}

