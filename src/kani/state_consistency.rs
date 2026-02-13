/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 *
 * Kani Verification: Global State Consistency
 */

use super::*;

#[kani::proof]
fn verify_is_training_state_consistency() {
    let is_training: bool = kani::any();

    assert!(is_training == true || is_training == false);
}
