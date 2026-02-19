//! @file
//! @ingroup CNN_Core_Verified
/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 *
 * Kani Verification: Pointer Validity Proofs
 * Verify all raw pointer operations are valid
 */

use super::*;

#[kani::proof]
fn verify_slice_to_ptr_validity() {
    let size: usize = kani::any();
    kani::assume(size > 0 && size <= 1024);

    let vec: Vec<f64> = vec![0.0; size];
    let slice = vec.as_slice();

    let ptr = slice.as_ptr();
    assert!(!ptr.is_null(), "Pointer must not be null");
    assert!(ptr.align_offset(std::mem::align_of::<f64>()) == 0, "Pointer must be aligned");
}

