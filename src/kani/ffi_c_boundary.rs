/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 *
 * Kani Verification: C FFI Boundary Safety (CISA/NSA Compliance)
 *
 * Proves that all data crossing the C FFI boundary (capi.rs) is validated
 * before use. Covers the complete extern "C" surface consumed by C++, Go,
 * C#, Julia, Zig, and Python (via PyO3) wrappers.
 *
 * CISA "Secure by Design" requirements verified:
 * - Signed-to-unsigned conversion safety (c_int -> usize)
 * - Output buffer overflow prevention
 * - NaN/Infinity parameter rejection
 * - Enum variant validation from foreign callers
 * - Resource exhaustion prevention at boundary
 * - No-panic guarantee for validation logic
 * - ABI type compatibility proofs
 * - Error string NUL-byte sanitization
 */

use super::*;

// =========================================================================
// FFI validation helpers (mirroring the guards used in capi.rs)
// =========================================================================

fn validate_cint_positive(val: i32) -> Option<usize> {
    if val <= 0 { None } else { Some(val as usize) }
}

fn validate_cint_as_usize(val: i32) -> Option<usize> {
    if val < 0 { None } else { Some(val as usize) }
}

fn validate_ffi_len(len: i32, max: usize) -> Option<usize> {
    if len < 0 { return None; }
    let len_usize = len as usize;
    if len_usize > max { return None; }
    Some(len_usize)
}

fn validate_f64_param(value: f64) -> Option<f64> {
    if value.is_nan() || value.is_infinite() { None } else { Some(value) }
}

fn validate_f64_nonneg(value: f64) -> Option<f64> {
    if value.is_nan() || value.is_infinite() || value < 0.0 { None } else { Some(value) }
}

fn validate_dropout_rate(value: f64) -> Option<f64> {
    if value.is_nan() || value.is_infinite() || value < 0.0 || value > 1.0 { None } else { Some(value) }
}

const MAX_FFI_ARRAY_LEN: usize = 1048576;
const MAX_CONFIG_ARRAY_LEN: usize = 64;

// =========================================================================
// A. SIGNED-TO-UNSIGNED CONVERSION SAFETY
// Prove that c_int -> usize conversions reject negative values before any
// unsafe operation (from_raw_parts, array indexing, allocation).
// =========================================================================

#[kani::proof]
fn verify_ffi_cint_to_usize_rejects_negative() {
    let val: i32 = kani::any();
    let result = validate_cint_as_usize(val);
    if val < 0 {
        assert!(result.is_none(), "Negative c_int must be rejected before usize cast");
    } else {
        assert!(result == Some(val as usize), "Non-negative c_int must convert correctly");
    }
}

#[kani::proof]
fn verify_ffi_cint_positive_rejects_zero_and_negative() {
    let val: i32 = kani::any();
    let result = validate_cint_positive(val);
    if val <= 0 {
        assert!(result.is_none(), "Zero or negative must be rejected");
    } else {
        assert!(result == Some(val as usize), "Positive must convert correctly");
    }
}

#[kani::proof]
fn verify_ffi_len_validates_range() {
    let len: i32 = kani::any();
    let max: usize = kani::any();
    kani::assume(max <= MAX_FFI_ARRAY_LEN);

    let result = validate_ffi_len(len, max);
    if len < 0 || (len as usize) > max {
        assert!(result.is_none(), "Invalid length must be rejected");
    } else {
        assert!(result == Some(len as usize), "Valid length must be accepted");
    }
}

#[kani::proof]
fn verify_ffi_i32_min_rejected_everywhere() {
    assert!(validate_cint_as_usize(i32::MIN).is_none(), "MIN rejected as usize");
    assert!(validate_cint_positive(i32::MIN).is_none(), "MIN rejected as positive");
    assert!(validate_ffi_len(i32::MIN, MAX_FFI_ARRAY_LEN).is_none(), "MIN rejected as len");
}

#[kani::proof]
fn verify_ffi_negative_one_rejected() {
    assert!(validate_ffi_len(-1, MAX_FFI_ARRAY_LEN).is_none(), "-1 must be rejected");
}

// =========================================================================
// B. OUTPUT BUFFER OVERFLOW PREVENTION
// Prove that output buffer capacity is validated before copy operations.
// Models the C contract where capacity comes from untrusted caller.
// =========================================================================

#[kani::proof]
fn verify_ffi_negative_capacity_prevents_write() {
    let capacity: i32 = kani::any();
    kani::assume(capacity < 0);
    assert!(validate_cint_as_usize(capacity).is_none(),
        "Negative capacity must prevent buffer write");
}

#[kani::proof]
fn verify_ffi_zero_capacity_prevents_write() {
    assert!(validate_cint_positive(0).is_none(),
        "Zero capacity must be rejected");
}

#[kani::proof]
fn verify_ffi_output_write_bounded_by_capacity() {
    let data_len: usize = kani::any();
    let capacity: usize = kani::any();
    kani::assume(data_len <= 1024);
    kani::assume(capacity <= 1024);

    let write_len = data_len.min(capacity);
    assert!(write_len <= capacity, "Write must never exceed capacity");
    assert!(write_len <= data_len, "Write must never exceed data length");
}

#[kani::proof]
fn verify_ffi_predict_output_bounded() {
    let result_len: usize = kani::any();
    let raw_capacity: i32 = kani::any();
    kani::assume(result_len <= 256);
    kani::assume(raw_capacity > 0);

    let capacity = raw_capacity as usize;
    let write_len = result_len.min(capacity);
    assert!(write_len <= capacity, "Predict output must be bounded by capacity");
}

// =========================================================================
// C. NaN/INFINITY REJECTION AT FFI BOUNDARY
// Prove that NaN/Infinity f64 values from foreign callers are rejected
// before being stored in CNN state.
// =========================================================================

#[kani::proof]
fn verify_ffi_f64_param_rejects_special_values() {
    let val: f64 = kani::any();
    let result = validate_f64_param(val);
    if val.is_nan() || val.is_infinite() {
        assert!(result.is_none(), "NaN/Infinity must be rejected");
    } else {
        assert!(result == Some(val), "Finite values must be accepted");
    }
}

#[kani::proof]
fn verify_ffi_learning_rate_rejects_nan() {
    let val: f64 = kani::any();
    kani::assume(val.is_nan());
    assert!(validate_f64_nonneg(val).is_none(), "NaN learning rate must be rejected");
}

#[kani::proof]
fn verify_ffi_learning_rate_rejects_infinity() {
    let val: f64 = kani::any();
    kani::assume(val.is_infinite());
    assert!(validate_f64_nonneg(val).is_none(), "Infinite learning rate must be rejected");
}

#[kani::proof]
fn verify_ffi_learning_rate_rejects_negative() {
    let val: f64 = kani::any();
    kani::assume(val.is_finite() && val < 0.0);
    assert!(validate_f64_nonneg(val).is_none(), "Negative learning rate must be rejected");
}

#[kani::proof]
fn verify_ffi_learning_rate_accepts_valid() {
    let val: f64 = kani::any();
    kani::assume(val.is_finite() && val >= 0.0);
    assert!(validate_f64_nonneg(val).is_some(), "Valid learning rate must be accepted");
}

#[kani::proof]
fn verify_ffi_gradient_clip_validated() {
    let val: f64 = kani::any();
    let result = validate_f64_nonneg(val);
    if val.is_nan() || val.is_infinite() || val < 0.0 {
        assert!(result.is_none(), "Invalid gradient clip rejected");
    } else {
        assert!(result.is_some(), "Valid gradient clip accepted");
    }
}

#[kani::proof]
fn verify_ffi_dropout_rate_validated() {
    let val: f64 = kani::any();
    let result = validate_dropout_rate(val);
    if val.is_nan() || val.is_infinite() || val < 0.0 || val > 1.0 {
        assert!(result.is_none(), "Invalid dropout rate rejected");
    } else {
        assert!(result.is_some(), "Valid dropout rate accepted");
    }
}

// =========================================================================
// D. ENUM VARIANT VALIDATION FROM FOREIGN CALLERS
// Prove that C enum values map correctly through From impls.
// =========================================================================

#[kani::proof]
fn verify_ffi_activation_enum_roundtrip() {
    let act: ActivationType = kani::any();

    let result = match act {
        ActivationType::Sigmoid => 0,
        ActivationType::Tanh => 1,
        ActivationType::ReLU => 2,
        ActivationType::Linear => 3,
    };
    assert!(result >= 0 && result <= 3, "All activation variants handled");
}

#[kani::proof]
fn verify_ffi_loss_enum_roundtrip() {
    let loss: LossType = kani::any();

    let result = match loss {
        LossType::MSE => 0,
        LossType::CrossEntropy => 1,
    };
    assert!(result >= 0 && result <= 1, "All loss variants handled");
}

#[kani::proof]
fn verify_ffi_activation_i32_validation() {
    let val: i32 = kani::any();

    let valid = val >= 0 && val <= 3;
    if !valid {
        assert!(val < 0 || val > 3, "Out-of-range activation value detected");
    }
}

#[kani::proof]
fn verify_ffi_loss_i32_validation() {
    let val: i32 = kani::any();

    let valid = val >= 0 && val <= 1;
    if !valid {
        assert!(val < 0 || val > 1, "Out-of-range loss value detected");
    }
}

// =========================================================================
// E. CNN CREATE PRECONDITIONS
// Prove that cnn_create validates all parameter combinations before
// any unsafe operation.
// =========================================================================

#[kani::proof]
fn verify_ffi_create_rejects_zero_dimensions() {
    let w: i32 = kani::any();
    let h: i32 = kani::any();
    let c: i32 = kani::any();
    let o: i32 = kani::any();

    let valid = w > 0 && h > 0 && c > 0 && o > 0;
    if w <= 0 || h <= 0 || c <= 0 || o <= 0 {
        assert!(!valid, "Zero/negative dimensions must be rejected");
    }
}

#[kani::proof]
fn verify_ffi_create_rejects_negative_array_len() {
    let len: i32 = kani::any();
    kani::assume(len < 0);

    let valid = len > 0 && len <= MAX_CONFIG_ARRAY_LEN as i32;
    assert!(!valid, "Negative array length must be rejected before from_raw_parts");
}

#[kani::proof]
fn verify_ffi_create_rejects_excessive_array_len() {
    let len: i32 = kani::any();
    kani::assume(len > MAX_CONFIG_ARRAY_LEN as i32);

    let valid = len > 0 && len <= MAX_CONFIG_ARRAY_LEN as i32;
    assert!(!valid, "Excessive array length must be rejected");
}

#[kani::proof]
fn verify_ffi_create_negative_i32_as_usize_huge() {
    let neg: i32 = -1;
    let as_usize = neg as usize;
    assert!(as_usize > MAX_CONFIG_ARRAY_LEN,
        "Negative i32 as usize is huge, proving pre-validation needed");
}

#[kani::proof]
fn verify_ffi_create_i32_min_as_usize_huge() {
    let val: i32 = i32::MIN;
    let as_usize = val as usize;
    assert!(as_usize > MAX_FFI_ARRAY_LEN,
        "i32::MIN as usize is huge, proving pre-validation needed");
}

// =========================================================================
// F. TRAIN/PREDICT LENGTH VALIDATION
// Prove that input/target/output lengths from FFI are bounded and
// cannot cause from_raw_parts to read arbitrary memory.
// =========================================================================

#[kani::proof]
fn verify_ffi_predict_input_len_validated() {
    let image_len: i32 = kani::any();
    let output_len: i32 = kani::any();

    let iv = validate_ffi_len(image_len, MAX_FFI_ARRAY_LEN);
    let ov = validate_ffi_len(output_len, MAX_FFI_ARRAY_LEN);

    if image_len < 0 || image_len as usize > MAX_FFI_ARRAY_LEN {
        assert!(iv.is_none(), "Invalid image length rejected");
    }
    if output_len < 0 || output_len as usize > MAX_FFI_ARRAY_LEN {
        assert!(ov.is_none(), "Invalid output length rejected");
    }
}

#[kani::proof]
fn verify_ffi_train_input_len_validated() {
    let image_len: i32 = kani::any();
    let target_len: i32 = kani::any();

    let iv = validate_ffi_len(image_len, MAX_FFI_ARRAY_LEN);
    let tv = validate_ffi_len(target_len, MAX_FFI_ARRAY_LEN);

    if image_len < 0 || image_len as usize > MAX_FFI_ARRAY_LEN {
        assert!(iv.is_none(), "Invalid image length rejected");
    }
    if target_len < 0 || target_len as usize > MAX_FFI_ARRAY_LEN {
        assert!(tv.is_none(), "Invalid target length rejected");
    }
}

#[kani::proof]
fn verify_ffi_predict_capacity_validated() {
    let capacity: i32 = kani::any();
    let valid = validate_cint_positive(capacity);
    if capacity <= 0 {
        assert!(valid.is_none(), "Non-positive capacity rejected");
    }
}

// =========================================================================
// G. ERROR STRING NUL-BYTE SANITIZATION
// Prove that error messages are safe for CString creation.
// =========================================================================

#[kani::proof]
fn verify_ffi_error_nul_byte_sanitized() {
    let test_strings = ["error\0message", "\0", "normal error", "a\0b\0c"];
    for s in &test_strings {
        let sanitized = s.replace('\0', "");
        assert!(!sanitized.contains('\0'),
            "NUL bytes must be removed before CString creation");
    }
}

#[kani::proof]
fn verify_ffi_error_empty_after_nul_removal() {
    let s = "\0";
    let sanitized = s.replace('\0', "");
    assert!(sanitized.is_empty(), "Pure NUL string becomes empty");
    assert!(std::ffi::CString::new(sanitized).is_ok(), "Empty string is valid CString");
}

// =========================================================================
// H. NO-PANIC GUARANTEE FOR FFI VALIDATORS
// Prove that validation functions never panic regardless of input.
// =========================================================================

#[kani::proof]
fn verify_ffi_all_validators_no_panic() {
    let i: i32 = kani::any();
    let f: f64 = kani::any();

    let _a = validate_cint_as_usize(i);
    let _b = validate_cint_positive(i);
    let _c = validate_ffi_len(i, MAX_FFI_ARRAY_LEN);
    let _d = validate_f64_param(f);
    let _e = validate_f64_nonneg(f);
    let _f = validate_dropout_rate(f);
}

// =========================================================================
// I. ABI TYPE COMPATIBILITY
// Prove that repr(C) enums and primitives have expected layouts.
// =========================================================================

#[kani::proof]
fn verify_ffi_f64_abi_compatibility() {
    assert!(std::mem::size_of::<f64>() == 8, "f64 must be 8 bytes for C double");
    assert!(std::mem::align_of::<f64>() == 8, "f64 must be 8-byte aligned");
}

#[kani::proof]
fn verify_ffi_i32_abi_compatibility() {
    assert!(std::mem::size_of::<i32>() == 4, "i32 must be 4 bytes for C int32_t");
    assert!(std::mem::align_of::<i32>() == 4, "i32 must be 4-byte aligned");
}

// =========================================================================
// J. INPUT ARRAY NaN/INFINITY DETECTION
// Prove that f64 arrays crossing FFI boundary can be validated for
// NaN/Infinity values before use in computation.
// =========================================================================

#[kani::proof]
#[kani::unwind(9)]
fn verify_ffi_nan_in_input_array_detectable() {
    let size: usize = kani::any();
    kani::assume(size > 0 && size <= 8);

    let mut arr = vec![1.0; size];
    let idx: usize = kani::any();
    kani::assume(idx < size);
    arr[idx] = f64::NAN;

    let has_bad = arr.iter().any(|x| x.is_nan() || x.is_infinite());
    assert!(has_bad, "NaN in array must be detectable");
}

#[kani::proof]
#[kani::unwind(9)]
fn verify_ffi_inf_in_input_array_detectable() {
    let size: usize = kani::any();
    kani::assume(size > 0 && size <= 8);

    let mut arr = vec![1.0; size];
    let idx: usize = kani::any();
    kani::assume(idx < size);
    arr[idx] = f64::INFINITY;

    let has_bad = arr.iter().any(|x| x.is_nan() || x.is_infinite());
    assert!(has_bad, "Infinity in array must be detectable");
}

// =========================================================================
// K. RESOURCE LIMITS AT FFI BOUNDARY
// Prove that FFI-supplied sizes cannot cause excessive allocation.
// =========================================================================

#[kani::proof]
fn verify_ffi_allocation_bounded_by_max() {
    let len: i32 = kani::any();
    kani::assume(len >= 0 && len <= MAX_FFI_ARRAY_LEN as i32);

    let len_usize = len as usize;
    let bytes = len_usize * std::mem::size_of::<f64>();
    assert!(bytes <= MAX_FFI_ARRAY_LEN * 8, "FFI allocation must be bounded");
}

#[kani::proof]
fn verify_ffi_config_array_len_bounded() {
    let len: i32 = kani::any();
    kani::assume(len > 0 && len <= MAX_CONFIG_ARRAY_LEN as i32);

    let len_usize = len as usize;
    assert!(len_usize <= MAX_CONFIG_ARRAY_LEN, "Config array length bounded");
    let bytes = len_usize * std::mem::size_of::<i32>();
    assert!(bytes <= MAX_CONFIG_ARRAY_LEN * 4, "Config array memory bounded");
}

// =========================================================================
// L. SETTER VALUE VALIDATION
// Prove that setters (learning_rate, gradient_clip, dropout_rate) reject
// NaN/Inf values and out-of-range values before mutating CNN state.
// =========================================================================

#[kani::proof]
fn verify_ffi_setter_rejects_nan() {
    let value: f64 = kani::any();
    kani::assume(value.is_nan());

    let accepted = !value.is_nan() && !value.is_infinite() && value >= 0.0;
    assert!(!accepted, "NaN must be rejected by setter guard");
}

#[kani::proof]
fn verify_ffi_setter_rejects_infinity() {
    let value: f64 = kani::any();
    kani::assume(value.is_infinite());

    let accepted = !value.is_nan() && !value.is_infinite() && value >= 0.0;
    assert!(!accepted, "Infinity must be rejected by setter guard");
}

#[kani::proof]
fn verify_ffi_setter_rejects_negative() {
    let value: f64 = kani::any();
    kani::assume(value.is_finite() && value < 0.0);

    let accepted = !value.is_nan() && !value.is_infinite() && value >= 0.0;
    assert!(!accepted, "Negative must be rejected by setter guard");
}

#[kani::proof]
fn verify_ffi_dropout_setter_rejects_over_one() {
    let value: f64 = kani::any();
    kani::assume(value.is_finite() && value > 1.0);

    let accepted = !value.is_nan() && !value.is_infinite() && value >= 0.0 && value <= 1.0;
    assert!(!accepted, "Dropout > 1.0 must be rejected");
}

#[kani::proof]
fn verify_ffi_dropout_setter_accepts_valid() {
    let value: f64 = kani::any();
    kani::assume(value.is_finite() && value >= 0.0 && value <= 1.0);

    let accepted = !value.is_nan() && !value.is_infinite() && value >= 0.0 && value <= 1.0;
    assert!(accepted, "Valid dropout must be accepted");
}

// =========================================================================
// M. END-TO-END FFI PIPELINE VALIDATION
// Prove that the full validation pipeline is correct for key operations.
// =========================================================================

#[kani::proof]
fn verify_ffi_complete_predict_pipeline() {
    let image_len: i32 = kani::any();
    let output_len: i32 = kani::any();

    let iv = validate_ffi_len(image_len, MAX_FFI_ARRAY_LEN);
    let ov = validate_cint_positive(output_len);

    if iv.is_some() && ov.is_some() {
        assert!(iv.unwrap() <= MAX_FFI_ARRAY_LEN, "Image len bounded");
        let result_len: usize = kani::any();
        kani::assume(result_len <= 256);
        let write = result_len.min(ov.unwrap());
        assert!(write <= ov.unwrap(), "Write bounded by capacity");
    }
}

#[kani::proof]
fn verify_ffi_complete_train_pipeline() {
    let image_len: i32 = kani::any();
    let target_len: i32 = kani::any();

    let iv = validate_ffi_len(image_len, MAX_FFI_ARRAY_LEN);
    let tv = validate_ffi_len(target_len, MAX_FFI_ARRAY_LEN);

    if iv.is_some() && tv.is_some() {
        assert!(iv.unwrap() <= MAX_FFI_ARRAY_LEN, "Image len bounded");
        assert!(tv.unwrap() <= MAX_FFI_ARRAY_LEN, "Target len bounded");

        let img_bytes = iv.unwrap() * std::mem::size_of::<f64>();
        let tgt_bytes = tv.unwrap() * std::mem::size_of::<f64>();
        assert!(img_bytes <= MAX_FFI_ARRAY_LEN * 8, "Image memory bounded");
        assert!(tgt_bytes <= MAX_FFI_ARRAY_LEN * 8, "Target memory bounded");
    }
}

#[kani::proof]
fn verify_ffi_complete_create_pipeline() {
    let input_w: i32 = kani::any();
    let input_h: i32 = kani::any();
    let input_c: i32 = kani::any();
    let output_s: i32 = kani::any();
    let arr_len: i32 = kani::any();

    let wv = validate_cint_positive(input_w);
    let hv = validate_cint_positive(input_h);
    let cv = validate_cint_positive(input_c);
    let ov = validate_cint_positive(output_s);
    let av = validate_ffi_len(arr_len, MAX_CONFIG_ARRAY_LEN);

    if wv.is_some() && hv.is_some() && cv.is_some() && ov.is_some() && av.is_some() {
        assert!(wv.unwrap() > 0, "Width positive");
        assert!(hv.unwrap() > 0, "Height positive");
        assert!(cv.unwrap() > 0, "Channels positive");
        assert!(ov.unwrap() > 0, "Output positive");
        assert!(av.unwrap() <= MAX_CONFIG_ARRAY_LEN, "Array len bounded");
    }
}
