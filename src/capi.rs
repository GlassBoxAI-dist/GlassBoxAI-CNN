/*
 * MIT License
 *
 * Copyright (c) 2025 Matthew Abbott
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

//! C API bindings for the CUDA CNN library.
//!
//! This module provides a C-compatible FFI for use from C and C++ code.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_double, c_int};
use std::ptr;
use std::slice;

use crate::cnn::{
    ActivationType, ConvolutionalNeuralNetworkCUDA as RustCNN, LossType,
};

/// Opaque handle to a CNN instance.
pub struct CnnHandle {
    inner: RustCNN,
}

/// Error codes returned by C API functions.
#[repr(C)]
pub enum CnnError {
    /// Operation completed successfully.
    Success = 0,
    /// Null pointer was passed where a valid pointer was expected.
    NullPointer = 1,
    /// Invalid parameter value.
    InvalidParameter = 2,
    /// Failed to create CNN.
    CreationFailed = 3,
    /// Prediction failed.
    PredictionFailed = 4,
    /// Training step failed.
    TrainingFailed = 5,
    /// Failed to save model.
    SaveFailed = 6,
    /// Failed to load model.
    LoadFailed = 7,
    /// Failed to export model.
    ExportFailed = 8,
    /// Failed to import model.
    ImportFailed = 9,
    /// Buffer too small.
    BufferTooSmall = 10,
    /// Unknown error.
    Unknown = 255,
}

/// Activation function types.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CnnActivationType {
    Sigmoid = 0,
    Tanh = 1,
    ReLU = 2,
    Linear = 3,
}

impl From<CnnActivationType> for ActivationType {
    fn from(c: CnnActivationType) -> Self {
        match c {
            CnnActivationType::Sigmoid => ActivationType::Sigmoid,
            CnnActivationType::Tanh => ActivationType::Tanh,
            CnnActivationType::ReLU => ActivationType::ReLU,
            CnnActivationType::Linear => ActivationType::Linear,
        }
    }
}

impl From<ActivationType> for CnnActivationType {
    fn from(rust: ActivationType) -> Self {
        match rust {
            ActivationType::Sigmoid => CnnActivationType::Sigmoid,
            ActivationType::Tanh => CnnActivationType::Tanh,
            ActivationType::ReLU => CnnActivationType::ReLU,
            ActivationType::Linear => CnnActivationType::Linear,
        }
    }
}

/// Loss function types.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CnnLossType {
    MSE = 0,
    CrossEntropy = 1,
}

impl From<CnnLossType> for LossType {
    fn from(c: CnnLossType) -> Self {
        match c {
            CnnLossType::MSE => LossType::MSE,
            CnnLossType::CrossEntropy => LossType::CrossEntropy,
        }
    }
}

impl From<LossType> for CnnLossType {
    fn from(rust: LossType) -> Self {
        match rust {
            LossType::MSE => CnnLossType::MSE,
            LossType::CrossEntropy => CnnLossType::CrossEntropy,
        }
    }
}

/// Configuration for creating a CNN.
#[repr(C)]
pub struct CnnConfig {
    pub input_width: c_int,
    pub input_height: c_int,
    pub input_channels: c_int,
    pub conv_filters: *const c_int,
    pub conv_filters_len: c_int,
    pub kernel_sizes: *const c_int,
    pub kernel_sizes_len: c_int,
    pub pool_sizes: *const c_int,
    pub pool_sizes_len: c_int,
    pub fc_sizes: *const c_int,
    pub fc_sizes_len: c_int,
    pub output_size: c_int,
    pub hidden_activation: CnnActivationType,
    pub output_activation: CnnActivationType,
    pub loss_type: CnnLossType,
    pub learning_rate: c_double,
    pub gradient_clip: c_double,
}

// Thread-local storage for the last error message.
thread_local! {
    static LAST_ERROR: std::cell::RefCell<Option<CString>> = const { std::cell::RefCell::new(None) };
}

fn set_last_error(msg: &str) {
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = CString::new(msg).ok();
    });
}

// ============================================================================
// Lifecycle Functions
// ============================================================================

/// Creates a new CNN with the specified configuration.
///
/// # Safety
/// - `config` must point to a valid `CnnConfig` struct.
/// - All array pointers in `config` must be valid and have the specified lengths.
/// - `out_handle` must point to a valid location to store the handle.
///
/// # Returns
/// `CnnError::Success` on success, or an error code on failure.
#[no_mangle]
pub unsafe extern "C" fn cnn_create(
    config: *const CnnConfig,
    out_handle: *mut *mut CnnHandle,
) -> CnnError {
    if config.is_null() || out_handle.is_null() {
        set_last_error("Null pointer passed to cnn_create");
        return CnnError::NullPointer;
    }

    let config = &*config;

    if config.conv_filters.is_null()
        || config.kernel_sizes.is_null()
        || config.pool_sizes.is_null()
        || config.fc_sizes.is_null()
    {
        set_last_error("Null array pointer in config");
        return CnnError::NullPointer;
    }

    let conv_filters =
        slice::from_raw_parts(config.conv_filters, config.conv_filters_len as usize);
    let kernel_sizes =
        slice::from_raw_parts(config.kernel_sizes, config.kernel_sizes_len as usize);
    let pool_sizes = slice::from_raw_parts(config.pool_sizes, config.pool_sizes_len as usize);
    let fc_sizes = slice::from_raw_parts(config.fc_sizes, config.fc_sizes_len as usize);

    let hidden_act = ActivationType::from(config.hidden_activation);
    let output_act = ActivationType::from(config.output_activation);
    let loss = LossType::from(config.loss_type);

    match RustCNN::new(
        config.input_width,
        config.input_height,
        config.input_channels,
        conv_filters,
        kernel_sizes,
        pool_sizes,
        fc_sizes,
        config.output_size,
        hidden_act,
        output_act,
        loss,
        config.learning_rate,
        config.gradient_clip,
    ) {
        Ok(cnn) => {
            let handle = Box::new(CnnHandle { inner: cnn });
            *out_handle = Box::into_raw(handle);
            CnnError::Success
        }
        Err(e) => {
            set_last_error(&format!("Failed to create CNN: {}", e));
            CnnError::CreationFailed
        }
    }
}

/// Destroys a CNN instance and frees its resources.
///
/// # Safety
/// - `handle` must be a valid handle returned by `cnn_create`, `cnn_load_from_json`,
///   or `cnn_import_from_onnx`.
/// - After this call, `handle` is no longer valid and must not be used.
#[no_mangle]
pub unsafe extern "C" fn cnn_destroy(handle: *mut CnnHandle) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

// ============================================================================
// Inference and Training
// ============================================================================

/// Performs inference on an input image.
///
/// # Safety
/// - `handle` must be a valid CNN handle.
/// - `image_data` must point to `image_len` valid doubles.
/// - `output` must point to a buffer of at least `output_len` doubles.
/// - `output_len` must be at least as large as the CNN's output size.
///
/// # Returns
/// `CnnError::Success` on success, or an error code on failure.
#[no_mangle]
pub unsafe extern "C" fn cnn_predict(
    handle: *mut CnnHandle,
    image_data: *const c_double,
    image_len: c_int,
    output: *mut c_double,
    output_len: c_int,
) -> CnnError {
    if handle.is_null() || image_data.is_null() || output.is_null() {
        set_last_error("Null pointer passed to cnn_predict");
        return CnnError::NullPointer;
    }

    let handle = &mut *handle;
    let image = slice::from_raw_parts(image_data, image_len as usize);

    match handle.inner.predict(image) {
        Ok(result) => {
            if result.len() > output_len as usize {
                set_last_error("Output buffer too small");
                return CnnError::BufferTooSmall;
            }
            let out_slice = slice::from_raw_parts_mut(output, result.len());
            out_slice.copy_from_slice(&result);
            CnnError::Success
        }
        Err(e) => {
            set_last_error(&format!("Prediction failed: {}", e));
            CnnError::PredictionFailed
        }
    }
}

/// Performs a single training step.
///
/// # Safety
/// - `handle` must be a valid CNN handle.
/// - `image_data` must point to `image_len` valid doubles.
/// - `target` must point to `target_len` valid doubles.
/// - `out_loss` must point to a valid location to store the loss.
///
/// # Returns
/// `CnnError::Success` on success, or an error code on failure.
#[no_mangle]
pub unsafe extern "C" fn cnn_train_step(
    handle: *mut CnnHandle,
    image_data: *const c_double,
    image_len: c_int,
    target: *const c_double,
    target_len: c_int,
    out_loss: *mut c_double,
) -> CnnError {
    if handle.is_null() || image_data.is_null() || target.is_null() || out_loss.is_null() {
        set_last_error("Null pointer passed to cnn_train_step");
        return CnnError::NullPointer;
    }

    let handle = &mut *handle;
    let image = slice::from_raw_parts(image_data, image_len as usize);
    let target = slice::from_raw_parts(target, target_len as usize);

    match handle.inner.train_step(image, target) {
        Ok(loss) => {
            *out_loss = loss;
            CnnError::Success
        }
        Err(e) => {
            set_last_error(&format!("Training step failed: {}", e));
            CnnError::TrainingFailed
        }
    }
}

// ============================================================================
// Serialization
// ============================================================================

/// Saves the model to a JSON file.
///
/// # Safety
/// - `handle` must be a valid CNN handle.
/// - `filename` must be a valid null-terminated C string.
///
/// # Returns
/// `CnnError::Success` on success, or an error code on failure.
#[no_mangle]
pub unsafe extern "C" fn cnn_save_to_json(
    handle: *const CnnHandle,
    filename: *const c_char,
) -> CnnError {
    if handle.is_null() || filename.is_null() {
        set_last_error("Null pointer passed to cnn_save_to_json");
        return CnnError::NullPointer;
    }

    let handle = &*handle;
    let filename = match CStr::from_ptr(filename).to_str() {
        Ok(s) => s,
        Err(_) => {
            set_last_error("Invalid UTF-8 in filename");
            return CnnError::InvalidParameter;
        }
    };

    match handle.inner.save_to_json(filename) {
        Ok(()) => CnnError::Success,
        Err(e) => {
            set_last_error(&format!("Failed to save model: {}", e));
            CnnError::SaveFailed
        }
    }
}

/// Loads a model from a JSON file.
///
/// # Safety
/// - `filename` must be a valid null-terminated C string.
/// - `out_handle` must point to a valid location to store the handle.
///
/// # Returns
/// `CnnError::Success` on success, or an error code on failure.
#[no_mangle]
pub unsafe extern "C" fn cnn_load_from_json(
    filename: *const c_char,
    out_handle: *mut *mut CnnHandle,
) -> CnnError {
    if filename.is_null() || out_handle.is_null() {
        set_last_error("Null pointer passed to cnn_load_from_json");
        return CnnError::NullPointer;
    }

    let filename = match CStr::from_ptr(filename).to_str() {
        Ok(s) => s,
        Err(_) => {
            set_last_error("Invalid UTF-8 in filename");
            return CnnError::InvalidParameter;
        }
    };

    match RustCNN::load_from_json(filename) {
        Ok(cnn) => {
            let handle = Box::new(CnnHandle { inner: cnn });
            *out_handle = Box::into_raw(handle);
            CnnError::Success
        }
        Err(e) => {
            set_last_error(&format!("Failed to load model: {}", e));
            CnnError::LoadFailed
        }
    }
}

/// Exports the model to ONNX binary format.
///
/// # Safety
/// - `handle` must be a valid CNN handle.
/// - `filename` must be a valid null-terminated C string.
///
/// # Returns
/// `CnnError::Success` on success, or an error code on failure.
#[no_mangle]
pub unsafe extern "C" fn cnn_export_to_onnx(
    handle: *const CnnHandle,
    filename: *const c_char,
) -> CnnError {
    if handle.is_null() || filename.is_null() {
        set_last_error("Null pointer passed to cnn_export_to_onnx");
        return CnnError::NullPointer;
    }

    let handle = &*handle;
    let filename = match CStr::from_ptr(filename).to_str() {
        Ok(s) => s,
        Err(_) => {
            set_last_error("Invalid UTF-8 in filename");
            return CnnError::InvalidParameter;
        }
    };

    match handle.inner.export_to_onnx(filename) {
        Ok(()) => CnnError::Success,
        Err(e) => {
            set_last_error(&format!("Failed to export model: {}", e));
            CnnError::ExportFailed
        }
    }
}

/// Imports a model from ONNX binary format.
///
/// # Safety
/// - `filename` must be a valid null-terminated C string.
/// - `out_handle` must point to a valid location to store the handle.
///
/// # Returns
/// `CnnError::Success` on success, or an error code on failure.
#[no_mangle]
pub unsafe extern "C" fn cnn_import_from_onnx(
    filename: *const c_char,
    out_handle: *mut *mut CnnHandle,
) -> CnnError {
    if filename.is_null() || out_handle.is_null() {
        set_last_error("Null pointer passed to cnn_import_from_onnx");
        return CnnError::NullPointer;
    }

    let filename = match CStr::from_ptr(filename).to_str() {
        Ok(s) => s,
        Err(_) => {
            set_last_error("Invalid UTF-8 in filename");
            return CnnError::InvalidParameter;
        }
    };

    match RustCNN::import_from_onnx(filename) {
        Ok(cnn) => {
            let handle = Box::new(CnnHandle { inner: cnn });
            *out_handle = Box::into_raw(handle);
            CnnError::Success
        }
        Err(e) => {
            set_last_error(&format!("Failed to import model: {}", e));
            CnnError::ImportFailed
        }
    }
}

// ============================================================================
// Property Getters
// ============================================================================

/// Gets the input width.
#[no_mangle]
pub unsafe extern "C" fn cnn_get_input_width(handle: *const CnnHandle) -> c_int {
    if handle.is_null() {
        return -1;
    }
    (*handle).inner.get_input_width()
}

/// Gets the input height.
#[no_mangle]
pub unsafe extern "C" fn cnn_get_input_height(handle: *const CnnHandle) -> c_int {
    if handle.is_null() {
        return -1;
    }
    (*handle).inner.get_input_height()
}

/// Gets the number of input channels.
#[no_mangle]
pub unsafe extern "C" fn cnn_get_input_channels(handle: *const CnnHandle) -> c_int {
    if handle.is_null() {
        return -1;
    }
    (*handle).inner.get_input_channels()
}

/// Gets the output size.
#[no_mangle]
pub unsafe extern "C" fn cnn_get_output_size(handle: *const CnnHandle) -> c_int {
    if handle.is_null() {
        return -1;
    }
    (*handle).inner.get_output_size()
}

/// Gets the learning rate.
#[no_mangle]
pub unsafe extern "C" fn cnn_get_learning_rate(handle: *const CnnHandle) -> c_double {
    if handle.is_null() {
        return -1.0;
    }
    (*handle).inner.get_learning_rate()
}

/// Sets the learning rate.
#[no_mangle]
pub unsafe extern "C" fn cnn_set_learning_rate(handle: *mut CnnHandle, lr: c_double) {
    if !handle.is_null() {
        (*handle).inner.set_learning_rate(lr);
    }
}

/// Gets the gradient clipping threshold.
#[no_mangle]
pub unsafe extern "C" fn cnn_get_gradient_clip(handle: *const CnnHandle) -> c_double {
    if handle.is_null() {
        return -1.0;
    }
    (*handle).inner.get_gradient_clip()
}

/// Sets the gradient clipping threshold.
#[no_mangle]
pub unsafe extern "C" fn cnn_set_gradient_clip(handle: *mut CnnHandle, clip: c_double) {
    if !handle.is_null() {
        (*handle).inner.set_gradient_clip(clip);
    }
}

/// Sets the dropout rate.
#[no_mangle]
pub unsafe extern "C" fn cnn_set_dropout_rate(handle: *mut CnnHandle, rate: c_double) {
    if !handle.is_null() {
        (*handle).inner.set_dropout_rate(rate);
    }
}

/// Gets the hidden activation type.
#[no_mangle]
pub unsafe extern "C" fn cnn_get_hidden_activation(handle: *const CnnHandle) -> CnnActivationType {
    if handle.is_null() {
        return CnnActivationType::ReLU;
    }
    CnnActivationType::from((*handle).inner.get_hidden_activation())
}

/// Gets the output activation type.
#[no_mangle]
pub unsafe extern "C" fn cnn_get_output_activation(handle: *const CnnHandle) -> CnnActivationType {
    if handle.is_null() {
        return CnnActivationType::Linear;
    }
    CnnActivationType::from((*handle).inner.get_output_activation())
}

/// Gets the loss function type.
#[no_mangle]
pub unsafe extern "C" fn cnn_get_loss_function(handle: *const CnnHandle) -> CnnLossType {
    if handle.is_null() {
        return CnnLossType::CrossEntropy;
    }
    CnnLossType::from((*handle).inner.get_loss_function())
}

/// Gets whether batch normalization is enabled.
#[no_mangle]
pub unsafe extern "C" fn cnn_uses_batch_norm(handle: *const CnnHandle) -> c_int {
    if handle.is_null() {
        return 0;
    }
    if (*handle).inner.uses_batch_norm() {
        1
    } else {
        0
    }
}

/// Initializes batch normalization for all convolutional layers.
#[no_mangle]
pub unsafe extern "C" fn cnn_initialize_batch_norm(handle: *mut CnnHandle) {
    if !handle.is_null() {
        (*handle).inner.initialize_batch_norm();
    }
}

// ============================================================================
// Error Handling
// ============================================================================

/// Gets the last error message.
///
/// # Safety
/// The returned pointer is valid until the next call to any C API function
/// on the current thread. Do not free the returned pointer.
///
/// # Returns
/// A pointer to a null-terminated string, or NULL if no error occurred.
#[no_mangle]
pub unsafe extern "C" fn cnn_get_last_error() -> *const c_char {
    LAST_ERROR.with(|e| {
        match &*e.borrow() {
            Some(msg) => msg.as_ptr(),
            None => ptr::null(),
        }
    })
}

/// Clears the last error message.
#[no_mangle]
pub extern "C" fn cnn_clear_error() {
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = None;
    });
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Returns the library version as a null-terminated string.
///
/// # Safety
/// The returned pointer is valid for the lifetime of the program.
/// Do not free the returned pointer.
#[no_mangle]
pub extern "C" fn cnn_version() -> *const c_char {
    static VERSION: &[u8] = b"0.1.0\0";
    VERSION.as_ptr() as *const c_char
}

/// Converts an activation type to a string.
///
/// # Safety
/// The returned pointer is valid for the lifetime of the program.
/// Do not free the returned pointer.
#[no_mangle]
pub extern "C" fn cnn_activation_to_string(activation: CnnActivationType) -> *const c_char {
    match activation {
        CnnActivationType::Sigmoid => b"sigmoid\0".as_ptr() as *const c_char,
        CnnActivationType::Tanh => b"tanh\0".as_ptr() as *const c_char,
        CnnActivationType::ReLU => b"relu\0".as_ptr() as *const c_char,
        CnnActivationType::Linear => b"linear\0".as_ptr() as *const c_char,
    }
}

/// Converts a loss type to a string.
///
/// # Safety
/// The returned pointer is valid for the lifetime of the program.
/// Do not free the returned pointer.
#[no_mangle]
pub extern "C" fn cnn_loss_to_string(loss: CnnLossType) -> *const c_char {
    match loss {
        CnnLossType::MSE => b"mse\0".as_ptr() as *const c_char,
        CnnLossType::CrossEntropy => b"crossentropy\0".as_ptr() as *const c_char,
    }
}
