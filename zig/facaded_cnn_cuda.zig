// MIT License
//
// Copyright (c) 2025 Matthew Abbott
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

//! Zig bindings for the CUDA-accelerated Convolutional Neural Network library.
//!
//! Example usage:
//! ```zig
//! const cnn = @import("facaded_cnn_cuda");
//!
//! pub fn main() !void {
//!     const conv_filters = [_]c_int{ 32, 64 };
//!     const kernel_sizes = [_]c_int{ 3, 3 };
//!     const pool_sizes = [_]c_int{ 2, 2 };
//!     const fc_sizes = [_]c_int{128};
//!
//!     var net = try cnn.CNN.init(.{
//!         .input_width = 28,
//!         .input_height = 28,
//!         .input_channels = 1,
//!         .conv_filters = &conv_filters,
//!         .conv_filters_len = 2,
//!         .kernel_sizes = &kernel_sizes,
//!         .kernel_sizes_len = 2,
//!         .pool_sizes = &pool_sizes,
//!         .pool_sizes_len = 2,
//!         .fc_sizes = &fc_sizes,
//!         .fc_sizes_len = 1,
//!         .output_size = 10,
//!     });
//!     defer net.deinit();
//!
//!     var input: [784]f64 = .{0.0} ** 784;
//!     const output = try net.predict(&input);
//! }
//! ```

const std = @import("std");

pub const CnnError = enum(c_int) {
    success = 0,
    null_pointer = 1,
    invalid_parameter = 2,
    creation_failed = 3,
    prediction_failed = 4,
    training_failed = 5,
    save_failed = 6,
    load_failed = 7,
    export_failed = 8,
    import_failed = 9,
    buffer_too_small = 10,
    unknown = 255,
};

pub const ActivationType = enum(c_int) {
    sigmoid = 0,
    tanh = 1,
    relu = 2,
    linear = 3,
};

pub const LossType = enum(c_int) {
    mse = 0,
    cross_entropy = 1,
};

pub const CnnConfig = extern struct {
    input_width: c_int = 28,
    input_height: c_int = 28,
    input_channels: c_int = 1,
    conv_filters: [*]const c_int,
    conv_filters_len: c_int,
    kernel_sizes: [*]const c_int,
    kernel_sizes_len: c_int,
    pool_sizes: [*]const c_int,
    pool_sizes_len: c_int,
    fc_sizes: [*]const c_int,
    fc_sizes_len: c_int,
    output_size: c_int = 10,
    hidden_activation: ActivationType = .relu,
    output_activation: ActivationType = .linear,
    loss_type: LossType = .cross_entropy,
    learning_rate: f64 = 0.001,
    gradient_clip: f64 = 5.0,
};

const CnnHandle = opaque {};

extern "c" fn cnn_create(config: *const CnnConfig, out_handle: *?*CnnHandle) CnnError;
extern "c" fn cnn_destroy(handle: ?*CnnHandle) void;
extern "c" fn cnn_predict(handle: ?*CnnHandle, image_data: [*]const f64, image_len: c_int, output: [*]f64, output_len: c_int) CnnError;
extern "c" fn cnn_train_step(handle: ?*CnnHandle, image_data: [*]const f64, image_len: c_int, target: [*]const f64, target_len: c_int, out_loss: *f64) CnnError;
extern "c" fn cnn_save_to_json(handle: ?*const CnnHandle, filename: [*:0]const u8) CnnError;
extern "c" fn cnn_load_from_json(filename: [*:0]const u8, out_handle: *?*CnnHandle) CnnError;
extern "c" fn cnn_export_to_onnx(handle: ?*const CnnHandle, filename: [*:0]const u8) CnnError;
extern "c" fn cnn_import_from_onnx(filename: [*:0]const u8, out_handle: *?*CnnHandle) CnnError;
extern "c" fn cnn_get_input_width(handle: ?*const CnnHandle) c_int;
extern "c" fn cnn_get_input_height(handle: ?*const CnnHandle) c_int;
extern "c" fn cnn_get_input_channels(handle: ?*const CnnHandle) c_int;
extern "c" fn cnn_get_output_size(handle: ?*const CnnHandle) c_int;
extern "c" fn cnn_get_learning_rate(handle: ?*const CnnHandle) f64;
extern "c" fn cnn_set_learning_rate(handle: ?*CnnHandle, lr: f64) void;
extern "c" fn cnn_get_gradient_clip(handle: ?*const CnnHandle) f64;
extern "c" fn cnn_set_gradient_clip(handle: ?*CnnHandle, clip: f64) void;
extern "c" fn cnn_set_dropout_rate(handle: ?*CnnHandle, rate: f64) void;
extern "c" fn cnn_get_hidden_activation(handle: ?*const CnnHandle) ActivationType;
extern "c" fn cnn_get_output_activation(handle: ?*const CnnHandle) ActivationType;
extern "c" fn cnn_get_loss_function(handle: ?*const CnnHandle) LossType;
extern "c" fn cnn_uses_batch_norm(handle: ?*const CnnHandle) c_int;
extern "c" fn cnn_initialize_batch_norm(handle: ?*CnnHandle) void;
extern "c" fn cnn_get_last_error() ?[*:0]const u8;
extern "c" fn cnn_clear_error() void;
extern "c" fn cnn_version() [*:0]const u8;
extern "c" fn cnn_activation_to_string(activation: ActivationType) [*:0]const u8;
extern "c" fn cnn_loss_to_string(loss: LossType) [*:0]const u8;

pub const Error = error{
    NullPointer,
    InvalidParameter,
    CreationFailed,
    PredictionFailed,
    TrainingFailed,
    SaveFailed,
    LoadFailed,
    ExportFailed,
    ImportFailed,
    BufferTooSmall,
    Unknown,
};

fn mapError(err: CnnError) Error {
    return switch (err) {
        .null_pointer => Error.NullPointer,
        .invalid_parameter => Error.InvalidParameter,
        .creation_failed => Error.CreationFailed,
        .prediction_failed => Error.PredictionFailed,
        .training_failed => Error.TrainingFailed,
        .save_failed => Error.SaveFailed,
        .load_failed => Error.LoadFailed,
        .export_failed => Error.ExportFailed,
        .import_failed => Error.ImportFailed,
        .buffer_too_small => Error.BufferTooSmall,
        else => Error.Unknown,
    };
}

fn checkError(err: CnnError) Error!void {
    if (err != .success) {
        return mapError(err);
    }
}

/// CUDA-accelerated Convolutional Neural Network.
pub const CNN = struct {
    handle: *CnnHandle,

    /// Creates a new CNN with the specified configuration.
    pub fn init(config: CnnConfig) Error!CNN {
        var handle: ?*CnnHandle = null;
        const err = cnn_create(&config, &handle);
        if (err != .success) {
            return mapError(err);
        }
        return CNN{ .handle = handle.? };
    }

    /// Loads a CNN from a JSON file.
    pub fn loadFromJson(filename: [*:0]const u8) Error!CNN {
        var handle: ?*CnnHandle = null;
        const err = cnn_load_from_json(filename, &handle);
        if (err != .success) {
            return mapError(err);
        }
        return CNN{ .handle = handle.? };
    }

    /// Imports a CNN from an ONNX file.
    pub fn importFromOnnx(filename: [*:0]const u8) Error!CNN {
        var handle: ?*CnnHandle = null;
        const err = cnn_import_from_onnx(filename, &handle);
        if (err != .success) {
            return mapError(err);
        }
        return CNN{ .handle = handle.? };
    }

    /// Destroys the CNN and frees its resources.
    pub fn deinit(self: *CNN) void {
        cnn_destroy(self.handle);
        self.handle = undefined;
    }

    /// Performs inference on input data.
    /// Returns a slice into the provided output buffer with the results.
    pub fn predict(self: *CNN, image_data: []const f64, output_buf: []f64) Error![]f64 {
        const out_size = self.outputSize();
        const len: usize = @intCast(if (out_size > 0) out_size else 0);
        if (output_buf.len < len) {
            return Error.BufferTooSmall;
        }
        const err = cnn_predict(
            self.handle,
            image_data.ptr,
            @intCast(image_data.len),
            output_buf.ptr,
            @intCast(len),
        );
        try checkError(err);
        return output_buf[0..len];
    }

    /// Performs a single training step.
    pub fn trainStep(self: *CNN, image_data: []const f64, target: []const f64) Error!f64 {
        var loss: f64 = 0.0;
        const err = cnn_train_step(
            self.handle,
            image_data.ptr,
            @intCast(image_data.len),
            target.ptr,
            @intCast(target.len),
            &loss,
        );
        try checkError(err);
        return loss;
    }

    /// Saves the model to a JSON file.
    pub fn saveToJson(self: *const CNN, filename: [*:0]const u8) Error!void {
        try checkError(cnn_save_to_json(self.handle, filename));
    }

    /// Exports the model to ONNX format.
    pub fn exportToOnnx(self: *const CNN, filename: [*:0]const u8) Error!void {
        try checkError(cnn_export_to_onnx(self.handle, filename));
    }

    /// Gets the input width.
    pub fn inputWidth(self: *const CNN) i32 {
        return @intCast(cnn_get_input_width(self.handle));
    }

    /// Gets the input height.
    pub fn inputHeight(self: *const CNN) i32 {
        return @intCast(cnn_get_input_height(self.handle));
    }

    /// Gets the number of input channels.
    pub fn inputChannels(self: *const CNN) i32 {
        return @intCast(cnn_get_input_channels(self.handle));
    }

    /// Gets the output size (number of classes).
    pub fn outputSize(self: *const CNN) i32 {
        return @intCast(cnn_get_output_size(self.handle));
    }

    /// Gets the learning rate.
    pub fn learningRate(self: *const CNN) f64 {
        return cnn_get_learning_rate(self.handle);
    }

    /// Sets the learning rate.
    pub fn setLearningRate(self: *CNN, lr: f64) void {
        cnn_set_learning_rate(self.handle, lr);
    }

    /// Gets the gradient clipping threshold.
    pub fn gradientClip(self: *const CNN) f64 {
        return cnn_get_gradient_clip(self.handle);
    }

    /// Sets the gradient clipping threshold.
    pub fn setGradientClip(self: *CNN, clip: f64) void {
        cnn_set_gradient_clip(self.handle, clip);
    }

    /// Sets the dropout rate.
    pub fn setDropoutRate(self: *CNN, rate: f64) void {
        cnn_set_dropout_rate(self.handle, rate);
    }

    /// Gets the hidden activation type.
    pub fn hiddenActivation(self: *const CNN) ActivationType {
        return cnn_get_hidden_activation(self.handle);
    }

    /// Gets the output activation type.
    pub fn outputActivation(self: *const CNN) ActivationType {
        return cnn_get_output_activation(self.handle);
    }

    /// Gets the loss function type.
    pub fn lossFunction(self: *const CNN) LossType {
        return cnn_get_loss_function(self.handle);
    }

    /// Gets whether batch normalization is enabled.
    pub fn usesBatchNorm(self: *const CNN) bool {
        return cnn_uses_batch_norm(self.handle) != 0;
    }

    /// Initializes batch normalization for all convolutional layers.
    pub fn initializeBatchNorm(self: *CNN) void {
        cnn_initialize_batch_norm(self.handle);
    }
};

/// Returns the last error message, or null if none.
pub fn getLastError() ?[]const u8 {
    const ptr = cnn_get_last_error();
    if (ptr) |p| {
        return std.mem.span(p);
    }
    return null;
}

/// Clears the last error message.
pub fn clearError() void {
    cnn_clear_error();
}

/// Returns the library version string.
pub fn version() []const u8 {
    return std.mem.span(cnn_version());
}

/// Converts an activation type to its string representation.
pub fn activationToString(activation: ActivationType) []const u8 {
    return std.mem.span(cnn_activation_to_string(activation));
}

/// Converts a loss type to its string representation.
pub fn lossToString(loss: LossType) []const u8 {
    return std.mem.span(cnn_loss_to_string(loss));
}
