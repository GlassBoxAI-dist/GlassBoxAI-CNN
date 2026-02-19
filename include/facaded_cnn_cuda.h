/**
 * @file
 * @ingroup CNN_Internal_Logic
 */
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

/**
 * @file facaded_cnn_cuda.h
 * @brief C API for the CUDA-accelerated Convolutional Neural Network library.
 *
 * This header provides a C-compatible API for creating, training, and using
 * CUDA-accelerated CNNs from C and C++ code.
 *
 * @example
 * @code
 * #include "facaded_cnn_cuda.h"
 *
 * int main() {
 *     // Configure the CNN
 *     int conv_filters[] = {32, 64};
 *     int kernel_sizes[] = {3, 3};
 *     int pool_sizes[] = {2, 2};
 *     int fc_sizes[] = {128};
 *
 *     CnnConfig config = {
 *         .input_width = 28,
 *         .input_height = 28,
 *         .input_channels = 1,
 *         .conv_filters = conv_filters,
 *         .conv_filters_len = 2,
 *         .kernel_sizes = kernel_sizes,
 *         .kernel_sizes_len = 2,
 *         .pool_sizes = pool_sizes,
 *         .pool_sizes_len = 2,
 *         .fc_sizes = fc_sizes,
 *         .fc_sizes_len = 1,
 *         .output_size = 10,
 *         .hidden_activation = CNN_ACTIVATION_RELU,
 *         .output_activation = CNN_ACTIVATION_LINEAR,
 *         .loss_type = CNN_LOSS_CROSS_ENTROPY,
 *         .learning_rate = 0.001,
 *         .gradient_clip = 5.0
 *     };
 *
 *     // Create the CNN
 *     CnnHandle* cnn = NULL;
 *     CnnError err = cnn_create(&config, &cnn);
 *     if (err != CNN_SUCCESS) {
 *         printf("Error: %s\n", cnn_get_last_error());
 *         return 1;
 *     }
 *
 *     // Make a prediction
 *     double input[784] = {0};
 *     double output[10];
 *     err = cnn_predict(cnn, input, 784, output, 10);
 *
 *     // Clean up
 *     cnn_destroy(cnn);
 *     return 0;
 * }
 * @endcode
 */

#ifndef FACADED_CNN_CUDA_H
#define FACADED_CNN_CUDA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Types
 * ============================================================================ */

/**
 * @brief Opaque handle to a CNN instance.
 *
 * This handle is created by cnn_create(), cnn_load_from_json(), or
 * cnn_import_from_onnx(), and must be destroyed with cnn_destroy().
 */
typedef struct CnnHandle CnnHandle;

/**
 * @brief Error codes returned by C API functions.
 */
typedef enum CnnError {
    /** Operation completed successfully. */
    CNN_SUCCESS = 0,
    /** Null pointer was passed where a valid pointer was expected. */
    CNN_ERROR_NULL_POINTER = 1,
    /** Invalid parameter value. */
    CNN_ERROR_INVALID_PARAMETER = 2,
    /** Failed to create CNN. */
    CNN_ERROR_CREATION_FAILED = 3,
    /** Prediction failed. */
    CNN_ERROR_PREDICTION_FAILED = 4,
    /** Training step failed. */
    CNN_ERROR_TRAINING_FAILED = 5,
    /** Failed to save model. */
    CNN_ERROR_SAVE_FAILED = 6,
    /** Failed to load model. */
    CNN_ERROR_LOAD_FAILED = 7,
    /** Failed to export model. */
    CNN_ERROR_EXPORT_FAILED = 8,
    /** Failed to import model. */
    CNN_ERROR_IMPORT_FAILED = 9,
    /** Buffer too small. */
    CNN_ERROR_BUFFER_TOO_SMALL = 10,
    /** Unknown error. */
    CNN_ERROR_UNKNOWN = 255
} CnnError;

/**
 * @brief Activation function types.
 */
typedef enum CnnActivationType {
    /** Sigmoid activation function. */
    CNN_ACTIVATION_SIGMOID = 0,
    /** Hyperbolic tangent activation function. */
    CNN_ACTIVATION_TANH = 1,
    /** Rectified Linear Unit activation function. */
    CNN_ACTIVATION_RELU = 2,
    /** Linear (identity) activation function. */
    CNN_ACTIVATION_LINEAR = 3
} CnnActivationType;

/**
 * @brief Loss function types.
 */
typedef enum CnnLossType {
    /** Mean Squared Error loss. */
    CNN_LOSS_MSE = 0,
    /** Cross-Entropy loss (for classification). */
    CNN_LOSS_CROSS_ENTROPY = 1
} CnnLossType;

/**
 * @brief Configuration for creating a CNN.
 */
typedef struct CnnConfig {
    /** Width of input images. */
    int input_width;
    /** Height of input images. */
    int input_height;
    /** Number of input channels (e.g., 1 for grayscale, 3 for RGB). */
    int input_channels;
    /** Array of filter counts for each convolutional layer. */
    const int* conv_filters;
    /** Length of conv_filters array. */
    int conv_filters_len;
    /** Array of kernel sizes for each convolutional layer. */
    const int* kernel_sizes;
    /** Length of kernel_sizes array. */
    int kernel_sizes_len;
    /** Array of pooling sizes for each pooling layer. */
    const int* pool_sizes;
    /** Length of pool_sizes array. */
    int pool_sizes_len;
    /** Array of neuron counts for each fully-connected hidden layer. */
    const int* fc_sizes;
    /** Length of fc_sizes array. */
    int fc_sizes_len;
    /** Number of output classes. */
    int output_size;
    /** Activation function for hidden layers. */
    CnnActivationType hidden_activation;
    /** Activation function for output layer. */
    CnnActivationType output_activation;
    /** Loss function for training. */
    CnnLossType loss_type;
    /** Learning rate for Adam optimizer. */
    double learning_rate;
    /** Gradient clipping threshold. */
    double gradient_clip;
} CnnConfig;

/* ============================================================================
 * Lifecycle Functions
 * ============================================================================ */

/**
 * @brief Creates a new CNN with the specified configuration.
 *
 * @param config Pointer to configuration struct.
 * @param out_handle Pointer to receive the created handle.
 * @return CNN_SUCCESS on success, or an error code on failure.
 */
CnnError cnn_create(const CnnConfig* config, CnnHandle** out_handle);

/**
 * @brief Destroys a CNN instance and frees its resources.
 *
 * @param handle The CNN handle to destroy. May be NULL.
 */
void cnn_destroy(CnnHandle* handle);

/* ============================================================================
 * Inference and Training
 * ============================================================================ */

/**
 * @brief Performs inference on an input image.
 *
 * @param handle The CNN handle.
 * @param image_data Flattened input image data.
 * @param image_len Length of image_data array.
 * @param output Buffer to receive softmax probabilities.
 * @param output_len Length of output buffer (must be >= output_size).
 * @return CNN_SUCCESS on success, or an error code on failure.
 */
CnnError cnn_predict(
    CnnHandle* handle,
    const double* image_data,
    int image_len,
    double* output,
    int output_len
);

/**
 * @brief Performs a single training step.
 *
 * @param handle The CNN handle.
 * @param image_data Flattened input image data.
 * @param image_len Length of image_data array.
 * @param target One-hot encoded target labels.
 * @param target_len Length of target array.
 * @param out_loss Pointer to receive the loss value.
 * @return CNN_SUCCESS on success, or an error code on failure.
 */
CnnError cnn_train_step(
    CnnHandle* handle,
    const double* image_data,
    int image_len,
    const double* target,
    int target_len,
    double* out_loss
);

/* ============================================================================
 * Serialization
 * ============================================================================ */

/**
 * @brief Saves the model to a JSON file.
 *
 * @param handle The CNN handle.
 * @param filename Path to the output JSON file.
 * @return CNN_SUCCESS on success, or an error code on failure.
 */
CnnError cnn_save_to_json(const CnnHandle* handle, const char* filename);

/**
 * @brief Loads a model from a JSON file.
 *
 * @param filename Path to the JSON file.
 * @param out_handle Pointer to receive the loaded handle.
 * @return CNN_SUCCESS on success, or an error code on failure.
 */
CnnError cnn_load_from_json(const char* filename, CnnHandle** out_handle);

/**
 * @brief Exports the model to ONNX binary format.
 *
 * @param handle The CNN handle.
 * @param filename Path to the output ONNX file.
 * @return CNN_SUCCESS on success, or an error code on failure.
 */
CnnError cnn_export_to_onnx(const CnnHandle* handle, const char* filename);

/**
 * @brief Imports a model from ONNX binary format.
 *
 * @param filename Path to the ONNX file.
 * @param out_handle Pointer to receive the loaded handle.
 * @return CNN_SUCCESS on success, or an error code on failure.
 */
CnnError cnn_import_from_onnx(const char* filename, CnnHandle** out_handle);

/* ============================================================================
 * Property Getters and Setters
 * ============================================================================ */

/**
 * @brief Gets the input width.
 * @param handle The CNN handle.
 * @return Input width, or -1 if handle is NULL.
 */
int cnn_get_input_width(const CnnHandle* handle);

/**
 * @brief Gets the input height.
 * @param handle The CNN handle.
 * @return Input height, or -1 if handle is NULL.
 */
int cnn_get_input_height(const CnnHandle* handle);

/**
 * @brief Gets the number of input channels.
 * @param handle The CNN handle.
 * @return Number of input channels, or -1 if handle is NULL.
 */
int cnn_get_input_channels(const CnnHandle* handle);

/**
 * @brief Gets the output size (number of classes).
 * @param handle The CNN handle.
 * @return Output size, or -1 if handle is NULL.
 */
int cnn_get_output_size(const CnnHandle* handle);

/**
 * @brief Gets the learning rate.
 * @param handle The CNN handle.
 * @return Learning rate, or -1.0 if handle is NULL.
 */
double cnn_get_learning_rate(const CnnHandle* handle);

/**
 * @brief Sets the learning rate.
 * @param handle The CNN handle.
 * @param lr New learning rate.
 */
void cnn_set_learning_rate(CnnHandle* handle, double lr);

/**
 * @brief Gets the gradient clipping threshold.
 * @param handle The CNN handle.
 * @return Gradient clip value, or -1.0 if handle is NULL.
 */
double cnn_get_gradient_clip(const CnnHandle* handle);

/**
 * @brief Sets the gradient clipping threshold.
 * @param handle The CNN handle.
 * @param clip New gradient clip value.
 */
void cnn_set_gradient_clip(CnnHandle* handle, double clip);

/**
 * @brief Sets the dropout rate.
 * @param handle The CNN handle.
 * @param rate Dropout rate (0.0 to 1.0).
 */
void cnn_set_dropout_rate(CnnHandle* handle, double rate);

/**
 * @brief Gets the hidden activation type.
 * @param handle The CNN handle.
 * @return Hidden activation type.
 */
CnnActivationType cnn_get_hidden_activation(const CnnHandle* handle);

/**
 * @brief Gets the output activation type.
 * @param handle The CNN handle.
 * @return Output activation type.
 */
CnnActivationType cnn_get_output_activation(const CnnHandle* handle);

/**
 * @brief Gets the loss function type.
 * @param handle The CNN handle.
 * @return Loss function type.
 */
CnnLossType cnn_get_loss_function(const CnnHandle* handle);

/**
 * @brief Gets whether batch normalization is enabled.
 * @param handle The CNN handle.
 * @return 1 if enabled, 0 if disabled or handle is NULL.
 */
int cnn_uses_batch_norm(const CnnHandle* handle);

/**
 * @brief Initializes batch normalization for all convolutional layers.
 * @param handle The CNN handle.
 */
void cnn_initialize_batch_norm(CnnHandle* handle);

/* ============================================================================
 * Error Handling
 * ============================================================================ */

/**
 * @brief Gets the last error message.
 *
 * The returned pointer is valid until the next call to any C API function
 * on the current thread.
 *
 * @return Pointer to error message, or NULL if no error occurred.
 */
const char* cnn_get_last_error(void);

/**
 * @brief Clears the last error message.
 */
void cnn_clear_error(void);

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

/**
 * @brief Returns the library version.
 * @return Pointer to version string (e.g., "0.1.0").
 */
const char* cnn_version(void);

/**
 * @brief Converts an activation type to a string.
 * @param activation The activation type.
 * @return Pointer to string representation.
 */
const char* cnn_activation_to_string(CnnActivationType activation);

/**
 * @brief Converts a loss type to a string.
 * @param loss The loss type.
 * @return Pointer to string representation.
 */
const char* cnn_loss_to_string(CnnLossType loss);

#ifdef __cplusplus
}
#endif

#endif /* FACADED_CNN_CUDA_H */
