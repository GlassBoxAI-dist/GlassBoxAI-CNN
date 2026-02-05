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
 * @file facaded_cnn_cuda.hpp
 * @brief C++ wrapper for the CUDA-accelerated Convolutional Neural Network library.
 *
 * This header provides an idiomatic C++ API that wraps the C API.
 *
 * @example
 * @code
 * #include "facaded_cnn_cuda.hpp"
 * #include <iostream>
 *
 * int main() {
 *     using namespace facaded_cnn;
 *
 *     // Create a CNN for MNIST classification
 *     CNN cnn({
 *         .inputWidth = 28,
 *         .inputHeight = 28,
 *         .inputChannels = 1,
 *         .convFilters = {32, 64},
 *         .kernelSizes = {3, 3},
 *         .poolSizes = {2, 2},
 *         .fcSizes = {128},
 *         .outputSize = 10,
 *         .hiddenActivation = ActivationType::ReLU,
 *         .outputActivation = ActivationType::Linear,
 *         .lossType = LossType::CrossEntropy,
 *         .learningRate = 0.001,
 *         .gradientClip = 5.0
 *     });
 *
 *     // Make a prediction
 *     std::vector<double> input(784, 0.0);
 *     auto output = cnn.predict(input);
 *
 *     // Find predicted class
 *     int predictedClass = std::max_element(output.begin(), output.end()) - output.begin();
 *     std::cout << "Predicted class: " << predictedClass << std::endl;
 *
 *     return 0;
 * }
 * @endcode
 */

#ifndef FACADED_CNN_CUDA_HPP
#define FACADED_CNN_CUDA_HPP

#include "facaded_cnn_cuda.h"
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace facaded_cnn {

/**
 * @brief Exception thrown when a CNN operation fails.
 */
class CnnException : public std::runtime_error {
public:
    explicit CnnException(const std::string& message)
        : std::runtime_error(message) {}

    explicit CnnException(CnnError error)
        : std::runtime_error(getErrorMessage(error)), error_(error) {}

    CnnError error() const { return error_; }

private:
    CnnError error_ = CNN_ERROR_UNKNOWN;

    static std::string getErrorMessage(CnnError error) {
        const char* lastError = cnn_get_last_error();
        if (lastError) {
            return std::string(lastError);
        }
        switch (error) {
            case CNN_SUCCESS: return "Success";
            case CNN_ERROR_NULL_POINTER: return "Null pointer";
            case CNN_ERROR_INVALID_PARAMETER: return "Invalid parameter";
            case CNN_ERROR_CREATION_FAILED: return "Creation failed";
            case CNN_ERROR_PREDICTION_FAILED: return "Prediction failed";
            case CNN_ERROR_TRAINING_FAILED: return "Training failed";
            case CNN_ERROR_SAVE_FAILED: return "Save failed";
            case CNN_ERROR_LOAD_FAILED: return "Load failed";
            case CNN_ERROR_EXPORT_FAILED: return "Export failed";
            case CNN_ERROR_IMPORT_FAILED: return "Import failed";
            case CNN_ERROR_BUFFER_TOO_SMALL: return "Buffer too small";
            default: return "Unknown error";
        }
    }
};

/**
 * @brief Activation function types.
 */
enum class ActivationType {
    Sigmoid = CNN_ACTIVATION_SIGMOID,
    Tanh = CNN_ACTIVATION_TANH,
    ReLU = CNN_ACTIVATION_RELU,
    Linear = CNN_ACTIVATION_LINEAR
};

/**
 * @brief Loss function types.
 */
enum class LossType {
    MSE = CNN_LOSS_MSE,
    CrossEntropy = CNN_LOSS_CROSS_ENTROPY
};

/**
 * @brief Configuration for creating a CNN.
 */
struct Config {
    int inputWidth = 28;
    int inputHeight = 28;
    int inputChannels = 1;
    std::vector<int> convFilters = {32, 64};
    std::vector<int> kernelSizes = {3, 3};
    std::vector<int> poolSizes = {2, 2};
    std::vector<int> fcSizes = {128};
    int outputSize = 10;
    ActivationType hiddenActivation = ActivationType::ReLU;
    ActivationType outputActivation = ActivationType::Linear;
    LossType lossType = LossType::CrossEntropy;
    double learningRate = 0.001;
    double gradientClip = 5.0;
};

/**
 * @brief CUDA-accelerated Convolutional Neural Network.
 *
 * This class provides an RAII wrapper around the C API for CNNs.
 */
class CNN {
public:
    /**
     * @brief Creates a new CNN with the specified configuration.
     * @param config Configuration for the CNN.
     * @throws CnnException if creation fails.
     */
    explicit CNN(const Config& config) {
        CnnConfig cConfig = {
            config.inputWidth,
            config.inputHeight,
            config.inputChannels,
            config.convFilters.data(),
            static_cast<int>(config.convFilters.size()),
            config.kernelSizes.data(),
            static_cast<int>(config.kernelSizes.size()),
            config.poolSizes.data(),
            static_cast<int>(config.poolSizes.size()),
            config.fcSizes.data(),
            static_cast<int>(config.fcSizes.size()),
            config.outputSize,
            static_cast<CnnActivationType>(config.hiddenActivation),
            static_cast<CnnActivationType>(config.outputActivation),
            static_cast<CnnLossType>(config.lossType),
            config.learningRate,
            config.gradientClip
        };

        CnnHandle* handle = nullptr;
        CnnError err = cnn_create(&cConfig, &handle);
        if (err != CNN_SUCCESS) {
            throw CnnException(err);
        }
        handle_.reset(handle);
    }

    /**
     * @brief Loads a CNN from a JSON file.
     * @param filename Path to the JSON file.
     * @return A new CNN instance.
     * @throws CnnException if loading fails.
     */
    static CNN loadFromJson(const std::string& filename) {
        CnnHandle* handle = nullptr;
        CnnError err = cnn_load_from_json(filename.c_str(), &handle);
        if (err != CNN_SUCCESS) {
            throw CnnException(err);
        }
        return CNN(handle);
    }

    /**
     * @brief Imports a CNN from an ONNX file.
     * @param filename Path to the ONNX file.
     * @return A new CNN instance.
     * @throws CnnException if importing fails.
     */
    static CNN importFromOnnx(const std::string& filename) {
        CnnHandle* handle = nullptr;
        CnnError err = cnn_import_from_onnx(filename.c_str(), &handle);
        if (err != CNN_SUCCESS) {
            throw CnnException(err);
        }
        return CNN(handle);
    }

    // Move-only semantics
    CNN(CNN&& other) noexcept = default;
    CNN& operator=(CNN&& other) noexcept = default;
    CNN(const CNN&) = delete;
    CNN& operator=(const CNN&) = delete;

    /**
     * @brief Performs inference on an input image.
     * @param imageData Flattened input image data.
     * @return Softmax probabilities for each output class.
     * @throws CnnException if prediction fails.
     */
    std::vector<double> predict(const std::vector<double>& imageData) {
        int outputSize = cnn_get_output_size(handle_.get());
        std::vector<double> output(outputSize);

        CnnError err = cnn_predict(
            handle_.get(),
            imageData.data(),
            static_cast<int>(imageData.size()),
            output.data(),
            static_cast<int>(output.size())
        );
        if (err != CNN_SUCCESS) {
            throw CnnException(err);
        }
        return output;
    }

    /**
     * @brief Performs a single training step.
     * @param imageData Flattened input image data.
     * @param target One-hot encoded target labels.
     * @return The cross-entropy loss for this sample.
     * @throws CnnException if training fails.
     */
    double trainStep(const std::vector<double>& imageData, const std::vector<double>& target) {
        double loss = 0.0;
        CnnError err = cnn_train_step(
            handle_.get(),
            imageData.data(),
            static_cast<int>(imageData.size()),
            target.data(),
            static_cast<int>(target.size()),
            &loss
        );
        if (err != CNN_SUCCESS) {
            throw CnnException(err);
        }
        return loss;
    }

    /**
     * @brief Saves the model to a JSON file.
     * @param filename Path to the output JSON file.
     * @throws CnnException if saving fails.
     */
    void saveToJson(const std::string& filename) const {
        CnnError err = cnn_save_to_json(handle_.get(), filename.c_str());
        if (err != CNN_SUCCESS) {
            throw CnnException(err);
        }
    }

    /**
     * @brief Exports the model to ONNX format.
     * @param filename Path to the output ONNX file.
     * @throws CnnException if exporting fails.
     */
    void exportToOnnx(const std::string& filename) const {
        CnnError err = cnn_export_to_onnx(handle_.get(), filename.c_str());
        if (err != CNN_SUCCESS) {
            throw CnnException(err);
        }
    }

    // Property getters
    int inputWidth() const { return cnn_get_input_width(handle_.get()); }
    int inputHeight() const { return cnn_get_input_height(handle_.get()); }
    int inputChannels() const { return cnn_get_input_channels(handle_.get()); }
    int outputSize() const { return cnn_get_output_size(handle_.get()); }

    double learningRate() const { return cnn_get_learning_rate(handle_.get()); }
    void setLearningRate(double lr) { cnn_set_learning_rate(handle_.get(), lr); }

    double gradientClip() const { return cnn_get_gradient_clip(handle_.get()); }
    void setGradientClip(double clip) { cnn_set_gradient_clip(handle_.get(), clip); }

    void setDropoutRate(double rate) { cnn_set_dropout_rate(handle_.get(), rate); }

    ActivationType hiddenActivation() const {
        return static_cast<ActivationType>(cnn_get_hidden_activation(handle_.get()));
    }

    ActivationType outputActivation() const {
        return static_cast<ActivationType>(cnn_get_output_activation(handle_.get()));
    }

    LossType lossFunction() const {
        return static_cast<LossType>(cnn_get_loss_function(handle_.get()));
    }

    bool usesBatchNorm() const { return cnn_uses_batch_norm(handle_.get()) != 0; }

    void initializeBatchNorm() { cnn_initialize_batch_norm(handle_.get()); }

private:
    struct HandleDeleter {
        void operator()(CnnHandle* handle) const {
            if (handle) {
                cnn_destroy(handle);
            }
        }
    };

    std::unique_ptr<CnnHandle, HandleDeleter> handle_;

    // Private constructor for static factory methods
    explicit CNN(CnnHandle* handle) : handle_(handle) {}
};

/**
 * @brief Returns the library version.
 * @return Version string (e.g., "0.1.0").
 */
inline std::string version() {
    return cnn_version();
}

/**
 * @brief Converts an activation type to a string.
 * @param activation The activation type.
 * @return String representation.
 */
inline std::string toString(ActivationType activation) {
    return cnn_activation_to_string(static_cast<CnnActivationType>(activation));
}

/**
 * @brief Converts a loss type to a string.
 * @param loss The loss type.
 * @return String representation.
 */
inline std::string toString(LossType loss) {
    return cnn_loss_to_string(static_cast<CnnLossType>(loss));
}

} // namespace facaded_cnn

#endif /* FACADED_CNN_CUDA_HPP */
