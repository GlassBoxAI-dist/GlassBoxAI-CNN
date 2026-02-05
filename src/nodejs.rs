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

//! Node.js bindings for the CUDA CNN library using napi-rs.

use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::sync::Mutex;

use crate::cnn::{
    ActivationType, BatchNormParams as RustBatchNormParams,
    ConvolutionalNeuralNetworkCUDA as RustCNN, LossType,
};

/// Activation function types for neural network layers.
#[napi(string_enum)]
pub enum JsActivationType {
    Sigmoid,
    Tanh,
    ReLU,
    Linear,
}

impl From<JsActivationType> for ActivationType {
    fn from(js: JsActivationType) -> Self {
        match js {
            JsActivationType::Sigmoid => ActivationType::Sigmoid,
            JsActivationType::Tanh => ActivationType::Tanh,
            JsActivationType::ReLU => ActivationType::ReLU,
            JsActivationType::Linear => ActivationType::Linear,
        }
    }
}

impl From<ActivationType> for JsActivationType {
    fn from(rust: ActivationType) -> Self {
        match rust {
            ActivationType::Sigmoid => JsActivationType::Sigmoid,
            ActivationType::Tanh => JsActivationType::Tanh,
            ActivationType::ReLU => JsActivationType::ReLU,
            ActivationType::Linear => JsActivationType::Linear,
        }
    }
}

/// Loss function types for training.
#[napi(string_enum)]
pub enum JsLossType {
    MSE,
    CrossEntropy,
}

impl From<JsLossType> for LossType {
    fn from(js: JsLossType) -> Self {
        match js {
            JsLossType::MSE => LossType::MSE,
            JsLossType::CrossEntropy => LossType::CrossEntropy,
        }
    }
}

impl From<LossType> for JsLossType {
    fn from(rust: LossType) -> Self {
        match rust {
            LossType::MSE => JsLossType::MSE,
            LossType::CrossEntropy => JsLossType::CrossEntropy,
        }
    }
}

/// Batch normalization parameters for a layer.
#[napi(object)]
#[derive(Clone)]
pub struct JsBatchNormParams {
    pub gamma: Vec<f64>,
    pub beta: Vec<f64>,
    pub running_mean: Vec<f64>,
    pub running_var: Vec<f64>,
    pub epsilon: f64,
    pub momentum: f64,
}

impl From<RustBatchNormParams> for JsBatchNormParams {
    fn from(rust: RustBatchNormParams) -> Self {
        Self {
            gamma: rust.gamma,
            beta: rust.beta,
            running_mean: rust.running_mean,
            running_var: rust.running_var,
            epsilon: rust.epsilon,
            momentum: rust.momentum,
        }
    }
}

impl From<JsBatchNormParams> for RustBatchNormParams {
    fn from(js: JsBatchNormParams) -> Self {
        Self {
            gamma: js.gamma,
            beta: js.beta,
            running_mean: js.running_mean,
            running_var: js.running_var,
            epsilon: js.epsilon,
            momentum: js.momentum,
        }
    }
}

/// Creates new batch normalization parameters.
#[napi]
pub fn create_batch_norm_params() -> JsBatchNormParams {
    JsBatchNormParams {
        gamma: Vec::new(),
        beta: Vec::new(),
        running_mean: Vec::new(),
        running_var: Vec::new(),
        epsilon: 1e-5,
        momentum: 0.1,
    }
}

/// Initializes batch normalization parameters for a given size.
#[napi]
pub fn initialize_batch_norm_params(size: u32) -> JsBatchNormParams {
    let size = size as usize;
    JsBatchNormParams {
        gamma: vec![1.0; size],
        beta: vec![0.0; size],
        running_mean: vec![0.0; size],
        running_var: vec![1.0; size],
        epsilon: 1e-5,
        momentum: 0.1,
    }
}

/// Options for creating a CNN.
#[napi(object)]
pub struct CnnOptions {
    pub input_width: i32,
    pub input_height: i32,
    pub input_channels: i32,
    pub conv_filters: Vec<i32>,
    pub kernel_sizes: Vec<i32>,
    pub pool_sizes: Vec<i32>,
    pub fc_sizes: Vec<i32>,
    pub output_size: i32,
    pub hidden_activation: Option<JsActivationType>,
    pub output_activation: Option<JsActivationType>,
    pub loss_type: Option<JsLossType>,
    pub learning_rate: Option<f64>,
    pub gradient_clip: Option<f64>,
}

/// CUDA-accelerated Convolutional Neural Network.
///
/// This class provides a complete CNN implementation with:
/// - Configurable convolutional, pooling, and fully-connected layers
/// - Adam optimizer for training
/// - Batch normalization support
/// - Model serialization to JSON and ONNX formats
#[napi]
pub struct CNN {
    inner: Mutex<RustCNN>,
}

#[napi]
impl CNN {
    /// Creates a new CNN with the specified architecture.
    #[napi(constructor)]
    pub fn new(options: CnnOptions) -> Result<Self> {
        let hidden_act = options
            .hidden_activation
            .map(ActivationType::from)
            .unwrap_or(ActivationType::ReLU);
        let output_act = options
            .output_activation
            .map(ActivationType::from)
            .unwrap_or(ActivationType::Linear);
        let loss = options
            .loss_type
            .map(LossType::from)
            .unwrap_or(LossType::CrossEntropy);
        let learning_rate = options.learning_rate.unwrap_or(0.001);
        let gradient_clip = options.gradient_clip.unwrap_or(5.0);

        let cnn = RustCNN::new(
            options.input_width,
            options.input_height,
            options.input_channels,
            &options.conv_filters,
            &options.kernel_sizes,
            &options.pool_sizes,
            &options.fc_sizes,
            options.output_size,
            hidden_act,
            output_act,
            loss,
            learning_rate,
            gradient_clip,
        )
        .map_err(|e| Error::from_reason(format!("Failed to create CNN: {}", e)))?;

        Ok(Self {
            inner: Mutex::new(cnn),
        })
    }

    /// Performs inference on an input image.
    ///
    /// @param imageData - Flattened input image data (size: width * height * channels)
    /// @returns Softmax probabilities for each output class
    #[napi]
    pub fn predict(&self, image_data: Vec<f64>) -> Result<Vec<f64>> {
        let mut cnn = self
            .inner
            .lock()
            .map_err(|e| Error::from_reason(format!("Failed to acquire lock: {}", e)))?;
        cnn.predict(&image_data)
            .map_err(|e| Error::from_reason(format!("Prediction failed: {}", e)))
    }

    /// Performs a single training step with one sample.
    ///
    /// @param imageData - Flattened input image data
    /// @param target - One-hot encoded target labels
    /// @returns The cross-entropy loss for this sample
    #[napi]
    pub fn train_step(&self, image_data: Vec<f64>, target: Vec<f64>) -> Result<f64> {
        let mut cnn = self
            .inner
            .lock()
            .map_err(|e| Error::from_reason(format!("Failed to acquire lock: {}", e)))?;
        cnn.train_step(&image_data, &target)
            .map_err(|e| Error::from_reason(format!("Training step failed: {}", e)))
    }

    /// Saves the model to a JSON file.
    ///
    /// @param filename - Path to the output JSON file
    #[napi]
    pub fn save_to_json(&self, filename: String) -> Result<()> {
        let cnn = self
            .inner
            .lock()
            .map_err(|e| Error::from_reason(format!("Failed to acquire lock: {}", e)))?;
        cnn.save_to_json(&filename)
            .map_err(|e| Error::from_reason(format!("Failed to save model: {}", e)))
    }

    /// Loads a model from a JSON file.
    ///
    /// @param filename - Path to the JSON file
    /// @returns A new CNN instance with loaded weights
    #[napi(factory)]
    pub fn load_from_json(filename: String) -> Result<Self> {
        let cnn = RustCNN::load_from_json(&filename)
            .map_err(|e| Error::from_reason(format!("Failed to load model: {}", e)))?;
        Ok(Self {
            inner: Mutex::new(cnn),
        })
    }

    /// Exports the model to ONNX binary format.
    ///
    /// @param filename - Path to the output ONNX file
    #[napi]
    pub fn export_to_onnx(&self, filename: String) -> Result<()> {
        let cnn = self
            .inner
            .lock()
            .map_err(|e| Error::from_reason(format!("Failed to acquire lock: {}", e)))?;
        cnn.export_to_onnx(&filename)
            .map_err(|e| Error::from_reason(format!("Failed to export model: {}", e)))
    }

    /// Imports a model from ONNX binary format.
    ///
    /// @param filename - Path to the ONNX file
    /// @returns A new CNN instance with loaded weights
    #[napi(factory)]
    pub fn import_from_onnx(filename: String) -> Result<Self> {
        let cnn = RustCNN::import_from_onnx(&filename)
            .map_err(|e| Error::from_reason(format!("Failed to import model: {}", e)))?;
        Ok(Self {
            inner: Mutex::new(cnn),
        })
    }

    /// Returns the input width.
    #[napi(getter)]
    pub fn input_width(&self) -> Result<i32> {
        let cnn = self
            .inner
            .lock()
            .map_err(|e| Error::from_reason(format!("Failed to acquire lock: {}", e)))?;
        Ok(cnn.get_input_width())
    }

    /// Returns the input height.
    #[napi(getter)]
    pub fn input_height(&self) -> Result<i32> {
        let cnn = self
            .inner
            .lock()
            .map_err(|e| Error::from_reason(format!("Failed to acquire lock: {}", e)))?;
        Ok(cnn.get_input_height())
    }

    /// Returns the number of input channels.
    #[napi(getter)]
    pub fn input_channels(&self) -> Result<i32> {
        let cnn = self
            .inner
            .lock()
            .map_err(|e| Error::from_reason(format!("Failed to acquire lock: {}", e)))?;
        Ok(cnn.get_input_channels())
    }

    /// Returns the output size (number of classes).
    #[napi(getter)]
    pub fn output_size(&self) -> Result<i32> {
        let cnn = self
            .inner
            .lock()
            .map_err(|e| Error::from_reason(format!("Failed to acquire lock: {}", e)))?;
        Ok(cnn.get_output_size())
    }

    /// Returns the learning rate.
    #[napi(getter)]
    pub fn learning_rate(&self) -> Result<f64> {
        let cnn = self
            .inner
            .lock()
            .map_err(|e| Error::from_reason(format!("Failed to acquire lock: {}", e)))?;
        Ok(cnn.get_learning_rate())
    }

    /// Sets the learning rate.
    #[napi(setter)]
    pub fn set_learning_rate(&self, lr: f64) -> Result<()> {
        let mut cnn = self
            .inner
            .lock()
            .map_err(|e| Error::from_reason(format!("Failed to acquire lock: {}", e)))?;
        cnn.set_learning_rate(lr);
        Ok(())
    }

    /// Returns the gradient clipping threshold.
    #[napi(getter)]
    pub fn gradient_clip(&self) -> Result<f64> {
        let cnn = self
            .inner
            .lock()
            .map_err(|e| Error::from_reason(format!("Failed to acquire lock: {}", e)))?;
        Ok(cnn.get_gradient_clip())
    }

    /// Sets the gradient clipping threshold.
    #[napi(setter)]
    pub fn set_gradient_clip(&self, clip: f64) -> Result<()> {
        let mut cnn = self
            .inner
            .lock()
            .map_err(|e| Error::from_reason(format!("Failed to acquire lock: {}", e)))?;
        cnn.set_gradient_clip(clip);
        Ok(())
    }

    /// Sets the dropout rate.
    #[napi]
    pub fn set_dropout_rate(&self, rate: f64) -> Result<()> {
        let mut cnn = self
            .inner
            .lock()
            .map_err(|e| Error::from_reason(format!("Failed to acquire lock: {}", e)))?;
        cnn.set_dropout_rate(rate);
        Ok(())
    }

    /// Returns the hidden activation type as a string.
    #[napi(getter)]
    pub fn hidden_activation(&self) -> Result<String> {
        let cnn = self
            .inner
            .lock()
            .map_err(|e| Error::from_reason(format!("Failed to acquire lock: {}", e)))?;
        Ok(crate::activation_to_str(cnn.get_hidden_activation()).to_string())
    }

    /// Returns the output activation type as a string.
    #[napi(getter)]
    pub fn output_activation(&self) -> Result<String> {
        let cnn = self
            .inner
            .lock()
            .map_err(|e| Error::from_reason(format!("Failed to acquire lock: {}", e)))?;
        Ok(crate::activation_to_str(cnn.get_output_activation()).to_string())
    }

    /// Returns the loss function type as a string.
    #[napi(getter)]
    pub fn loss_function(&self) -> Result<String> {
        let cnn = self
            .inner
            .lock()
            .map_err(|e| Error::from_reason(format!("Failed to acquire lock: {}", e)))?;
        Ok(crate::loss_to_str(cnn.get_loss_function()).to_string())
    }

    /// Returns the convolutional filter counts.
    #[napi(getter)]
    pub fn conv_filters(&self) -> Result<Vec<i32>> {
        let cnn = self
            .inner
            .lock()
            .map_err(|e| Error::from_reason(format!("Failed to acquire lock: {}", e)))?;
        Ok(cnn.get_conv_filters().to_vec())
    }

    /// Returns the kernel sizes.
    #[napi(getter)]
    pub fn kernel_sizes(&self) -> Result<Vec<i32>> {
        let cnn = self
            .inner
            .lock()
            .map_err(|e| Error::from_reason(format!("Failed to acquire lock: {}", e)))?;
        Ok(cnn.get_kernel_sizes().to_vec())
    }

    /// Returns the pool sizes.
    #[napi(getter)]
    pub fn pool_sizes(&self) -> Result<Vec<i32>> {
        let cnn = self
            .inner
            .lock()
            .map_err(|e| Error::from_reason(format!("Failed to acquire lock: {}", e)))?;
        Ok(cnn.get_pool_sizes().to_vec())
    }

    /// Returns the fully-connected layer sizes.
    #[napi(getter)]
    pub fn fc_sizes(&self) -> Result<Vec<i32>> {
        let cnn = self
            .inner
            .lock()
            .map_err(|e| Error::from_reason(format!("Failed to acquire lock: {}", e)))?;
        Ok(cnn.get_fc_sizes().to_vec())
    }

    /// Returns whether batch normalization is enabled.
    #[napi(getter)]
    pub fn uses_batch_norm(&self) -> Result<bool> {
        let cnn = self
            .inner
            .lock()
            .map_err(|e| Error::from_reason(format!("Failed to acquire lock: {}", e)))?;
        Ok(cnn.uses_batch_norm())
    }

    /// Initializes batch normalization for all convolutional layers.
    #[napi]
    pub fn initialize_batch_norm(&self) -> Result<()> {
        let mut cnn = self
            .inner
            .lock()
            .map_err(|e| Error::from_reason(format!("Failed to acquire lock: {}", e)))?;
        cnn.initialize_batch_norm();
        Ok(())
    }

    /// Applies batch normalization to input data for a specific layer.
    ///
    /// @param input - Input data to normalize
    /// @param layerIdx - Index of the layer
    /// @param training - Whether the model is in training mode
    /// @returns Normalized output data
    #[napi]
    pub fn apply_batch_norm(
        &self,
        input: Vec<f64>,
        layer_idx: u32,
        training: bool,
    ) -> Result<Vec<f64>> {
        let cnn = self
            .inner
            .lock()
            .map_err(|e| Error::from_reason(format!("Failed to acquire lock: {}", e)))?;
        Ok(cnn.apply_batch_norm(&input, layer_idx as usize, training))
    }
}
