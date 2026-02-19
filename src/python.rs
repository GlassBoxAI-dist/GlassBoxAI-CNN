//! @file
//! @ingroup CNN_Internal_Logic
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

//! Python bindings for the CUDA CNN library using PyO3.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::sync::Mutex;

use crate::cnn::{
    ActivationType, BatchNormParams as RustBatchNormParams,
    ConvolutionalNeuralNetworkCUDA as RustCNN, LossType,
};

/// Python wrapper for ActivationType enum.
#[pyclass(name = "ActivationType")]
#[derive(Clone)]
pub struct PyActivationType {
    inner: ActivationType,
}

#[pymethods]
impl PyActivationType {
    #[staticmethod]
    pub fn sigmoid() -> Self {
        Self { inner: ActivationType::Sigmoid }
    }

    #[staticmethod]
    pub fn tanh() -> Self {
        Self { inner: ActivationType::Tanh }
    }

    #[staticmethod]
    pub fn relu() -> Self {
        Self { inner: ActivationType::ReLU }
    }

    #[staticmethod]
    pub fn linear() -> Self {
        Self { inner: ActivationType::Linear }
    }

    #[staticmethod]
    pub fn from_str(s: &str) -> PyResult<Self> {
        let inner = match s.to_lowercase().as_str() {
            "sigmoid" => ActivationType::Sigmoid,
            "tanh" => ActivationType::Tanh,
            "relu" => ActivationType::ReLU,
            "linear" => ActivationType::Linear,
            _ => return Err(PyValueError::new_err(format!("Unknown activation type: {}", s))),
        };
        Ok(Self { inner })
    }

    pub fn __str__(&self) -> &'static str {
        match self.inner {
            ActivationType::Sigmoid => "sigmoid",
            ActivationType::Tanh => "tanh",
            ActivationType::ReLU => "relu",
            ActivationType::Linear => "linear",
        }
    }

    pub fn __repr__(&self) -> String {
        format!("ActivationType.{}", self.__str__())
    }
}

/// Python wrapper for LossType enum.
#[pyclass(name = "LossType")]
#[derive(Clone)]
pub struct PyLossType {
    inner: LossType,
}

#[pymethods]
impl PyLossType {
    #[staticmethod]
    pub fn mse() -> Self {
        Self { inner: LossType::MSE }
    }

    #[staticmethod]
    pub fn cross_entropy() -> Self {
        Self { inner: LossType::CrossEntropy }
    }

    #[staticmethod]
    pub fn from_str(s: &str) -> PyResult<Self> {
        let inner = match s.to_lowercase().as_str() {
            "mse" => LossType::MSE,
            "crossentropy" | "cross_entropy" => LossType::CrossEntropy,
            _ => return Err(PyValueError::new_err(format!("Unknown loss type: {}", s))),
        };
        Ok(Self { inner })
    }

    pub fn __str__(&self) -> &'static str {
        match self.inner {
            LossType::MSE => "mse",
            LossType::CrossEntropy => "crossentropy",
        }
    }

    pub fn __repr__(&self) -> String {
        format!("LossType.{}", self.__str__())
    }
}

/// Python wrapper for BatchNormParams.
#[pyclass(name = "BatchNormParams")]
#[derive(Clone)]
pub struct PyBatchNormParams {
    inner: RustBatchNormParams,
}

#[pymethods]
impl PyBatchNormParams {
    #[new]
    pub fn new() -> Self {
        Self { inner: RustBatchNormParams::new() }
    }

    pub fn initialize(&mut self, size: usize) {
        self.inner.initialize(size);
    }

    #[getter]
    pub fn gamma(&self) -> Vec<f64> {
        self.inner.gamma.clone()
    }

    #[setter]
    pub fn set_gamma(&mut self, gamma: Vec<f64>) {
        self.inner.gamma = gamma;
    }

    #[getter]
    pub fn beta(&self) -> Vec<f64> {
        self.inner.beta.clone()
    }

    #[setter]
    pub fn set_beta(&mut self, beta: Vec<f64>) {
        self.inner.beta = beta;
    }

    #[getter]
    pub fn running_mean(&self) -> Vec<f64> {
        self.inner.running_mean.clone()
    }

    #[setter]
    pub fn set_running_mean(&mut self, running_mean: Vec<f64>) {
        self.inner.running_mean = running_mean;
    }

    #[getter]
    pub fn running_var(&self) -> Vec<f64> {
        self.inner.running_var.clone()
    }

    #[setter]
    pub fn set_running_var(&mut self, running_var: Vec<f64>) {
        self.inner.running_var = running_var;
    }

    #[getter]
    pub fn epsilon(&self) -> f64 {
        self.inner.epsilon
    }

    #[setter]
    pub fn set_epsilon(&mut self, epsilon: f64) {
        self.inner.epsilon = epsilon;
    }

    #[getter]
    pub fn momentum(&self) -> f64 {
        self.inner.momentum
    }

    #[setter]
    pub fn set_momentum(&mut self, momentum: f64) {
        self.inner.momentum = momentum;
    }
}

/// CUDA-accelerated Convolutional Neural Network.
///
/// This class provides a complete CNN implementation with:
/// - Configurable convolutional, pooling, and fully-connected layers
/// - Adam optimizer for training
/// - Batch normalization support
/// - Model serialization to JSON and ONNX formats
///
/// Example:
///     >>> from facaded_cnn_cuda import CNN, ActivationType, LossType
///     >>> cnn = CNN(
///     ...     input_width=28, input_height=28, input_channels=1,
///     ...     conv_filters=[32, 64], kernel_sizes=[3, 3], pool_sizes=[2, 2],
///     ...     fc_sizes=[128], output_size=10,
///     ...     hidden_activation=ActivationType.relu(),
///     ...     output_activation=ActivationType.linear(),
///     ...     loss_type=LossType.cross_entropy(),
///     ...     learning_rate=0.001, gradient_clip=5.0
///     ... )
///     >>> output = cnn.predict([0.0] * 784)
#[pyclass(name = "CNN")]
pub struct PyCNN {
    inner: Mutex<RustCNN>,
}

#[pymethods]
impl PyCNN {
    /// Creates a new CNN with the specified architecture.
    ///
    /// Args:
    ///     input_width: Width of input images
    ///     input_height: Height of input images
    ///     input_channels: Number of input channels (e.g., 1 for grayscale, 3 for RGB)
    ///     conv_filters: List of filter counts for each convolutional layer
    ///     kernel_sizes: List of kernel sizes for each convolutional layer
    ///     pool_sizes: List of pooling sizes for each pooling layer
    ///     fc_sizes: List of neuron counts for each fully-connected hidden layer
    ///     output_size: Number of output classes
    ///     hidden_activation: Activation function for hidden layers
    ///     output_activation: Activation function for output layer
    ///     loss_type: Loss function for training
    ///     learning_rate: Learning rate for Adam optimizer
    ///     gradient_clip: Gradient clipping threshold
    ///
    /// Returns:
    ///     A new CNN instance
    #[new]
    #[pyo3(signature = (
        input_width,
        input_height,
        input_channels,
        conv_filters,
        kernel_sizes,
        pool_sizes,
        fc_sizes,
        output_size,
        hidden_activation = None,
        output_activation = None,
        loss_type = None,
        learning_rate = 0.001,
        gradient_clip = 5.0
    ))]
    pub fn new(
        input_width: i32,
        input_height: i32,
        input_channels: i32,
        conv_filters: Vec<i32>,
        kernel_sizes: Vec<i32>,
        pool_sizes: Vec<i32>,
        fc_sizes: Vec<i32>,
        output_size: i32,
        hidden_activation: Option<PyActivationType>,
        output_activation: Option<PyActivationType>,
        loss_type: Option<PyLossType>,
        learning_rate: f64,
        gradient_clip: f64,
    ) -> PyResult<Self> {
        let hidden_act = hidden_activation
            .map(|a| a.inner)
            .unwrap_or(ActivationType::ReLU);
        let output_act = output_activation
            .map(|a| a.inner)
            .unwrap_or(ActivationType::Linear);
        let loss = loss_type
            .map(|l| l.inner)
            .unwrap_or(LossType::CrossEntropy);

        let cnn = RustCNN::new(
            input_width,
            input_height,
            input_channels,
            &conv_filters,
            &kernel_sizes,
            &pool_sizes,
            &fc_sizes,
            output_size,
            hidden_act,
            output_act,
            loss,
            learning_rate,
            gradient_clip,
        ).map_err(|e| PyValueError::new_err(format!("Failed to create CNN: {}", e)))?;

        Ok(Self { inner: Mutex::new(cnn) })
    }

    /// Performs inference on an input image.
    ///
    /// Args:
    ///     image_data: Flattened input image data (size: width * height * channels)
    ///
    /// Returns:
    ///     List of softmax probabilities for each output class
    pub fn predict(&self, image_data: Vec<f64>) -> PyResult<Vec<f64>> {
        let mut cnn = self.inner.lock().map_err(|e| {
            PyValueError::new_err(format!("Failed to acquire lock: {}", e))
        })?;
        cnn.predict(&image_data)
            .map_err(|e| PyValueError::new_err(format!("Prediction failed: {}", e)))
    }

    /// Performs a single training step with one sample.
    ///
    /// Args:
    ///     image_data: Flattened input image data
    ///     target: One-hot encoded target labels
    ///
    /// Returns:
    ///     The cross-entropy loss for this sample
    pub fn train_step(&self, image_data: Vec<f64>, target: Vec<f64>) -> PyResult<f64> {
        let mut cnn = self.inner.lock().map_err(|e| {
            PyValueError::new_err(format!("Failed to acquire lock: {}", e))
        })?;
        cnn.train_step(&image_data, &target)
            .map_err(|e| PyValueError::new_err(format!("Training step failed: {}", e)))
    }

    /// Saves the model to a JSON file.
    ///
    /// Args:
    ///     filename: Path to the output JSON file
    pub fn save_to_json(&self, filename: &str) -> PyResult<()> {
        let cnn = self.inner.lock().map_err(|e| {
            PyValueError::new_err(format!("Failed to acquire lock: {}", e))
        })?;
        cnn.save_to_json(filename)
            .map_err(|e| PyValueError::new_err(format!("Failed to save model: {}", e)))
    }

    /// Loads a model from a JSON file.
    ///
    /// Args:
    ///     filename: Path to the JSON file
    ///
    /// Returns:
    ///     A new CNN instance with loaded weights
    #[staticmethod]
    pub fn load_from_json(filename: &str) -> PyResult<Self> {
        let cnn = RustCNN::load_from_json(filename)
            .map_err(|e| PyValueError::new_err(format!("Failed to load model: {}", e)))?;
        Ok(Self { inner: Mutex::new(cnn) })
    }

    /// Exports the model to ONNX binary format.
    ///
    /// Args:
    ///     filename: Path to the output ONNX file
    pub fn export_to_onnx(&self, filename: &str) -> PyResult<()> {
        let cnn = self.inner.lock().map_err(|e| {
            PyValueError::new_err(format!("Failed to acquire lock: {}", e))
        })?;
        cnn.export_to_onnx(filename)
            .map_err(|e| PyValueError::new_err(format!("Failed to export model: {}", e)))
    }

    /// Imports a model from ONNX binary format.
    ///
    /// Args:
    ///     filename: Path to the ONNX file
    ///
    /// Returns:
    ///     A new CNN instance with loaded weights
    #[staticmethod]
    pub fn import_from_onnx(filename: &str) -> PyResult<Self> {
        let cnn = RustCNN::import_from_onnx(filename)
            .map_err(|e| PyValueError::new_err(format!("Failed to import model: {}", e)))?;
        Ok(Self { inner: Mutex::new(cnn) })
    }

    /// Returns the input width.
    #[getter]
    pub fn input_width(&self) -> PyResult<i32> {
        let cnn = self.inner.lock().map_err(|e| {
            PyValueError::new_err(format!("Failed to acquire lock: {}", e))
        })?;
        Ok(cnn.get_input_width())
    }

    /// Returns the input height.
    #[getter]
    pub fn input_height(&self) -> PyResult<i32> {
        let cnn = self.inner.lock().map_err(|e| {
            PyValueError::new_err(format!("Failed to acquire lock: {}", e))
        })?;
        Ok(cnn.get_input_height())
    }

    /// Returns the number of input channels.
    #[getter]
    pub fn input_channels(&self) -> PyResult<i32> {
        let cnn = self.inner.lock().map_err(|e| {
            PyValueError::new_err(format!("Failed to acquire lock: {}", e))
        })?;
        Ok(cnn.get_input_channels())
    }

    /// Returns the output size (number of classes).
    #[getter]
    pub fn output_size(&self) -> PyResult<i32> {
        let cnn = self.inner.lock().map_err(|e| {
            PyValueError::new_err(format!("Failed to acquire lock: {}", e))
        })?;
        Ok(cnn.get_output_size())
    }

    /// Returns the learning rate.
    #[getter]
    pub fn learning_rate(&self) -> PyResult<f64> {
        let cnn = self.inner.lock().map_err(|e| {
            PyValueError::new_err(format!("Failed to acquire lock: {}", e))
        })?;
        Ok(cnn.get_learning_rate())
    }

    /// Sets the learning rate.
    #[setter]
    pub fn set_learning_rate(&self, lr: f64) -> PyResult<()> {
        let mut cnn = self.inner.lock().map_err(|e| {
            PyValueError::new_err(format!("Failed to acquire lock: {}", e))
        })?;
        cnn.set_learning_rate(lr);
        Ok(())
    }

    /// Returns the gradient clipping threshold.
    #[getter]
    pub fn gradient_clip(&self) -> PyResult<f64> {
        let cnn = self.inner.lock().map_err(|e| {
            PyValueError::new_err(format!("Failed to acquire lock: {}", e))
        })?;
        Ok(cnn.get_gradient_clip())
    }

    /// Sets the gradient clipping threshold.
    #[setter]
    pub fn set_gradient_clip(&self, clip: f64) -> PyResult<()> {
        let mut cnn = self.inner.lock().map_err(|e| {
            PyValueError::new_err(format!("Failed to acquire lock: {}", e))
        })?;
        cnn.set_gradient_clip(clip);
        Ok(())
    }

    /// Sets the dropout rate.
    pub fn set_dropout_rate(&self, rate: f64) -> PyResult<()> {
        let mut cnn = self.inner.lock().map_err(|e| {
            PyValueError::new_err(format!("Failed to acquire lock: {}", e))
        })?;
        cnn.set_dropout_rate(rate);
        Ok(())
    }

    /// Returns the hidden activation type as a string.
    #[getter]
    pub fn hidden_activation(&self) -> PyResult<String> {
        let cnn = self.inner.lock().map_err(|e| {
            PyValueError::new_err(format!("Failed to acquire lock: {}", e))
        })?;
        Ok(crate::activation_to_str(cnn.get_hidden_activation()).to_string())
    }

    /// Returns the output activation type as a string.
    #[getter]
    pub fn output_activation(&self) -> PyResult<String> {
        let cnn = self.inner.lock().map_err(|e| {
            PyValueError::new_err(format!("Failed to acquire lock: {}", e))
        })?;
        Ok(crate::activation_to_str(cnn.get_output_activation()).to_string())
    }

    /// Returns the loss function type as a string.
    #[getter]
    pub fn loss_function(&self) -> PyResult<String> {
        let cnn = self.inner.lock().map_err(|e| {
            PyValueError::new_err(format!("Failed to acquire lock: {}", e))
        })?;
        Ok(crate::loss_to_str(cnn.get_loss_function()).to_string())
    }

    /// Returns the convolutional filter counts.
    #[getter]
    pub fn conv_filters(&self) -> PyResult<Vec<i32>> {
        let cnn = self.inner.lock().map_err(|e| {
            PyValueError::new_err(format!("Failed to acquire lock: {}", e))
        })?;
        Ok(cnn.get_conv_filters().to_vec())
    }

    /// Returns the kernel sizes.
    #[getter]
    pub fn kernel_sizes(&self) -> PyResult<Vec<i32>> {
        let cnn = self.inner.lock().map_err(|e| {
            PyValueError::new_err(format!("Failed to acquire lock: {}", e))
        })?;
        Ok(cnn.get_kernel_sizes().to_vec())
    }

    /// Returns the pool sizes.
    #[getter]
    pub fn pool_sizes(&self) -> PyResult<Vec<i32>> {
        let cnn = self.inner.lock().map_err(|e| {
            PyValueError::new_err(format!("Failed to acquire lock: {}", e))
        })?;
        Ok(cnn.get_pool_sizes().to_vec())
    }

    /// Returns the fully-connected layer sizes.
    #[getter]
    pub fn fc_sizes(&self) -> PyResult<Vec<i32>> {
        let cnn = self.inner.lock().map_err(|e| {
            PyValueError::new_err(format!("Failed to acquire lock: {}", e))
        })?;
        Ok(cnn.get_fc_sizes().to_vec())
    }

    /// Returns whether batch normalization is enabled.
    #[getter]
    pub fn uses_batch_norm(&self) -> PyResult<bool> {
        let cnn = self.inner.lock().map_err(|e| {
            PyValueError::new_err(format!("Failed to acquire lock: {}", e))
        })?;
        Ok(cnn.uses_batch_norm())
    }

    /// Initializes batch normalization for all convolutional layers.
    pub fn initialize_batch_norm(&self) -> PyResult<()> {
        let mut cnn = self.inner.lock().map_err(|e| {
            PyValueError::new_err(format!("Failed to acquire lock: {}", e))
        })?;
        cnn.initialize_batch_norm();
        Ok(())
    }

    /// Applies batch normalization to input data for a specific layer.
    ///
    /// Args:
    ///     input: Input data to normalize
    ///     layer_idx: Index of the layer
    ///     training: Whether the model is in training mode
    ///
    /// Returns:
    ///     Normalized output data
    pub fn apply_batch_norm(&self, input: Vec<f64>, layer_idx: usize, training: bool) -> PyResult<Vec<f64>> {
        let cnn = self.inner.lock().map_err(|e| {
            PyValueError::new_err(format!("Failed to acquire lock: {}", e))
        })?;
        Ok(cnn.apply_batch_norm(&input, layer_idx, training))
    }
}

/// Python module for the CUDA CNN library.
#[pymodule]
pub fn facaded_cnn_cuda(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyActivationType>()?;
    m.add_class::<PyLossType>()?;
    m.add_class::<PyBatchNormParams>()?;
    m.add_class::<PyCNN>()?;
    Ok(())
}

