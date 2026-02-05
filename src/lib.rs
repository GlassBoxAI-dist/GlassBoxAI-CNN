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

//! # GPU-Accelerated Convolutional Neural Network Library
//!
//! A high-performance CNN implementation with CUDA and OpenCL acceleration.
//!
//! ## Rust Example
//!
//! ```rust,no_run
//! use facaded_cnn_cuda::{ConvolutionalNeuralNetworkCUDA, ActivationType, LossType};
//!
//! // Create a CNN for MNIST-like image classification
//! let mut cnn = ConvolutionalNeuralNetworkCUDA::new(
//!     28, 28, 1,                    // Input: 28x28 grayscale
//!     &[32, 64],                    // Conv filters per layer
//!     &[3, 3],                      // Kernel sizes
//!     &[2, 2],                      // Pool sizes
//!     &[128],                       // FC hidden layer sizes
//!     10,                           // Output classes
//!     ActivationType::ReLU,         // Hidden activation
//!     ActivationType::Linear,       // Output activation
//!     LossType::CrossEntropy,       // Loss function
//!     0.001,                        // Learning rate
//!     5.0,                          // Gradient clipping
//! ).expect("Failed to create CNN");
//!
//! // Make a prediction
//! let input = vec![0.0; 28 * 28];
//! let output = cnn.predict(&input).expect("Prediction failed");
//! ```
//!
//! ## Python Example
//!
//! ```python
//! from facaded_cnn_cuda import CNN, ActivationType, LossType
//!
//! # Create a CNN for MNIST-like image classification
//! cnn = CNN(
//!     input_width=28, input_height=28, input_channels=1,
//!     conv_filters=[32, 64], kernel_sizes=[3, 3], pool_sizes=[2, 2],
//!     fc_sizes=[128], output_size=10,
//!     hidden_activation=ActivationType.relu(),
//!     output_activation=ActivationType.linear(),
//!     loss_type=LossType.cross_entropy(),
//!     learning_rate=0.001, gradient_clip=5.0
//! )
//!
//! # Make a prediction
//! output = cnn.predict([0.0] * 784)
//! ```

// Kani verification tests (CISA Hardening)
#[cfg(kani)]
mod kani_tests;

pub mod cnn;

#[cfg(feature = "python")]
mod python;

#[cfg(feature = "python")]
pub use python::facaded_cnn_cuda;

#[cfg(feature = "nodejs")]
mod nodejs;

#[cfg(feature = "capi")]
pub mod capi;

#[cfg(feature = "opencl")]
pub mod opencl;

#[cfg(feature = "opencl")]
pub use opencl::ConvolutionalNeuralNetworkOpenCL;

pub use cnn::{
    ActivationType,
    BatchNormParams,
    Command,
    ConvolutionalNeuralNetworkCUDA,
    LossType,
};

pub use cnn::{
    activation_to_str,
    loss_to_str,
    parse_activation,
    parse_command,
    parse_loss,
};
