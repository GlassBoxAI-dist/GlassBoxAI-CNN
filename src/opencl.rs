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

//! OpenCL backend for Convolutional Neural Network.
//!
//! This module provides an OpenCL implementation equivalent to the CUDA version,
//! enabling GPU acceleration on AMD, Intel, and other OpenCL-compatible devices.

use ocl::{Buffer, Context, Device, Kernel, Platform, Program, Queue};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};

const WORKGROUP_SIZE: usize = 256;

/// OpenCL kernel source code (equivalent to CUDA kernels)
const OPENCL_KERNEL_SRC: &str = r#"
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

double relu(double x) {
    return (x > 0.0) ? x : 0.0;
}

double relu_derivative(double x) {
    return (x > 0.0) ? 1.0 : 0.0;
}

double clamp_val(double x, double min_val, double max_val) {
    if (x < min_val) return min_val;
    if (x > max_val) return max_val;
    return x;
}

double clip_grad(double x) {
    if (!isfinite(x)) return 0.0;
    return clamp_val(x, -1.0, 1.0);
}

__kernel void conv_forward_kernel(
    __global double* output,
    __global double* pre_activation,
    __global const double* input,
    __global const double* weights,
    __global const double* biases,
    int input_channels,
    int kernel_size,
    int input_h,
    int input_w,
    int output_h,
    int output_w,
    int stride,
    int padding,
    int num_filters
) {
    int idx = get_global_id(0);
    int total = num_filters * output_h * output_w;
    if (idx >= total) return;

    int f = idx / (output_h * output_w);
    int rem = idx % (output_h * output_w);
    int oh = rem / output_w;
    int ow = rem % output_w;

    double sum = biases[f];

    for (int c = 0; c < input_channels; c++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int ih = oh * stride + kh;
                int iw = ow * stride + kw;
                int padded_h = input_h + 2 * padding;
                int padded_w = input_w + 2 * padding;
                int input_idx = c * padded_h * padded_w + ih * padded_w + iw;
                int weight_idx = f * input_channels * kernel_size * kernel_size +
                               c * kernel_size * kernel_size + kh * kernel_size + kw;
                sum += input[input_idx] * weights[weight_idx];
            }
        }
    }

    if (!isfinite(sum)) sum = 0.0;

    int out_idx = f * output_h * output_w + oh * output_w + ow;
    pre_activation[out_idx] = sum;
    output[out_idx] = relu(sum);
}

__kernel void pool_forward_kernel(
    __global double* output,
    __global int* max_indices_y,
    __global int* max_indices_x,
    __global const double* input,
    int channels,
    int input_h,
    int input_w,
    int output_h,
    int output_w,
    int pool_size
) {
    int idx = get_global_id(0);
    int total = channels * output_h * output_w;
    if (idx >= total) return;

    int c = idx / (output_h * output_w);
    int rem = idx % (output_h * output_w);
    int oh = rem / output_w;
    int ow = rem % output_w;

    double max_val = -1e308;
    int max_ph = 0, max_pw = 0;

    for (int ph = 0; ph < pool_size; ph++) {
        for (int pw = 0; pw < pool_size; pw++) {
            int ih = oh * pool_size + ph;
            int iw = ow * pool_size + pw;
            int input_idx = c * input_h * input_w + ih * input_w + iw;
            double val = input[input_idx];
            if (val > max_val) {
                max_val = val;
                max_ph = ph;
                max_pw = pw;
            }
        }
    }

    int out_idx = c * output_h * output_w + oh * output_w + ow;
    output[out_idx] = max_val;
    max_indices_y[out_idx] = max_ph;
    max_indices_x[out_idx] = max_pw;
}

__kernel void fc_forward_kernel(
    __global double* output,
    __global double* pre_activation,
    __global const double* input,
    __global const double* weights,
    __global const double* biases,
    __global const double* dropout_mask,
    int num_neurons,
    int num_inputs,
    int apply_relu
) {
    int i = get_global_id(0);
    if (i >= num_neurons) return;

    double sum = biases[i];
    for (int j = 0; j < num_inputs; j++) {
        sum += input[j] * weights[i * num_inputs + j];
    }

    if (!isfinite(sum)) sum = 0.0;

    pre_activation[i] = sum;
    if (apply_relu)
        output[i] = relu(sum) * dropout_mask[i];
    else
        output[i] = sum;
}

__kernel void softmax_kernel(
    __global double* output,
    __global const double* input,
    int n,
    double max_val,
    double sum_exp
) {
    int i = get_global_id(0);
    if (i >= n) return;

    double val = exp(input[i] - max_val) / sum_exp;
    if (val < 1e-15) val = 1e-15;
    if (val > 1.0 - 1e-15) val = 1.0 - 1e-15;
    output[i] = val;
}

__kernel void fc_backward_kernel(
    __global double* errors,
    __global const double* grad,
    __global const double* weights,
    __global const double* pre_activation,
    __global const double* dropout_mask,
    int num_neurons,
    int num_inputs,
    int is_output_layer
) {
    int i = get_global_id(0);
    if (i >= num_neurons) return;

    double delta;
    if (is_output_layer) {
        delta = grad[i];
    } else {
        delta = grad[i] * relu_derivative(pre_activation[i]) * dropout_mask[i];
    }
    errors[i] = delta;
}

__kernel void fc_input_grad_kernel(
    __global double* input_grad,
    __global const double* errors,
    __global const double* weights,
    int num_neurons,
    int num_inputs
) {
    int j = get_global_id(0);
    if (j >= num_inputs) return;

    double sum = 0.0;
    for (int i = 0; i < num_neurons; i++) {
        sum += errors[i] * weights[i * num_inputs + j];
    }
    input_grad[j] = sum;
}

__kernel void pool_backward_kernel(
    __global double* input_grad,
    __global const double* grad,
    __global const int* max_indices_y,
    __global const int* max_indices_x,
    int channels,
    int input_h,
    int input_w,
    int output_h,
    int output_w,
    int pool_size
) {
    int idx = get_global_id(0);
    int total = channels * output_h * output_w;
    if (idx >= total) return;

    int c = idx / (output_h * output_w);
    int rem = idx % (output_h * output_w);
    int oh = rem / output_w;
    int ow = rem % output_w;

    int out_idx = c * output_h * output_w + oh * output_w + ow;
    int src_h = oh * pool_size + max_indices_y[out_idx];
    int src_w = ow * pool_size + max_indices_x[out_idx];
    int input_idx = c * input_h * input_w + src_h * input_w + src_w;

    // Note: This is not atomic, may need serialization for overlapping gradients
    input_grad[input_idx] += grad[out_idx];
}

__kernel void conv_weight_grad_kernel(
    __global double* weight_grads,
    __global const double* grad_with_relu,
    __global const double* padded_input,
    int num_filters,
    int input_channels,
    int kernel_size,
    int output_h,
    int output_w,
    int padded_h,
    int padded_w,
    int stride
) {
    int idx = get_global_id(0);
    int total_weights = num_filters * input_channels * kernel_size * kernel_size;
    if (idx >= total_weights) return;

    int f = idx / (input_channels * kernel_size * kernel_size);
    int rem = idx % (input_channels * kernel_size * kernel_size);
    int c = rem / (kernel_size * kernel_size);
    rem = rem % (kernel_size * kernel_size);
    int kh = rem / kernel_size;
    int kw = rem % kernel_size;

    double w_grad = 0.0;
    for (int h = 0; h < output_h; h++) {
        for (int w = 0; w < output_w; w++) {
            int in_h = h * stride + kh;
            int in_w = w * stride + kw;
            int grad_idx = f * output_h * output_w + h * output_w + w;
            int input_idx = c * padded_h * padded_w + in_h * padded_w + in_w;
            w_grad += grad_with_relu[grad_idx] * padded_input[input_idx];
        }
    }
    weight_grads[idx] = w_grad;
}

__kernel void conv_bias_grad_kernel(
    __global double* bias_grads,
    __global const double* grad_with_relu,
    int num_filters,
    int output_h,
    int output_w
) {
    int f = get_global_id(0);
    if (f >= num_filters) return;

    double sum = 0.0;
    for (int h = 0; h < output_h; h++) {
        for (int w = 0; w < output_w; w++) {
            sum += grad_with_relu[f * output_h * output_w + h * output_w + w];
        }
    }
    bias_grads[f] = sum;
}

__kernel void apply_relu_deriv_kernel(
    __global double* grad_with_relu,
    __global const double* grad,
    __global const double* pre_activation,
    int n
) {
    int i = get_global_id(0);
    if (i >= n) return;
    grad_with_relu[i] = grad[i] * relu_derivative(pre_activation[i]);
}

__kernel void conv_input_grad_kernel(
    __global double* input_grad,
    __global const double* grad_with_relu,
    __global const double* weights,
    int num_filters,
    int input_channels,
    int kernel_size,
    int input_h,
    int input_w,
    int output_h,
    int output_w,
    int stride,
    int padding
) {
    int idx = get_global_id(0);
    int total = input_channels * input_h * input_w;
    if (idx >= total) return;

    int c = idx / (input_h * input_w);
    int rem = idx % (input_h * input_w);
    int ih = rem / input_w;
    int iw = rem % input_w;

    double sum = 0.0;
    for (int f = 0; f < num_filters; f++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int oh = ih + padding - kh;
                int ow = iw + padding - kw;
                if (oh >= 0 && oh < output_h && ow >= 0 && ow < output_w &&
                    oh % stride == 0 && ow % stride == 0) {
                    oh /= stride;
                    ow /= stride;
                    int grad_idx = f * output_h * output_w + oh * output_w + ow;
                    int weight_idx = f * input_channels * kernel_size * kernel_size +
                                   c * kernel_size * kernel_size + kh * kernel_size + kw;
                    sum += grad_with_relu[grad_idx] * weights[weight_idx];
                }
            }
        }
    }
    input_grad[idx] = sum;
}

__kernel void adam_update_kernel(
    __global double* weights,
    __global double* m,
    __global double* v,
    __global const double* grads,
    double learning_rate,
    double beta1,
    double beta2,
    int timestep,
    int n
) {
    int i = get_global_id(0);
    if (i >= n) return;

    double grad = clip_grad(grads[i]);
    m[i] = beta1 * m[i] + (1.0 - beta1) * grad;
    v[i] = beta2 * v[i] + (1.0 - beta2) * grad * grad;

    double m_hat = m[i] / (1.0 - pow(beta1, (double)timestep));
    double v_hat = v[i] / (1.0 - pow(beta2, (double)timestep));
    double update = learning_rate * m_hat / (sqrt(v_hat) + 1e-8);

    if (isfinite(update))
        weights[i] -= update;
}

__kernel void zero_array_kernel(__global double* arr, int n) {
    int i = get_global_id(0);
    if (i < n) arr[i] = 0.0;
}

__kernel void pad_input_kernel(
    __global double* padded,
    __global const double* input,
    int channels,
    int height,
    int width,
    int padding
) {
    int idx = get_global_id(0);
    int padded_h = height + 2 * padding;
    int padded_w = width + 2 * padding;
    int total = channels * padded_h * padded_w;
    if (idx >= total) return;

    int c = idx / (padded_h * padded_w);
    int rem = idx % (padded_h * padded_w);
    int ph = rem / padded_w;
    int pw = rem % padded_w;

    int src_h = ph - padding;
    int src_w = pw - padding;

    if (src_h >= 0 && src_h < height && src_w >= 0 && src_w < width) {
        padded[idx] = input[c * height * width + src_h * width + src_w];
    } else {
        padded[idx] = 0.0;
    }
}
"#;

/// Activation function types for neural network layers.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum ActivationType {
    Sigmoid,
    Tanh,
    ReLU,
    Linear,
}

/// Loss function types for training.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum LossType {
    MSE,
    CrossEntropy,
}

/// Batch normalization parameters for a layer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchNormParams {
    pub gamma: Vec<f64>,
    pub beta: Vec<f64>,
    pub running_mean: Vec<f64>,
    pub running_var: Vec<f64>,
    pub epsilon: f64,
    pub momentum: f64,
}

impl BatchNormParams {
    pub fn new() -> Self {
        Self {
            gamma: Vec::new(),
            beta: Vec::new(),
            running_mean: Vec::new(),
            running_var: Vec::new(),
            epsilon: 1e-5,
            momentum: 0.1,
        }
    }

    pub fn initialize(&mut self, size: usize) {
        self.gamma = vec![1.0; size];
        self.beta = vec![0.0; size];
        self.running_mean = vec![0.0; size];
        self.running_var = vec![1.0; size];
    }
}

impl Default for BatchNormParams {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(dead_code)]
struct ConvLayerCL {
    weights: Buffer<f64>,
    biases: Buffer<f64>,
    weights_m: Buffer<f64>,
    weights_v: Buffer<f64>,
    bias_m: Buffer<f64>,
    bias_v: Buffer<f64>,
    weight_grads: Buffer<f64>,
    bias_grads: Buffer<f64>,
    output: Buffer<f64>,
    pre_activation: Buffer<f64>,
    padded_input: Buffer<f64>,
    num_filters: i32,
    input_channels: i32,
    kernel_size: i32,
    stride: i32,
    padding: i32,
    output_h: i32,
    output_w: i32,
}

struct PoolLayerCL {
    output: Buffer<f64>,
    max_indices_y: Buffer<i32>,
    max_indices_x: Buffer<i32>,
    pool_size: i32,
    #[allow(dead_code)]
    stride: i32,
    output_h: i32,
    output_w: i32,
}

#[allow(dead_code)]
struct FCLayerCL {
    weights: Buffer<f64>,
    biases: Buffer<f64>,
    weights_m: Buffer<f64>,
    weights_v: Buffer<f64>,
    bias_m: Buffer<f64>,
    bias_v: Buffer<f64>,
    output: Buffer<f64>,
    pre_activation: Buffer<f64>,
    errors: Buffer<f64>,
    dropout_mask: Buffer<f64>,
    num_neurons: i32,
    num_inputs: i32,
}

#[derive(Serialize, Deserialize)]
struct ModelJson {
    input_width: i32,
    input_height: i32,
    input_channels: i32,
    output_size: i32,
    conv_filters: Vec<i32>,
    kernel_sizes: Vec<i32>,
    pool_sizes: Vec<i32>,
    fc_layer_sizes: Vec<i32>,
    learning_rate: f64,
    dropout_rate: f64,
    activation: String,
    output_activation: String,
    loss_type: String,
    gradient_clip: f64,
    conv_layers: Vec<ConvLayerJson>,
    pool_layers: Vec<PoolLayerJson>,
    fc_layers: Vec<FCLayerJson>,
    output_layer: FCLayerJson,
}

#[derive(Serialize, Deserialize)]
struct ConvLayerJson {
    filters: Vec<FilterJson>,
}

#[derive(Serialize, Deserialize)]
struct FilterJson {
    bias: f64,
    weights: Vec<Vec<Vec<f64>>>,
}

#[derive(Serialize, Deserialize)]
struct PoolLayerJson {
    pool_size: i32,
}

#[derive(Serialize, Deserialize)]
struct FCLayerJson {
    neurons: Vec<NeuronJson>,
}

#[derive(Serialize, Deserialize)]
struct NeuronJson {
    bias: f64,
    weights: Vec<f64>,
}

/// Helper functions for activation types
pub fn activation_to_str(act: ActivationType) -> &'static str {
    match act {
        ActivationType::Sigmoid => "sigmoid",
        ActivationType::Tanh => "tanh",
        ActivationType::ReLU => "relu",
        ActivationType::Linear => "linear",
    }
}

pub fn parse_activation(s: &str) -> ActivationType {
    match s.to_lowercase().as_str() {
        "sigmoid" => ActivationType::Sigmoid,
        "tanh" => ActivationType::Tanh,
        "relu" => ActivationType::ReLU,
        "linear" => ActivationType::Linear,
        _ => ActivationType::ReLU,
    }
}

pub fn loss_to_str(loss: LossType) -> &'static str {
    match loss {
        LossType::MSE => "mse",
        LossType::CrossEntropy => "crossentropy",
    }
}

pub fn parse_loss(s: &str) -> LossType {
    match s.to_lowercase().as_str() {
        "mse" => LossType::MSE,
        "crossentropy" | "cross_entropy" => LossType::CrossEntropy,
        _ => LossType::MSE,
    }
}

fn global_size(n: usize) -> usize {
    ((n + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE) * WORKGROUP_SIZE
}

/// OpenCL-accelerated Convolutional Neural Network.
///
/// This struct provides the same API as the CUDA version but uses OpenCL
/// for cross-platform GPU acceleration (AMD, Intel, NVIDIA).
#[allow(dead_code)]
pub struct ConvolutionalNeuralNetworkOpenCL {
    context: Context,
    queue: Queue,
    program: Program,
    learning_rate: f64,
    dropout_rate: f64,
    gradient_clip: f64,
    beta1: f64,
    beta2: f64,
    adam_t: i32,
    is_training: bool,
    hidden_activation: ActivationType,
    output_activation: ActivationType,
    loss_function: LossType,
    conv_layers: Vec<ConvLayerCL>,
    pool_layers: Vec<PoolLayerCL>,
    fc_layers: Vec<FCLayerCL>,
    output_layer: Option<FCLayerCL>,
    input_width: i32,
    input_height: i32,
    input_channels: i32,
    flattened_size: i32,
    last_conv_h: i32,
    last_conv_w: i32,
    last_conv_c: i32,
    output_size: i32,
    flattened_features: Buffer<f64>,
    conv_grad: Buffer<f64>,
    fc_grad: Buffer<f64>,
    logits: Buffer<f64>,
    softmax_output: Buffer<f64>,
    max_neurons: i32,
    f_conv_filters: Vec<i32>,
    f_kernel_sizes: Vec<i32>,
    f_pool_sizes: Vec<i32>,
    f_fc_sizes: Vec<i32>,
    use_batch_norm: bool,
    batch_norm_params: Vec<BatchNormParams>,
}

impl ConvolutionalNeuralNetworkOpenCL {
    /// Creates a new CNN with OpenCL acceleration.
    ///
    /// # Arguments
    ///
    /// * `input_width` - Width of input images
    /// * `input_height` - Height of input images
    /// * `input_channels` - Number of input channels (1 for grayscale, 3 for RGB)
    /// * `conv_filters` - Number of filters in each convolutional layer
    /// * `kernel_sizes` - Kernel size for each convolutional layer
    /// * `pool_sizes` - Pool size for each pooling layer
    /// * `fc_sizes` - Number of neurons in each fully connected hidden layer
    /// * `output_size` - Number of output classes
    /// * `hidden_act` - Activation function for hidden layers
    /// * `output_act` - Activation function for output layer
    /// * `loss_type` - Loss function for training
    /// * `learning_rate` - Learning rate for Adam optimizer
    /// * `gradient_clip` - Gradient clipping threshold
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input_width: i32,
        input_height: i32,
        input_channels: i32,
        conv_filters: &[i32],
        kernel_sizes: &[i32],
        pool_sizes: &[i32],
        fc_sizes: &[i32],
        output_size: i32,
        hidden_act: ActivationType,
        output_act: ActivationType,
        loss_type: LossType,
        learning_rate: f64,
        gradient_clip: f64,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut rng = rand::thread_rng();

        // Initialize OpenCL
        let platform = Platform::default();
        let device = Device::first(platform)?;
        let context = Context::builder()
            .platform(platform)
            .devices(device)
            .build()?;
        let queue = Queue::new(&context, device, None)?;

        // Compile OpenCL program
        let program = Program::builder()
            .src(OPENCL_KERNEL_SRC)
            .devices(device)
            .build(&context)?;

        let mut current_w = input_width;
        let mut current_h = input_height;
        let mut current_c = input_channels;

        let mut conv_layers_cl = Vec::new();
        let mut pool_layers_cl = Vec::new();

        for i in 0..conv_filters.len() {
            let kernel_padding = kernel_sizes[i] / 2;
            let output_w = (current_w - kernel_sizes[i] + 2 * kernel_padding) / 1 + 1;
            let output_h = (current_h - kernel_sizes[i] + 2 * kernel_padding) / 1 + 1;

            let weight_size = (conv_filters[i] * current_c * kernel_sizes[i] * kernel_sizes[i]) as usize;
            let output_sz = (conv_filters[i] * output_h * output_w) as usize;
            let padded_h = current_h + 2 * kernel_padding;
            let padded_w = current_w + 2 * kernel_padding;
            let padded_size = (current_c * padded_h * padded_w) as usize;

            // He initialization
            let scale = (2.0 / (current_c * kernel_sizes[i] * kernel_sizes[i]) as f64).sqrt();
            let weights: Vec<f64> = (0..weight_size)
                .map(|_| (rng.gen::<f64>() - 0.5) * scale)
                .collect();

            let conv = ConvLayerCL {
                weights: Buffer::builder()
                    .queue(queue.clone())
                    .len(weight_size)
                    .copy_host_slice(&weights)
                    .build()?,
                biases: Buffer::builder()
                    .queue(queue.clone())
                    .len(conv_filters[i] as usize)
                    .fill_val(0.0f64)
                    .build()?,
                weights_m: Buffer::builder()
                    .queue(queue.clone())
                    .len(weight_size)
                    .fill_val(0.0f64)
                    .build()?,
                weights_v: Buffer::builder()
                    .queue(queue.clone())
                    .len(weight_size)
                    .fill_val(0.0f64)
                    .build()?,
                bias_m: Buffer::builder()
                    .queue(queue.clone())
                    .len(conv_filters[i] as usize)
                    .fill_val(0.0f64)
                    .build()?,
                bias_v: Buffer::builder()
                    .queue(queue.clone())
                    .len(conv_filters[i] as usize)
                    .fill_val(0.0f64)
                    .build()?,
                weight_grads: Buffer::builder()
                    .queue(queue.clone())
                    .len(weight_size)
                    .fill_val(0.0f64)
                    .build()?,
                bias_grads: Buffer::builder()
                    .queue(queue.clone())
                    .len(conv_filters[i] as usize)
                    .fill_val(0.0f64)
                    .build()?,
                output: Buffer::builder()
                    .queue(queue.clone())
                    .len(output_sz)
                    .fill_val(0.0f64)
                    .build()?,
                pre_activation: Buffer::builder()
                    .queue(queue.clone())
                    .len(output_sz)
                    .fill_val(0.0f64)
                    .build()?,
                padded_input: Buffer::builder()
                    .queue(queue.clone())
                    .len(padded_size)
                    .fill_val(0.0f64)
                    .build()?,
                num_filters: conv_filters[i],
                input_channels: current_c,
                kernel_size: kernel_sizes[i],
                stride: 1,
                padding: kernel_padding,
                output_h,
                output_w,
            };
            conv_layers_cl.push(conv);

            current_w = output_w;
            current_h = output_h;
            current_c = conv_filters[i];

            if i < pool_sizes.len() {
                let pool_out_h = current_h / pool_sizes[i];
                let pool_out_w = current_w / pool_sizes[i];
                let pool_output_sz = (current_c * pool_out_h * pool_out_w) as usize;

                let pool = PoolLayerCL {
                    output: Buffer::builder()
                        .queue(queue.clone())
                        .len(pool_output_sz)
                        .fill_val(0.0f64)
                        .build()?,
                    max_indices_y: Buffer::builder()
                        .queue(queue.clone())
                        .len(pool_output_sz)
                        .fill_val(0i32)
                        .build()?,
                    max_indices_x: Buffer::builder()
                        .queue(queue.clone())
                        .len(pool_output_sz)
                        .fill_val(0i32)
                        .build()?,
                    pool_size: pool_sizes[i],
                    stride: pool_sizes[i],
                    output_h: pool_out_h,
                    output_w: pool_out_w,
                };
                pool_layers_cl.push(pool);

                current_w = pool_out_w;
                current_h = pool_out_h;
            }
        }

        let last_conv_h = current_h;
        let last_conv_w = current_w;
        let last_conv_c = current_c;
        let flattened_size = current_w * current_h * current_c;

        let mut fc_layers_cl = Vec::new();
        let mut num_inputs = flattened_size;

        for &fc_size in fc_sizes {
            let weight_size = (fc_size * num_inputs) as usize;
            let scale = (2.0 / num_inputs as f64).sqrt();
            let weights: Vec<f64> = (0..weight_size)
                .map(|_| (rng.gen::<f64>() - 0.5) * scale)
                .collect();
            let mask: Vec<f64> = vec![1.0; fc_size as usize];

            let fc = FCLayerCL {
                weights: Buffer::builder()
                    .queue(queue.clone())
                    .len(weight_size)
                    .copy_host_slice(&weights)
                    .build()?,
                biases: Buffer::builder()
                    .queue(queue.clone())
                    .len(fc_size as usize)
                    .fill_val(0.0f64)
                    .build()?,
                weights_m: Buffer::builder()
                    .queue(queue.clone())
                    .len(weight_size)
                    .fill_val(0.0f64)
                    .build()?,
                weights_v: Buffer::builder()
                    .queue(queue.clone())
                    .len(weight_size)
                    .fill_val(0.0f64)
                    .build()?,
                bias_m: Buffer::builder()
                    .queue(queue.clone())
                    .len(fc_size as usize)
                    .fill_val(0.0f64)
                    .build()?,
                bias_v: Buffer::builder()
                    .queue(queue.clone())
                    .len(fc_size as usize)
                    .fill_val(0.0f64)
                    .build()?,
                output: Buffer::builder()
                    .queue(queue.clone())
                    .len(fc_size as usize)
                    .fill_val(0.0f64)
                    .build()?,
                pre_activation: Buffer::builder()
                    .queue(queue.clone())
                    .len(fc_size as usize)
                    .fill_val(0.0f64)
                    .build()?,
                errors: Buffer::builder()
                    .queue(queue.clone())
                    .len(fc_size as usize)
                    .fill_val(0.0f64)
                    .build()?,
                dropout_mask: Buffer::builder()
                    .queue(queue.clone())
                    .len(fc_size as usize)
                    .copy_host_slice(&mask)
                    .build()?,
                num_neurons: fc_size,
                num_inputs,
            };
            fc_layers_cl.push(fc);
            num_inputs = fc_size;
        }

        // Output layer
        let out_weight_size = (output_size * num_inputs) as usize;
        let scale = (2.0 / num_inputs as f64).sqrt();
        let out_weights: Vec<f64> = (0..out_weight_size)
            .map(|_| (rng.gen::<f64>() - 0.5) * scale)
            .collect();
        let out_mask: Vec<f64> = vec![1.0; output_size as usize];

        let output_layer = FCLayerCL {
            weights: Buffer::builder()
                .queue(queue.clone())
                .len(out_weight_size)
                .copy_host_slice(&out_weights)
                .build()?,
            biases: Buffer::builder()
                .queue(queue.clone())
                .len(output_size as usize)
                .fill_val(0.0f64)
                .build()?,
            weights_m: Buffer::builder()
                .queue(queue.clone())
                .len(out_weight_size)
                .fill_val(0.0f64)
                .build()?,
            weights_v: Buffer::builder()
                .queue(queue.clone())
                .len(out_weight_size)
                .fill_val(0.0f64)
                .build()?,
            bias_m: Buffer::builder()
                .queue(queue.clone())
                .len(output_size as usize)
                .fill_val(0.0f64)
                .build()?,
            bias_v: Buffer::builder()
                .queue(queue.clone())
                .len(output_size as usize)
                .fill_val(0.0f64)
                .build()?,
            output: Buffer::builder()
                .queue(queue.clone())
                .len(output_size as usize)
                .fill_val(0.0f64)
                .build()?,
            pre_activation: Buffer::builder()
                .queue(queue.clone())
                .len(output_size as usize)
                .fill_val(0.0f64)
                .build()?,
            errors: Buffer::builder()
                .queue(queue.clone())
                .len(output_size as usize)
                .fill_val(0.0f64)
                .build()?,
            dropout_mask: Buffer::builder()
                .queue(queue.clone())
                .len(output_size as usize)
                .copy_host_slice(&out_mask)
                .build()?,
            num_neurons: output_size,
            num_inputs,
        };

        let mut max_neurons = flattened_size;
        for &fc_size in fc_sizes {
            if fc_size > max_neurons {
                max_neurons = fc_size;
            }
        }
        if output_size > max_neurons {
            max_neurons = output_size;
        }

        Ok(Self {
            context,
            queue: queue.clone(),
            program,
            learning_rate,
            dropout_rate: 0.0,
            gradient_clip,
            beta1: 0.9,
            beta2: 0.999,
            adam_t: 0,
            is_training: false,
            hidden_activation: hidden_act,
            output_activation: output_act,
            loss_function: loss_type,
            conv_layers: conv_layers_cl,
            pool_layers: pool_layers_cl,
            fc_layers: fc_layers_cl,
            output_layer: Some(output_layer),
            input_width,
            input_height,
            input_channels,
            flattened_size,
            last_conv_h,
            last_conv_w,
            last_conv_c,
            output_size,
            flattened_features: Buffer::builder()
                .queue(queue.clone())
                .len(flattened_size as usize)
                .fill_val(0.0f64)
                .build()?,
            conv_grad: Buffer::builder()
                .queue(queue.clone())
                .len(flattened_size as usize)
                .fill_val(0.0f64)
                .build()?,
            fc_grad: Buffer::builder()
                .queue(queue.clone())
                .len(max_neurons as usize)
                .fill_val(0.0f64)
                .build()?,
            logits: Buffer::builder()
                .queue(queue.clone())
                .len(output_size as usize)
                .fill_val(0.0f64)
                .build()?,
            softmax_output: Buffer::builder()
                .queue(queue)
                .len(output_size as usize)
                .fill_val(0.0f64)
                .build()?,
            max_neurons,
            f_conv_filters: conv_filters.to_vec(),
            f_kernel_sizes: kernel_sizes.to_vec(),
            f_pool_sizes: pool_sizes.to_vec(),
            f_fc_sizes: fc_sizes.to_vec(),
            use_batch_norm: false,
            batch_norm_params: Vec::new(),
        })
    }

    /// Performs inference on an input image.
    pub fn predict(&mut self, image_data: &[f64]) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        self.is_training = false;

        let input_buf: Buffer<f64> = Buffer::builder()
            .queue(self.queue.clone())
            .len(image_data.len())
            .copy_host_slice(image_data)
            .build()?;

        let mut current_h = self.input_height;
        let mut current_w = self.input_width;
        let mut current_c = self.input_channels;

        // Forward through conv layers
        for i in 0..self.conv_layers.len() {
            let conv = &self.conv_layers[i];
            let padded_h = current_h + 2 * conv.padding;
            let padded_w = current_w + 2 * conv.padding;
            let padded_size = (current_c * padded_h * padded_w) as usize;

            // Pad input
            let pad_kernel = Kernel::builder()
                .program(&self.program)
                .name("pad_input_kernel")
                .queue(self.queue.clone())
                .global_work_size(global_size(padded_size))
                .arg(&conv.padded_input)
                .arg(if i == 0 { &input_buf } else if i > 0 && i - 1 < self.pool_layers.len() {
                    &self.pool_layers[i - 1].output
                } else {
                    &self.conv_layers[i - 1].output
                })
                .arg(current_c)
                .arg(current_h)
                .arg(current_w)
                .arg(conv.padding)
                .build()?;
            unsafe { pad_kernel.enq()?; }

            // Conv forward
            let output_size = (conv.num_filters * conv.output_h * conv.output_w) as usize;
            let conv_kernel = Kernel::builder()
                .program(&self.program)
                .name("conv_forward_kernel")
                .queue(self.queue.clone())
                .global_work_size(global_size(output_size))
                .arg(&conv.output)
                .arg(&conv.pre_activation)
                .arg(&conv.padded_input)
                .arg(&conv.weights)
                .arg(&conv.biases)
                .arg(conv.input_channels)
                .arg(conv.kernel_size)
                .arg(current_h)
                .arg(current_w)
                .arg(conv.output_h)
                .arg(conv.output_w)
                .arg(conv.stride)
                .arg(conv.padding)
                .arg(conv.num_filters)
                .build()?;
            unsafe { conv_kernel.enq()?; }

            current_h = conv.output_h;
            current_w = conv.output_w;
            current_c = conv.num_filters;

            // Pooling
            if i < self.pool_layers.len() {
                let pool = &self.pool_layers[i];
                let pool_output_size = (current_c * pool.output_h * pool.output_w) as usize;

                let pool_kernel = Kernel::builder()
                    .program(&self.program)
                    .name("pool_forward_kernel")
                    .queue(self.queue.clone())
                    .global_work_size(global_size(pool_output_size))
                    .arg(&pool.output)
                    .arg(&pool.max_indices_y)
                    .arg(&pool.max_indices_x)
                    .arg(&self.conv_layers[i].output)
                    .arg(current_c)
                    .arg(current_h)
                    .arg(current_w)
                    .arg(pool.output_h)
                    .arg(pool.output_w)
                    .arg(pool.pool_size)
                    .build()?;
                unsafe { pool_kernel.enq()?; }

                current_h = pool.output_h;
                current_w = pool.output_w;
            }
        }

        // Copy to flattened features
        if let Some(last_pool) = self.pool_layers.last() {
            last_pool.output.copy(&mut self.flattened_features, None, None).enq()?;
        } else if let Some(last_conv) = self.conv_layers.last() {
            last_conv.output.copy(&mut self.flattened_features, None, None).enq()?;
        }

        // FC forward
        for i in 0..self.fc_layers.len() {
            let fc = &self.fc_layers[i];
            let fc_kernel = Kernel::builder()
                .program(&self.program)
                .name("fc_forward_kernel")
                .queue(self.queue.clone())
                .global_work_size(global_size(fc.num_neurons as usize))
                .arg(&fc.output)
                .arg(&fc.pre_activation)
                .arg(if i == 0 { &self.flattened_features } else { &self.fc_layers[i - 1].output })
                .arg(&fc.weights)
                .arg(&fc.biases)
                .arg(&fc.dropout_mask)
                .arg(fc.num_neurons)
                .arg(fc.num_inputs)
                .arg(1i32) // apply_relu
                .build()?;
            unsafe { fc_kernel.enq()?; }
        }

        // Output layer forward
        if let Some(ref output_layer) = self.output_layer {
            let input_for_output = if self.fc_layers.is_empty() {
                &self.flattened_features
            } else {
                &self.fc_layers.last().unwrap().output
            };

            let fc_kernel = Kernel::builder()
                .program(&self.program)
                .name("fc_forward_kernel")
                .queue(self.queue.clone())
                .global_work_size(global_size(output_layer.num_neurons as usize))
                .arg(&self.logits)
                .arg(&output_layer.pre_activation)
                .arg(input_for_output)
                .arg(&output_layer.weights)
                .arg(&output_layer.biases)
                .arg(&output_layer.dropout_mask)
                .arg(output_layer.num_neurons)
                .arg(output_layer.num_inputs)
                .arg(0i32) // no relu for output
                .build()?;
            unsafe { fc_kernel.enq()?; }
        }

        // Softmax (compute on CPU for simplicity)
        let mut logits_host = vec![0.0f64; self.output_size as usize];
        self.logits.read(&mut logits_host).enq()?;
        self.queue.finish()?;

        let max_val = logits_host.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = logits_host.iter().map(|&x| (x - max_val).exp()).sum();

        let softmax_kernel = Kernel::builder()
            .program(&self.program)
            .name("softmax_kernel")
            .queue(self.queue.clone())
            .global_work_size(global_size(self.output_size as usize))
            .arg(&self.softmax_output)
            .arg(&self.logits)
            .arg(self.output_size)
            .arg(max_val)
            .arg(sum_exp)
            .build()?;
        unsafe { softmax_kernel.enq()?; }

        let mut result = vec![0.0f64; self.output_size as usize];
        self.softmax_output.read(&mut result).enq()?;
        self.queue.finish()?;

        Ok(result)
    }

    /// Performs a single training step.
    pub fn train_step(
        &mut self,
        image_data: &[f64],
        target: &[f64],
    ) -> Result<f64, Box<dyn std::error::Error>> {
        self.is_training = true;
        self.adam_t += 1;

        // Forward pass
        let prediction = self.predict(image_data)?;

        // Compute cross-entropy loss
        let mut loss = 0.0;
        for i in 0..target.len() {
            if target[i] > 0.0 {
                let p = prediction[i].max(1e-15).min(1.0 - 1e-15);
                loss -= target[i] * p.ln();
            }
        }

        // Compute output gradient
        let output_grad: Vec<f64> = prediction
            .iter()
            .zip(target.iter())
            .map(|(&p, &t)| p - t)
            .collect();

        // Upload gradient
        let grad_buf: Buffer<f64> = Buffer::builder()
            .queue(self.queue.clone())
            .len(output_grad.len())
            .copy_host_slice(&output_grad)
            .build()?;

        // Backward through output layer
        if let Some(ref output_layer) = self.output_layer {
            let fc_back_kernel = Kernel::builder()
                .program(&self.program)
                .name("fc_backward_kernel")
                .queue(self.queue.clone())
                .global_work_size(global_size(output_layer.num_neurons as usize))
                .arg(&output_layer.errors)
                .arg(&grad_buf)
                .arg(&output_layer.weights)
                .arg(&output_layer.pre_activation)
                .arg(&output_layer.dropout_mask)
                .arg(output_layer.num_neurons)
                .arg(output_layer.num_inputs)
                .arg(1i32) // is_output_layer
                .build()?;
            unsafe { fc_back_kernel.enq()?; }

            // Compute input gradient
            let input_grad_kernel = Kernel::builder()
                .program(&self.program)
                .name("fc_input_grad_kernel")
                .queue(self.queue.clone())
                .global_work_size(global_size(output_layer.num_inputs as usize))
                .arg(&self.fc_grad)
                .arg(&output_layer.errors)
                .arg(&output_layer.weights)
                .arg(output_layer.num_neurons)
                .arg(output_layer.num_inputs)
                .build()?;
            unsafe { input_grad_kernel.enq()?; }
        }

        // Backward through FC layers (simplified - full implementation would include weight updates)
        for i in (0..self.fc_layers.len()).rev() {
            let fc = &self.fc_layers[i];
            let fc_back_kernel = Kernel::builder()
                .program(&self.program)
                .name("fc_backward_kernel")
                .queue(self.queue.clone())
                .global_work_size(global_size(fc.num_neurons as usize))
                .arg(&fc.errors)
                .arg(&self.fc_grad)
                .arg(&fc.weights)
                .arg(&fc.pre_activation)
                .arg(&fc.dropout_mask)
                .arg(fc.num_neurons)
                .arg(fc.num_inputs)
                .arg(0i32) // not output layer
                .build()?;
            unsafe { fc_back_kernel.enq()?; }

            if i > 0 {
                let input_grad_kernel = Kernel::builder()
                    .program(&self.program)
                    .name("fc_input_grad_kernel")
                    .queue(self.queue.clone())
                    .global_work_size(global_size(fc.num_inputs as usize))
                    .arg(&self.fc_grad)
                    .arg(&fc.errors)
                    .arg(&fc.weights)
                    .arg(fc.num_neurons)
                    .arg(fc.num_inputs)
                    .build()?;
                unsafe { input_grad_kernel.enq()?; }
            }
        }

        self.queue.finish()?;
        Ok(loss)
    }

    /// Saves the model to a JSON file.
    pub fn save_to_json(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        let mut conv_layers_json = Vec::new();
        for conv in &self.conv_layers {
            let weight_size = (conv.num_filters * conv.input_channels * conv.kernel_size * conv.kernel_size) as usize;
            let mut weights_host = vec![0.0f64; weight_size];
            conv.weights.read(&mut weights_host).enq()?;

            let mut biases_host = vec![0.0f64; conv.num_filters as usize];
            conv.biases.read(&mut biases_host).enq()?;

            let mut filters = Vec::new();
            for f in 0..conv.num_filters {
                let mut weights_3d = Vec::new();
                for c in 0..conv.input_channels {
                    let mut weights_2d = Vec::new();
                    for kh in 0..conv.kernel_size {
                        let mut weights_1d = Vec::new();
                        for kw in 0..conv.kernel_size {
                            let idx = f * conv.input_channels * conv.kernel_size * conv.kernel_size
                                + c * conv.kernel_size * conv.kernel_size
                                + kh * conv.kernel_size
                                + kw;
                            weights_1d.push(weights_host[idx as usize]);
                        }
                        weights_2d.push(weights_1d);
                    }
                    weights_3d.push(weights_2d);
                }
                filters.push(FilterJson {
                    bias: biases_host[f as usize],
                    weights: weights_3d,
                });
            }
            conv_layers_json.push(ConvLayerJson { filters });
        }

        let pool_layers_json: Vec<PoolLayerJson> = self
            .pool_layers
            .iter()
            .map(|p| PoolLayerJson {
                pool_size: p.pool_size,
            })
            .collect();

        let mut fc_layers_json = Vec::new();
        for fc in &self.fc_layers {
            let weight_size = (fc.num_neurons * fc.num_inputs) as usize;
            let mut weights_host = vec![0.0f64; weight_size];
            fc.weights.read(&mut weights_host).enq()?;

            let mut biases_host = vec![0.0f64; fc.num_neurons as usize];
            fc.biases.read(&mut biases_host).enq()?;

            let mut neurons = Vec::new();
            for n in 0..fc.num_neurons {
                let start = (n * fc.num_inputs) as usize;
                let end = ((n + 1) * fc.num_inputs) as usize;
                neurons.push(NeuronJson {
                    bias: biases_host[n as usize],
                    weights: weights_host[start..end].to_vec(),
                });
            }
            fc_layers_json.push(FCLayerJson { neurons });
        }

        let output_layer_json = if let Some(ref ol) = self.output_layer {
            let weight_size = (ol.num_neurons * ol.num_inputs) as usize;
            let mut weights_host = vec![0.0f64; weight_size];
            ol.weights.read(&mut weights_host).enq()?;

            let mut biases_host = vec![0.0f64; ol.num_neurons as usize];
            ol.biases.read(&mut biases_host).enq()?;

            let mut neurons = Vec::new();
            for n in 0..ol.num_neurons {
                let start = (n * ol.num_inputs) as usize;
                let end = ((n + 1) * ol.num_inputs) as usize;
                neurons.push(NeuronJson {
                    bias: biases_host[n as usize],
                    weights: weights_host[start..end].to_vec(),
                });
            }
            FCLayerJson { neurons }
        } else {
            FCLayerJson { neurons: Vec::new() }
        };

        self.queue.finish()?;

        let model = ModelJson {
            input_width: self.input_width,
            input_height: self.input_height,
            input_channels: self.input_channels,
            output_size: self.output_size,
            conv_filters: self.f_conv_filters.clone(),
            kernel_sizes: self.f_kernel_sizes.clone(),
            pool_sizes: self.f_pool_sizes.clone(),
            fc_layer_sizes: self.f_fc_sizes.clone(),
            learning_rate: self.learning_rate,
            dropout_rate: self.dropout_rate,
            activation: activation_to_str(self.hidden_activation).to_string(),
            output_activation: activation_to_str(self.output_activation).to_string(),
            loss_type: loss_to_str(self.loss_function).to_string(),
            gradient_clip: self.gradient_clip,
            conv_layers: conv_layers_json,
            pool_layers: pool_layers_json,
            fc_layers: fc_layers_json,
            output_layer: output_layer_json,
        };

        let file = File::create(filename)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &model)?;

        Ok(())
    }

    /// Loads a model from a JSON file.
    pub fn load_from_json(filename: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(filename)?;
        let reader = BufReader::new(file);
        let model: ModelJson = serde_json::from_reader(reader)?;

        let hidden_act = parse_activation(&model.activation);
        let output_act = parse_activation(&model.output_activation);
        let loss_type = parse_loss(&model.loss_type);

        let mut cnn = Self::new(
            model.input_width,
            model.input_height,
            model.input_channels,
            &model.conv_filters,
            &model.kernel_sizes,
            &model.pool_sizes,
            &model.fc_layer_sizes,
            model.output_size,
            hidden_act,
            output_act,
            loss_type,
            model.learning_rate,
            model.gradient_clip,
        )?;

        cnn.dropout_rate = model.dropout_rate;

        // Load conv weights
        for (i, conv_json) in model.conv_layers.iter().enumerate() {
            if i >= cnn.conv_layers.len() {
                break;
            }
            let conv = &cnn.conv_layers[i];
            let weight_size = (conv.num_filters * conv.input_channels * conv.kernel_size * conv.kernel_size) as usize;
            let mut weights_host = vec![0.0f64; weight_size];
            let mut biases_host = vec![0.0f64; conv.num_filters as usize];

            for (f, filter) in conv_json.filters.iter().enumerate() {
                biases_host[f] = filter.bias;
                for (c, channel) in filter.weights.iter().enumerate() {
                    for (kh, row) in channel.iter().enumerate() {
                        for (kw, &val) in row.iter().enumerate() {
                            let idx = f as i32 * conv.input_channels * conv.kernel_size * conv.kernel_size
                                + c as i32 * conv.kernel_size * conv.kernel_size
                                + kh as i32 * conv.kernel_size
                                + kw as i32;
                            weights_host[idx as usize] = val;
                        }
                    }
                }
            }

            conv.weights.write(&weights_host).enq()?;
            conv.biases.write(&biases_host).enq()?;
        }

        // Load FC weights
        for (i, fc_json) in model.fc_layers.iter().enumerate() {
            if i >= cnn.fc_layers.len() {
                break;
            }
            let fc = &cnn.fc_layers[i];
            let weight_size = (fc.num_neurons * fc.num_inputs) as usize;
            let mut weights_host = vec![0.0f64; weight_size];
            let mut biases_host = vec![0.0f64; fc.num_neurons as usize];

            for (n, neuron) in fc_json.neurons.iter().enumerate() {
                biases_host[n] = neuron.bias;
                let start = n * fc.num_inputs as usize;
                for (w, &val) in neuron.weights.iter().enumerate() {
                    weights_host[start + w] = val;
                }
            }

            fc.weights.write(&weights_host).enq()?;
            fc.biases.write(&biases_host).enq()?;
        }

        // Load output layer weights
        if let Some(ref ol) = cnn.output_layer {
            let weight_size = (ol.num_neurons * ol.num_inputs) as usize;
            let mut weights_host = vec![0.0f64; weight_size];
            let mut biases_host = vec![0.0f64; ol.num_neurons as usize];

            for (n, neuron) in model.output_layer.neurons.iter().enumerate() {
                biases_host[n] = neuron.bias;
                let start = n * ol.num_inputs as usize;
                for (w, &val) in neuron.weights.iter().enumerate() {
                    weights_host[start + w] = val;
                }
            }

            ol.weights.write(&weights_host).enq()?;
            ol.biases.write(&biases_host).enq()?;
        }

        cnn.queue.finish()?;
        Ok(cnn)
    }

    // Getter methods
    pub fn get_input_width(&self) -> i32 {
        self.input_width
    }

    pub fn get_input_height(&self) -> i32 {
        self.input_height
    }

    pub fn get_input_channels(&self) -> i32 {
        self.input_channels
    }

    pub fn get_output_size(&self) -> i32 {
        self.output_size
    }

    pub fn get_learning_rate(&self) -> f64 {
        self.learning_rate
    }

    pub fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr;
    }

    pub fn get_gradient_clip(&self) -> f64 {
        self.gradient_clip
    }

    pub fn set_gradient_clip(&mut self, clip: f64) {
        self.gradient_clip = clip;
    }

    pub fn set_dropout_rate(&mut self, rate: f64) {
        self.dropout_rate = rate;
    }

    pub fn get_hidden_activation(&self) -> ActivationType {
        self.hidden_activation
    }

    pub fn get_output_activation(&self) -> ActivationType {
        self.output_activation
    }

    pub fn get_loss_function(&self) -> LossType {
        self.loss_function
    }

    pub fn get_conv_filters(&self) -> &[i32] {
        &self.f_conv_filters
    }

    pub fn get_kernel_sizes(&self) -> &[i32] {
        &self.f_kernel_sizes
    }

    pub fn get_pool_sizes(&self) -> &[i32] {
        &self.f_pool_sizes
    }

    pub fn get_fc_sizes(&self) -> &[i32] {
        &self.f_fc_sizes
    }

    pub fn uses_batch_norm(&self) -> bool {
        self.use_batch_norm
    }

    pub fn initialize_batch_norm(&mut self) {
        self.use_batch_norm = true;
        self.batch_norm_params.clear();
        for conv in &self.conv_layers {
            let mut params = BatchNormParams::new();
            params.initialize((conv.num_filters * conv.output_h * conv.output_w) as usize);
            self.batch_norm_params.push(params);
        }
    }

    pub fn apply_batch_norm(&self, input: &[f64], layer_idx: usize, training: bool) -> Vec<f64> {
        if layer_idx >= self.batch_norm_params.len() {
            return input.to_vec();
        }

        let params = &self.batch_norm_params[layer_idx];
        let mut output = vec![0.0; input.len()];

        if training {
            let mean: f64 = input.iter().sum::<f64>() / input.len() as f64;
            let var: f64 = input.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / input.len() as f64;

            for i in 0..input.len() {
                let gamma = if i < params.gamma.len() { params.gamma[i] } else { 1.0 };
                let beta = if i < params.beta.len() { params.beta[i] } else { 0.0 };
                let x_norm = (input[i] - mean) / (var + params.epsilon).sqrt();
                output[i] = gamma * x_norm + beta;
            }
        } else {
            for i in 0..input.len() {
                let gamma = if i < params.gamma.len() { params.gamma[i] } else { 1.0 };
                let beta = if i < params.beta.len() { params.beta[i] } else { 0.0 };
                let mean = if i < params.running_mean.len() { params.running_mean[i] } else { 0.0 };
                let var = if i < params.running_var.len() { params.running_var[i] } else { 1.0 };
                let x_norm = (input[i] - mean) / (var + params.epsilon).sqrt();
                output[i] = gamma * x_norm + beta;
            }
        }

        output
    }
}
