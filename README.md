# GlassBoxAI-CNN

## **GPU-Accelerated Convolutional Neural Network**

### *Multi-Language CNN with CUDA/OpenCL Support and Formal Verification*

---

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA](https://img.shields.io/badge/CUDA-12.0-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![OpenCL](https://img.shields.io/badge/OpenCL-3.0-red.svg)](https://www.khronos.org/opencl/)
[![Rust](https://img.shields.io/badge/Rust-1.75+-orange.svg)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Node.js](https://img.shields.io/badge/Node.js-14+-green.svg)](https://nodejs.org/)
[![Go](https://img.shields.io/badge/Go-1.21+-00ADD8.svg)](https://go.dev/)
[![Julia](https://img.shields.io/badge/Julia-1.6+-9558B2.svg)](https://julialang.org/)
[![ONNX](https://img.shields.io/badge/ONNX-Export%2FImport-purple.svg)](https://onnx.ai/)
[![Kani](https://img.shields.io/badge/Kani-Verified-brightgreen.svg)](https://model-checking.github.io/kani/)
[![CISA Compliant](https://img.shields.io/badge/CISA-Secure%20by%20Design-blue.svg)](https://www.cisa.gov/securebydesign)

---

## **Overview**

GlassBoxAI-CNN is a production-ready, GPU-accelerated Convolutional Neural Network implementation featuring:

- **Dual GPU backends**: CUDA for NVIDIA GPUs, OpenCL for AMD/Intel/cross-platform GPU acceleration
- **Multi-language bindings**: Native support for Rust, Python, Node.js, C, C++, Julia, and Go
- **Facade pattern architecture**: Clean API separation with deep introspection capabilities
- **Formal verification**: Kani-verified implementation for memory safety guarantees
- **Qt GUI application**: Visual training interface
- **CISA/NSA Secure by Design compliance**: Built following government cybersecurity standards

This project demonstrates enterprise-grade software engineering practices including comprehensive testing, formal verification, cross-platform compatibility, and security-first development.

---

## **Table of Contents**

1. [Features](#features)
2. [Architecture](#architecture)
3. [File Structure](#file-structure)
4. [Prerequisites](#prerequisites)
5. [Installation & Compilation](#installation--compilation)
6. [Language Bindings](#language-bindings)
   - [Rust API](#rust-api)
   - [Python API](#python-api)
   - [Node.js API](#nodejs-api)
   - [C API](#c-api)
   - [C++ API](#c-api-1)
   - [Julia API](#julia-api)
   - [Go API](#go-api)
7. [CLI Reference](#cli-reference)
8. [Formal Verification with Kani](#formal-verification-with-kani)
9. [CISA/NSA Compliance](#cisansa-compliance)
10. [License](#license)
11. [Author](#author)

---

## **Features**

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **Convolutional Layers** | Configurable multi-layer convolutions with custom kernel sizes |
| **Max Pooling** | Spatial downsampling with configurable pool sizes |
| **Fully Connected Layers** | Dense layers with arbitrary neuron counts |
| **Training** | Backpropagation with Adam optimizer and gradient clipping |
| **Activation Functions** | ReLU, Sigmoid, Tanh, Linear |
| **Loss Functions** | MSE, Cross-Entropy with stable softmax |
| **Model Persistence** | JSON serialization for model save/load |
| **Dropout** | Regularization support during training |
| **Batch Normalization** | Stabilize training with learnable scale/shift parameters |
| **ONNX Export/Import** | Interoperability with the global AI ecosystem |

### GPU Backends

| Backend | Platform | Features |
|---------|----------|----------|
| **CUDA** | NVIDIA GPUs | Full training and inference, optimized for NVIDIA hardware |
| **OpenCL** | AMD, Intel, NVIDIA, Apple | Cross-platform GPU acceleration via OpenCL 1.2+ |

### Multi-Language Support

| Language | Binding Technology | Status |
|----------|-------------------|--------|
| **Rust** | Native | ✓ Full API |
| **Python** | PyO3 | ✓ Full API |
| **Node.js** | napi-rs | ✓ Full API |
| **C** | FFI | ✓ Full API |
| **C++** | FFI + RAII Wrapper | ✓ Full API |
| **Julia** | ccall | ✓ Full API |
| **Go** | cgo | ✓ Full API |

### Safety & Security

| Feature | Technology |
|---------|------------|
| **Memory Safety** | Rust ownership model |
| **Formal Verification** | Kani proof harnesses (40+ proofs) |
| **Bounds Checking** | Verified array access |
| **Input Validation** | CLI argument validation |

---

## **Architecture**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          GlassBoxAI-CNN                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Rust Core Library                            │   │
│  │           (src/cnn.rs + src/opencl.rs)                          │   │
│  │  • CUDA/OpenCL Kernels  • Adam Optimizer  • Batch Normalization │   │
│  │  • Conv/Pool/FC Layers  • ONNX Export/Import  • JSON I/O        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                               │                                         │
│  ┌────────────────────────────┼────────────────────────────────────┐   │
│  │                    Language Bindings                             │   │
│  ├──────────┬──────────┬──────┴───┬──────────┬──────────┬─────────┤   │
│  │  Python  │  Node.js │   C/C++  │   Julia  │    Go    │   CLI   │   │
│  │  (PyO3)  │ (napi-rs)│   (FFI)  │  (ccall) │  (cgo)   │  (Rust) │   │
│  └──────────┴──────────┴──────────┴──────────┴──────────┴─────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     Security Features                            │   │
│  │  • Kani Formal Verification  • CISA Secure by Design            │   │
│  │  • Memory Safe Rust  • Comprehensive Error Handling             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## **File Structure**

```
GlassBoxAI-CNN/
│
├── src/                        # Rust source code
│   ├── lib.rs                  # Library entry point
│   ├── cnn.rs                  # Core CNN implementation (CUDA)
│   ├── opencl.rs               # OpenCL backend implementation
│   ├── main.rs                 # CLI binary
│   ├── python.rs               # Python bindings (PyO3)
│   ├── nodejs.rs               # Node.js bindings (napi-rs)
│   ├── capi.rs                 # C FFI bindings
│   └── kani_tests.rs           # Formal verification proofs
│
├── include/                    # C/C++ headers
│   ├── facaded_cnn_cuda.h      # C API header
│   └── facaded_cnn_cuda.hpp    # C++ wrapper header
│
├── python/                     # Python package
│   └── facaded_cnn_cuda/
│       ├── __init__.py         # Python module
│       └── __init__.pyi        # Type stubs
│
├── julia/                      # Julia package
│   ├── Project.toml            # Julia manifest
│   └── src/
│       └── FacadedCNNCUDA.jl   # Julia module
│
├── go/                         # Go package
│   ├── go.mod                  # Go module
│   ├── cnn.go                  # Go bindings
│   └── cnn_test.go             # Go tests
│
├── gui/                        # Qt GUI application
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs
│       └── cxx_qt_bridge.rs
│
├── Cargo.toml                  # Rust manifest
├── pyproject.toml              # Python build config
├── package.json                # Node.js package config
├── index.js                    # Node.js entry point
├── index.d.ts                  # TypeScript definitions
├── KANI_TESTS.md               # Verification documentation
└── README.md                   # This file
```

---

## **Prerequisites**

### Required

| Dependency | Version | Purpose |
|------------|---------|---------|
| **Rust** | 1.75+ | Core library compilation |

### GPU Backend (at least one)

| Dependency | Version | Purpose |
|------------|---------|---------|
| **CUDA Toolkit** | 12.0+ | NVIDIA GPU acceleration |
| **OpenCL SDK** | 1.2+ | Cross-platform GPU acceleration (AMD, Intel, NVIDIA, Apple) |

### Optional (Language Bindings)

| Dependency | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.8+ | Python bindings |
| **maturin** | 1.0+ | Python package build |
| **Node.js** | 14+ | Node.js bindings |
| **@napi-rs/cli** | 2.18+ | Node.js native module build |
| **GCC/Clang** | 11+ | C/C++ compilation |
| **Julia** | 1.6+ | Julia bindings |
| **Go** | 1.21+ | Go bindings |
| **Kani** | 0.67+ | Formal verification |
| **Qt 6** | 6.x | GUI version |

---

## **Installation & Compilation**

### **Rust Library & CLI (CUDA)**

```bash
# Build release binary and library with CUDA backend (default)
cargo build --release

# Run CLI
./target/release/facaded_cnn_cuda help
```

### **Rust Library with OpenCL Backend**

```bash
# Build with OpenCL backend (cross-platform GPU support)
cargo build --release --features opencl

# Use in your Rust project
# [dependencies]
# facaded_cnn_cuda = { path = ".", features = ["opencl"] }
```

### **Python Bindings**

```bash
# Install maturin
pip install maturin

# Build and install
maturin develop --features python

# Or build wheel
maturin build --features python --release
pip install target/wheels/facaded_cnn_cuda-*.whl
```

### **Node.js Bindings**

```bash
# Install dependencies
npm install

# Build native module
npm run build

# Development build
npm run build:debug
```

### **C/C++ Library**

```bash
# Build shared library with C API
cargo build --release --features capi

# Library located at: target/release/libfacaded_cnn_cuda.so
```

### **Julia Package**

```bash
# Build shared library first
cargo build --release --features capi

# Use in Julia
cd julia
julia --project=.
# julia> using Pkg; Pkg.instantiate()
# julia> using FacadedCNNCUDA
```

### **Go Package**

```bash
# Build shared library first
cargo build --release --features capi

# Set library path
export LD_LIBRARY_PATH=$PWD/target/release:$LD_LIBRARY_PATH

# Build Go program
cd go
go build ./...
```

### **Build All**

```bash
# Build everything
cargo build --release
cargo build --release --features python
cargo build --release --features nodejs
cargo build --release --features capi
```

---

## **Language Bindings**

### **Rust API (CUDA)**

```rust
use facaded_cnn_cuda::{ConvolutionalNeuralNetworkCUDA, ActivationType, LossType};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create CNN with CUDA backend
    let mut cnn = ConvolutionalNeuralNetworkCUDA::new(
        28, 28, 1,                    // Input dimensions
        &[32, 64],                    // Conv filters
        &[3, 3],                      // Kernel sizes
        &[2, 2],                      // Pool sizes
        &[128],                       // FC layers
        10,                           // Output classes
        ActivationType::ReLU,
        ActivationType::Linear,
        LossType::CrossEntropy,
        0.001,                        // Learning rate
        5.0,                          // Gradient clip
    )?;

    // Predict
    let input = vec![0.0; 784];
    let output = cnn.predict(&input)?;

    // Train
    let target = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let loss = cnn.train_step(&input, &target)?;

    // Save/Load
    cnn.save_to_json("model.json")?;
    let loaded = ConvolutionalNeuralNetworkCUDA::load_from_json("model.json")?;

    Ok(())
}
```

### **Rust API (OpenCL)**

```rust
// Enable with: cargo build --features opencl
use facaded_cnn_cuda::opencl::{
    ConvolutionalNeuralNetworkOpenCL, ActivationType, LossType
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create CNN with OpenCL backend (AMD, Intel, NVIDIA, Apple GPUs)
    let mut cnn = ConvolutionalNeuralNetworkOpenCL::new(
        28, 28, 1,                    // Input dimensions
        &[32, 64],                    // Conv filters
        &[3, 3],                      // Kernel sizes
        &[2, 2],                      // Pool sizes
        &[128],                       // FC layers
        10,                           // Output classes
        ActivationType::ReLU,
        ActivationType::Linear,
        LossType::CrossEntropy,
        0.001,                        // Learning rate
        5.0,                          // Gradient clip
    )?;

    // Same API as CUDA version
    let input = vec![0.0; 784];
    let output = cnn.predict(&input)?;

    let target = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let loss = cnn.train_step(&input, &target)?;

    cnn.save_to_json("model.json")?;
    let loaded = ConvolutionalNeuralNetworkOpenCL::load_from_json("model.json")?;

    Ok(())
}
```

### **Python API**

```python
from facaded_cnn_cuda import CNN, ActivationType, LossType

# Create CNN
cnn = CNN(
    input_width=28, input_height=28, input_channels=1,
    conv_filters=[32, 64], kernel_sizes=[3, 3], pool_sizes=[2, 2],
    fc_sizes=[128], output_size=10,
    hidden_activation=ActivationType.relu(),
    output_activation=ActivationType.linear(),
    loss_type=LossType.cross_entropy(),
    learning_rate=0.001, gradient_clip=5.0
)

# Predict
output = cnn.predict([0.0] * 784)
print(f"Predicted class: {output.index(max(output))}")

# Train
target = [0.0] * 10
target[3] = 1.0
loss = cnn.train_step([0.0] * 784, target)

# Save/Load
cnn.save_to_json("model.json")
loaded = CNN.load_from_json("model.json")

# ONNX
cnn.export_to_onnx("model.onnx")
onnx_cnn = CNN.import_from_onnx("model.onnx")
```

### **Node.js API**

```javascript
const { CNN, ActivationType, LossType } = require('facaded-cnn-cuda');

// Create CNN
const cnn = new CNN({
  inputWidth: 28, inputHeight: 28, inputChannels: 1,
  convFilters: [32, 64], kernelSizes: [3, 3], poolSizes: [2, 2],
  fcSizes: [128], outputSize: 10,
  hiddenActivation: ActivationType.ReLU,
  outputActivation: ActivationType.Linear,
  lossType: LossType.CrossEntropy,
  learningRate: 0.001, gradientClip: 5.0
});

// Predict
const output = cnn.predict(new Array(784).fill(0));
console.log('Predicted class:', output.indexOf(Math.max(...output)));

// Train
const target = new Array(10).fill(0);
target[3] = 1.0;
const loss = cnn.trainStep(new Array(784).fill(0), target);

// Save/Load
cnn.saveToJson('model.json');
const loaded = CNN.loadFromJson('model.json');

// ONNX
cnn.exportToOnnx('model.onnx');
const onnxCnn = CNN.importFromOnnx('model.onnx');
```

### **C API**

```c
#include "facaded_cnn_cuda.h"

int main() {
    int conv_filters[] = {32, 64};
    int kernel_sizes[] = {3, 3};
    int pool_sizes[] = {2, 2};
    int fc_sizes[] = {128};

    CnnConfig config = {
        .input_width = 28, .input_height = 28, .input_channels = 1,
        .conv_filters = conv_filters, .conv_filters_len = 2,
        .kernel_sizes = kernel_sizes, .kernel_sizes_len = 2,
        .pool_sizes = pool_sizes, .pool_sizes_len = 2,
        .fc_sizes = fc_sizes, .fc_sizes_len = 1,
        .output_size = 10,
        .hidden_activation = CNN_ACTIVATION_RELU,
        .output_activation = CNN_ACTIVATION_LINEAR,
        .loss_type = CNN_LOSS_CROSS_ENTROPY,
        .learning_rate = 0.001, .gradient_clip = 5.0
    };

    CnnHandle* cnn = NULL;
    cnn_create(&config, &cnn);

    double input[784] = {0};
    double output[10];
    cnn_predict(cnn, input, 784, output, 10);

    cnn_save_to_json(cnn, "model.json");
    cnn_destroy(cnn);
    return 0;
}
```

### **C++ API**

```cpp
#include "facaded_cnn_cuda.hpp"
#include <iostream>

int main() {
    using namespace facaded_cnn;

    CNN cnn({
        .inputWidth = 28, .inputHeight = 28, .inputChannels = 1,
        .convFilters = {32, 64}, .kernelSizes = {3, 3}, .poolSizes = {2, 2},
        .fcSizes = {128}, .outputSize = 10,
        .hiddenActivation = ActivationType::ReLU,
        .outputActivation = ActivationType::Linear,
        .lossType = LossType::CrossEntropy,
        .learningRate = 0.001, .gradientClip = 5.0
    });

    std::vector<double> input(784, 0.0);
    auto output = cnn.predict(input);

    std::vector<double> target(10, 0.0);
    target[3] = 1.0;
    double loss = cnn.trainStep(input, target);

    cnn.saveToJson("model.json");
    auto loaded = CNN::loadFromJson("model.json");

    return 0;
}
```

### **Julia API**

```julia
using FacadedCNNCUDA

# Create CNN
cnn = CNN(
    input_width=28, input_height=28, input_channels=1,
    conv_filters=[32, 64], kernel_sizes=[3, 3], pool_sizes=[2, 2],
    fc_sizes=[128], output_size=10,
    hidden_activation=ReLU, output_activation=Linear,
    loss_type=CrossEntropy, learning_rate=0.001, gradient_clip=5.0
)

# Predict
input = zeros(Float64, 784)
output = predict(cnn, input)
println("Predicted class: ", argmax(output) - 1)

# Train
target = zeros(Float64, 10)
target[4] = 1.0  # Class 3 (1-indexed)
loss = train_step!(cnn, input, target)

# Save/Load
save_to_json(cnn, "model.json")
loaded = load_from_json("model.json")

# ONNX
export_to_onnx(cnn, "model.onnx")
onnx_cnn = import_from_onnx("model.onnx")
```

### **Go API**

```go
package main

import (
    "fmt"
    "log"
    "github.com/GlassBoxAI/GlassBoxAI-CNN/go/cnn"
)

func main() {
    net, err := cnn.New(cnn.Config{
        InputWidth: 28, InputHeight: 28, InputChannels: 1,
        ConvFilters: []int{32, 64}, KernelSizes: []int{3, 3},
        PoolSizes: []int{2, 2}, FCSizes: []int{128}, OutputSize: 10,
        HiddenActivation: cnn.ReLU, OutputActivation: cnn.Linear,
        LossType: cnn.CrossEntropy,
        LearningRate: 0.001, GradientClip: 5.0,
    })
    if err != nil {
        log.Fatal(err)
    }
    defer net.Close()

    input := make([]float64, 784)
    output, _ := net.Predict(input)

    target := make([]float64, 10)
    target[3] = 1.0
    loss, _ := net.TrainStep(input, target)
    fmt.Println("Loss:", loss)

    net.SaveToJSON("model.json")
    loaded, _ := cnn.LoadFromJSON("model.json")
    defer loaded.Close()
}
```

---

## **CLI Reference**

### Usage

```
facaded_cnn_cuda <command> [options]
```

### Commands

| Command | Description |
|---------|-------------|
| `create` | Create a new CNN model |
| `train` | Train an existing model |
| `predict` | Make predictions |
| `info` | Display model information |
| `export-onnx` | Export to ONNX format |
| `import-onnx` | Import from ONNX format |
| `help` | Show help message |

### Create Options

| Option | Description |
|--------|-------------|
| `--input-w=N` | Input width (required) |
| `--input-h=N` | Input height (required) |
| `--input-c=N` | Input channels (required) |
| `--conv=N,N,...` | Conv filter counts (required) |
| `--kernels=N,N,...` | Kernel sizes (required) |
| `--pools=N,N,...` | Pool sizes (required) |
| `--fc=N,N,...` | FC layer sizes (required) |
| `--output=N` | Output size (required) |
| `--save=FILE` | Save path (required) |
| `--lr=VALUE` | Learning rate (default: 0.001) |
| `--hidden-act=TYPE` | Hidden activation (default: relu) |
| `--output-act=TYPE` | Output activation (default: linear) |
| `--loss=TYPE` | Loss function (default: mse) |
| `--clip=VALUE` | Gradient clipping (default: 5.0) |
| `--batch-norm` | Enable batch normalization |

### Examples

```bash
# Create model
facaded_cnn_cuda create \
    --input-w=28 --input-h=28 --input-c=1 \
    --conv=32,64 --kernels=3,3 --pools=2,2 \
    --fc=128 --output=10 --save=model.json

# Create with batch normalization
facaded_cnn_cuda create \
    --input-w=28 --input-h=28 --input-c=1 \
    --conv=32,64 --kernels=3,3 --pools=2,2 \
    --fc=128 --output=10 --batch-norm --save=model.json

# Model info
facaded_cnn_cuda info --model=model.json

# Export to ONNX
facaded_cnn_cuda export-onnx --model=model.json --output=model.onnx

# Import from ONNX
facaded_cnn_cuda import-onnx --input=model.onnx --save=model.json
```

---

## **Formal Verification with Kani**

### Overview

The implementation includes **Kani formal verification proofs** that mathematically prove the absence of certain classes of bugs.

### Verification Categories

| Category | Description |
|----------|-------------|
| **Strict Bound Checks** | Array/collection indexing safety |
| **Pointer Validity** | Slice-to-pointer conversion safety |
| **No-Panic Guarantee** | Enum and command handling safety |
| **Integer Overflow Prevention** | Weight size, dimension calculations |
| **Division-by-Zero Exclusion** | Launch config, pooling stride |
| **Input Sanitization Bounds** | Loop iteration limits |
| **Floating-Point Sanity** | NaN/Infinity prevention |
| **Resource Limit Compliance** | Memory budget enforcement |

### Key Proofs

- `verify_conv_filter_indexing` ✓
- `verify_output_indexing` ✓
- `verify_weight_size_no_overflow` ✓
- `verify_output_dimension_no_overflow` ✓
- `verify_activation_type_no_panic` ✓
- `verify_relu_no_nan` ✓
- `verify_gradient_clipping` ✓

### Running Verification

```bash
# Install Kani
cargo install --locked kani-verifier

# Run all proofs
cargo kani

# Run specific proof
cargo kani --harness verify_conv_filter_indexing
```

---

## **CISA/NSA Compliance**

### Secure by Design

This project follows **CISA** and **NSA** Secure by Design principles:

| Principle | Implementation |
|-----------|---------------|
| **Memory Safety** | Rust ownership model eliminates buffer overflows and data races |
| **Formal Verification** | Kani proofs mathematically verify absence of critical bugs |
| **Input Validation** | All CLI inputs validated before processing |
| **Defense in Depth** | Multiple layers of safety (language, compiler, runtime) |
| **Secure Defaults** | Safe default configurations throughout |
| **Transparency** | Open source with full code visibility |

### Compliance Checklist

- [x] **Memory-safe language** (Rust implementation)
- [x] **Static analysis** (Rust compiler + Clippy)
- [x] **Formal verification** (Kani proof harnesses)
- [x] **Comprehensive testing** (Unit + integration tests)
- [x] **Bounds checking** (Verified array access)
- [x] **Input validation** (CLI argument parsing)
- [x] **Documentation** (Inline docs + README)
- [x] **Version control** (Git)
- [x] **License clarity** (MIT License)

---

## **License**

MIT License

Copyright (c) 2025 Matthew Abbott

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

## **Author**

**Matthew Abbott**  
Email: mattbachg@gmail.com

---

*Built with precision. Verified with rigor. Secured by design.*
