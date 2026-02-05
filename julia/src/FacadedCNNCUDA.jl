#=
MIT License

Copyright (c) 2025 Matthew Abbott

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
=#

"""
    FacadedCNNCUDA

CUDA-accelerated Convolutional Neural Network library for Julia.

# Example
```julia
using FacadedCNNCUDA

# Create a CNN for MNIST classification
cnn = CNN(
    input_width=28, input_height=28, input_channels=1,
    conv_filters=[32, 64], kernel_sizes=[3, 3], pool_sizes=[2, 2],
    fc_sizes=[128], output_size=10,
    hidden_activation=ReLU, output_activation=Linear,
    loss_type=CrossEntropy, learning_rate=0.001, gradient_clip=5.0
)

# Make a prediction
input = zeros(Float64, 784)
output = predict(cnn, input)
println("Predicted class: ", argmax(output) - 1)

# Training step
target = zeros(Float64, 10)
target[4] = 1.0  # Class 3 (1-indexed)
loss = train_step!(cnn, input, target)
println("Training loss: ", loss)

# Save and load
save_to_json(cnn, "model.json")
loaded_cnn = load_from_json("model.json")
```
"""
module FacadedCNNCUDA

export CNN, ActivationType, LossType
export Sigmoid, Tanh, ReLU, Linear
export MSE, CrossEntropy
export predict, train_step!, save_to_json, load_from_json
export export_to_onnx, import_from_onnx
export input_width, input_height, input_channels, output_size
export learning_rate, learning_rate!, gradient_clip, gradient_clip!
export dropout_rate!, hidden_activation, output_activation, loss_function
export uses_batch_norm, initialize_batch_norm!
export version

using Libdl

# Find the library
const LIBNAME = "libfacaded_cnn_cuda"

function find_library()
    # Search paths
    search_paths = [
        joinpath(@__DIR__, "..", "..", "target", "release"),
        joinpath(@__DIR__, "..", "..", "target", "debug"),
        joinpath(@__DIR__, ".."),
        "/usr/local/lib",
        "/usr/lib",
    ]
    
    for path in search_paths
        libpath = joinpath(path, LIBNAME * "." * Libdl.dlext)
        if isfile(libpath)
            return libpath
        end
    end
    
    # Try system path
    return LIBNAME
end

const libcnn = Ref{Ptr{Cvoid}}(C_NULL)

function __init__()
    libpath = find_library()
    libcnn[] = Libdl.dlopen(libpath)
end

# Error codes
@enum CnnError::Cint begin
    CNN_SUCCESS = 0
    CNN_ERROR_NULL_POINTER = 1
    CNN_ERROR_INVALID_PARAMETER = 2
    CNN_ERROR_CREATION_FAILED = 3
    CNN_ERROR_PREDICTION_FAILED = 4
    CNN_ERROR_TRAINING_FAILED = 5
    CNN_ERROR_SAVE_FAILED = 6
    CNN_ERROR_LOAD_FAILED = 7
    CNN_ERROR_EXPORT_FAILED = 8
    CNN_ERROR_IMPORT_FAILED = 9
    CNN_ERROR_BUFFER_TOO_SMALL = 10
    CNN_ERROR_UNKNOWN = 255
end

"""
Activation function types.
"""
@enum ActivationType::Cint begin
    Sigmoid = 0
    Tanh = 1
    ReLU = 2
    Linear = 3
end

"""
Loss function types.
"""
@enum LossType::Cint begin
    MSE = 0
    CrossEntropy = 1
end

# C struct for configuration
struct CnnConfig
    input_width::Cint
    input_height::Cint
    input_channels::Cint
    conv_filters::Ptr{Cint}
    conv_filters_len::Cint
    kernel_sizes::Ptr{Cint}
    kernel_sizes_len::Cint
    pool_sizes::Ptr{Cint}
    pool_sizes_len::Cint
    fc_sizes::Ptr{Cint}
    fc_sizes_len::Cint
    output_size::Cint
    hidden_activation::ActivationType
    output_activation::ActivationType
    loss_type::LossType
    learning_rate::Cdouble
    gradient_clip::Cdouble
end

# Get last error message
function get_last_error()
    ptr = ccall(Libdl.dlsym(libcnn[], :cnn_get_last_error), Ptr{Cchar}, ())
    if ptr == C_NULL
        return "Unknown error"
    end
    return unsafe_string(ptr)
end

# Check error and throw if failed
function check_error(err::CnnError)
    if err != CNN_SUCCESS
        error("CNN error: $(get_last_error())")
    end
end

"""
    CNN

CUDA-accelerated Convolutional Neural Network.

# Constructor Arguments
- `input_width::Int`: Width of input images
- `input_height::Int`: Height of input images
- `input_channels::Int`: Number of input channels
- `conv_filters::Vector{Int}`: Filter counts for each conv layer
- `kernel_sizes::Vector{Int}`: Kernel sizes for each conv layer
- `pool_sizes::Vector{Int}`: Pool sizes for each pool layer
- `fc_sizes::Vector{Int}`: Sizes for each FC hidden layer
- `output_size::Int`: Number of output classes
- `hidden_activation::ActivationType`: Hidden layer activation (default: ReLU)
- `output_activation::ActivationType`: Output layer activation (default: Linear)
- `loss_type::LossType`: Loss function (default: CrossEntropy)
- `learning_rate::Float64`: Learning rate (default: 0.001)
- `gradient_clip::Float64`: Gradient clipping threshold (default: 5.0)
"""
mutable struct CNN
    handle::Ptr{Cvoid}
    
    function CNN(handle::Ptr{Cvoid})
        cnn = new(handle)
        finalizer(cnn) do x
            if x.handle != C_NULL
                ccall(Libdl.dlsym(libcnn[], :cnn_destroy), Cvoid, (Ptr{Cvoid},), x.handle)
                x.handle = C_NULL
            end
        end
        return cnn
    end
end

function CNN(;
    input_width::Int,
    input_height::Int,
    input_channels::Int,
    conv_filters::Vector{Int},
    kernel_sizes::Vector{Int},
    pool_sizes::Vector{Int},
    fc_sizes::Vector{Int},
    output_size::Int,
    hidden_activation::ActivationType = ReLU,
    output_activation::ActivationType = Linear,
    loss_type::LossType = CrossEntropy,
    learning_rate::Float64 = 0.001,
    gradient_clip::Float64 = 5.0
)
    # Convert to Cint arrays
    conv_filters_c = Cint.(conv_filters)
    kernel_sizes_c = Cint.(kernel_sizes)
    pool_sizes_c = Cint.(pool_sizes)
    fc_sizes_c = Cint.(fc_sizes)
    
    config = CnnConfig(
        Cint(input_width),
        Cint(input_height),
        Cint(input_channels),
        pointer(conv_filters_c),
        Cint(length(conv_filters)),
        pointer(kernel_sizes_c),
        Cint(length(kernel_sizes)),
        pointer(pool_sizes_c),
        Cint(length(pool_sizes)),
        pointer(fc_sizes_c),
        Cint(length(fc_sizes)),
        Cint(output_size),
        hidden_activation,
        output_activation,
        loss_type,
        Cdouble(learning_rate),
        Cdouble(gradient_clip)
    )
    
    handle_ref = Ref{Ptr{Cvoid}}(C_NULL)
    
    # Keep arrays alive during ccall
    GC.@preserve conv_filters_c kernel_sizes_c pool_sizes_c fc_sizes_c begin
        err = ccall(
            Libdl.dlsym(libcnn[], :cnn_create),
            CnnError,
            (Ref{CnnConfig}, Ref{Ptr{Cvoid}}),
            config, handle_ref
        )
        check_error(err)
    end
    
    return CNN(handle_ref[])
end

"""
    predict(cnn::CNN, input::Vector{Float64}) -> Vector{Float64}

Perform inference on input data.

# Arguments
- `cnn`: The CNN instance
- `input`: Flattened input image data

# Returns
Softmax probabilities for each output class.
"""
function predict(cnn::CNN, input::Vector{Float64})
    out_size = output_size(cnn)
    output = Vector{Float64}(undef, out_size)
    
    err = ccall(
        Libdl.dlsym(libcnn[], :cnn_predict),
        CnnError,
        (Ptr{Cvoid}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint),
        cnn.handle, input, length(input), output, out_size
    )
    check_error(err)
    
    return output
end

"""
    train_step!(cnn::CNN, input::Vector{Float64}, target::Vector{Float64}) -> Float64

Perform a single training step.

# Arguments
- `cnn`: The CNN instance
- `input`: Flattened input image data
- `target`: One-hot encoded target labels

# Returns
The cross-entropy loss for this sample.
"""
function train_step!(cnn::CNN, input::Vector{Float64}, target::Vector{Float64})
    loss = Ref{Cdouble}(0.0)
    
    err = ccall(
        Libdl.dlsym(libcnn[], :cnn_train_step),
        CnnError,
        (Ptr{Cvoid}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Ref{Cdouble}),
        cnn.handle, input, length(input), target, length(target), loss
    )
    check_error(err)
    
    return loss[]
end

"""
    save_to_json(cnn::CNN, filename::String)

Save the model to a JSON file.
"""
function save_to_json(cnn::CNN, filename::String)
    err = ccall(
        Libdl.dlsym(libcnn[], :cnn_save_to_json),
        CnnError,
        (Ptr{Cvoid}, Cstring),
        cnn.handle, filename
    )
    check_error(err)
end

"""
    load_from_json(filename::String) -> CNN

Load a model from a JSON file.
"""
function load_from_json(filename::String)
    handle_ref = Ref{Ptr{Cvoid}}(C_NULL)
    
    err = ccall(
        Libdl.dlsym(libcnn[], :cnn_load_from_json),
        CnnError,
        (Cstring, Ref{Ptr{Cvoid}}),
        filename, handle_ref
    )
    check_error(err)
    
    return CNN(handle_ref[])
end

"""
    export_to_onnx(cnn::CNN, filename::String)

Export the model to ONNX format.
"""
function export_to_onnx(cnn::CNN, filename::String)
    err = ccall(
        Libdl.dlsym(libcnn[], :cnn_export_to_onnx),
        CnnError,
        (Ptr{Cvoid}, Cstring),
        cnn.handle, filename
    )
    check_error(err)
end

"""
    import_from_onnx(filename::String) -> CNN

Import a model from ONNX format.
"""
function import_from_onnx(filename::String)
    handle_ref = Ref{Ptr{Cvoid}}(C_NULL)
    
    err = ccall(
        Libdl.dlsym(libcnn[], :cnn_import_from_onnx),
        CnnError,
        (Cstring, Ref{Ptr{Cvoid}}),
        filename, handle_ref
    )
    check_error(err)
    
    return CNN(handle_ref[])
end

# Property getters

"""Get the input width."""
function input_width(cnn::CNN)
    return Int(ccall(Libdl.dlsym(libcnn[], :cnn_get_input_width), Cint, (Ptr{Cvoid},), cnn.handle))
end

"""Get the input height."""
function input_height(cnn::CNN)
    return Int(ccall(Libdl.dlsym(libcnn[], :cnn_get_input_height), Cint, (Ptr{Cvoid},), cnn.handle))
end

"""Get the number of input channels."""
function input_channels(cnn::CNN)
    return Int(ccall(Libdl.dlsym(libcnn[], :cnn_get_input_channels), Cint, (Ptr{Cvoid},), cnn.handle))
end

"""Get the output size (number of classes)."""
function output_size(cnn::CNN)
    return Int(ccall(Libdl.dlsym(libcnn[], :cnn_get_output_size), Cint, (Ptr{Cvoid},), cnn.handle))
end

"""Get the learning rate."""
function learning_rate(cnn::CNN)
    return ccall(Libdl.dlsym(libcnn[], :cnn_get_learning_rate), Cdouble, (Ptr{Cvoid},), cnn.handle)
end

"""Set the learning rate."""
function learning_rate!(cnn::CNN, lr::Float64)
    ccall(Libdl.dlsym(libcnn[], :cnn_set_learning_rate), Cvoid, (Ptr{Cvoid}, Cdouble), cnn.handle, lr)
end

"""Get the gradient clipping threshold."""
function gradient_clip(cnn::CNN)
    return ccall(Libdl.dlsym(libcnn[], :cnn_get_gradient_clip), Cdouble, (Ptr{Cvoid},), cnn.handle)
end

"""Set the gradient clipping threshold."""
function gradient_clip!(cnn::CNN, clip::Float64)
    ccall(Libdl.dlsym(libcnn[], :cnn_set_gradient_clip), Cvoid, (Ptr{Cvoid}, Cdouble), cnn.handle, clip)
end

"""Set the dropout rate."""
function dropout_rate!(cnn::CNN, rate::Float64)
    ccall(Libdl.dlsym(libcnn[], :cnn_set_dropout_rate), Cvoid, (Ptr{Cvoid}, Cdouble), cnn.handle, rate)
end

"""Get the hidden activation type."""
function hidden_activation(cnn::CNN)
    return ActivationType(ccall(Libdl.dlsym(libcnn[], :cnn_get_hidden_activation), Cint, (Ptr{Cvoid},), cnn.handle))
end

"""Get the output activation type."""
function output_activation(cnn::CNN)
    return ActivationType(ccall(Libdl.dlsym(libcnn[], :cnn_get_output_activation), Cint, (Ptr{Cvoid},), cnn.handle))
end

"""Get the loss function type."""
function loss_function(cnn::CNN)
    return LossType(ccall(Libdl.dlsym(libcnn[], :cnn_get_loss_function), Cint, (Ptr{Cvoid},), cnn.handle))
end

"""Check if batch normalization is enabled."""
function uses_batch_norm(cnn::CNN)
    return ccall(Libdl.dlsym(libcnn[], :cnn_uses_batch_norm), Cint, (Ptr{Cvoid},), cnn.handle) != 0
end

"""Initialize batch normalization for all convolutional layers."""
function initialize_batch_norm!(cnn::CNN)
    ccall(Libdl.dlsym(libcnn[], :cnn_initialize_batch_norm), Cvoid, (Ptr{Cvoid},), cnn.handle)
end

"""Get the library version."""
function version()
    ptr = ccall(Libdl.dlsym(libcnn[], :cnn_version), Ptr{Cchar}, ())
    return unsafe_string(ptr)
end

# Pretty printing
function Base.show(io::IO, cnn::CNN)
    print(io, "CNN($(input_width(cnn))×$(input_height(cnn))×$(input_channels(cnn)) → $(output_size(cnn)))")
end

function Base.show(io::IO, ::MIME"text/plain", cnn::CNN)
    println(io, "CNN:")
    println(io, "  Input: $(input_width(cnn))×$(input_height(cnn))×$(input_channels(cnn))")
    println(io, "  Output: $(output_size(cnn)) classes")
    println(io, "  Learning rate: $(learning_rate(cnn))")
    println(io, "  Gradient clip: $(gradient_clip(cnn))")
    println(io, "  Hidden activation: $(hidden_activation(cnn))")
    println(io, "  Output activation: $(output_activation(cnn))")
    println(io, "  Loss function: $(loss_function(cnn))")
    print(io, "  Batch normalization: $(uses_batch_norm(cnn) ? "enabled" : "disabled")")
end

end # module
