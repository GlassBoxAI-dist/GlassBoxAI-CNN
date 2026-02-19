## @file
## @ingroup CNN_Wrappers
"""
CUDA-accelerated Convolutional Neural Network library.

This package provides a high-performance CNN implementation using CUDA acceleration,
with both Rust and Python APIs.

Example:
    >>> from facaded_cnn_cuda import CNN, ActivationType, LossType
    >>> 
    >>> # Create a CNN for MNIST-like image classification
    >>> cnn = CNN(
    ...     input_width=28, input_height=28, input_channels=1,
    ...     conv_filters=[32, 64], kernel_sizes=[3, 3], pool_sizes=[2, 2],
    ...     fc_sizes=[128], output_size=10,
    ...     hidden_activation=ActivationType.relu(),
    ...     output_activation=ActivationType.linear(),
    ...     loss_type=LossType.cross_entropy(),
    ...     learning_rate=0.001, gradient_clip=5.0
    ... )
    >>> 
    >>> # Make a prediction
    >>> output = cnn.predict([0.0] * 784)
    >>> print(f"Predicted class: {output.index(max(output))}")
"""

from .facaded_cnn_cuda import (
    ActivationType,
    LossType,
    BatchNormParams,
    CNN,
)

__all__ = [
    "ActivationType",
    "LossType", 
    "BatchNormParams",
    "CNN",
]

__version__ = "0.1.0"
