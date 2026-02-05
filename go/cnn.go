/*
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
*/

// Package cnn provides Go bindings for the CUDA-accelerated CNN library.
//
// Example usage:
//
//	package main
//
//	import (
//		"fmt"
//		"github.com/GlassBoxAI/GlassBoxAI-CNN/go/cnn"
//	)
//
//	func main() {
//		// Create a CNN for MNIST classification
//		config := cnn.Config{
//			InputWidth:       28,
//			InputHeight:      28,
//			InputChannels:    1,
//			ConvFilters:      []int{32, 64},
//			KernelSizes:      []int{3, 3},
//			PoolSizes:        []int{2, 2},
//			FCSizes:          []int{128},
//			OutputSize:       10,
//			HiddenActivation: cnn.ReLU,
//			OutputActivation: cnn.Linear,
//			LossType:         cnn.CrossEntropy,
//			LearningRate:     0.001,
//			GradientClip:     5.0,
//		}
//
//		net, err := cnn.New(config)
//		if err != nil {
//			panic(err)
//		}
//		defer net.Close()
//
//		// Make a prediction
//		input := make([]float64, 784)
//		output, err := net.Predict(input)
//		if err != nil {
//			panic(err)
//		}
//		fmt.Println("Predictions:", output)
//	}
package cnn

/*
#cgo CFLAGS: -I${SRCDIR}/../include
#cgo LDFLAGS: -L${SRCDIR}/../target/release -lfacaded_cnn_cuda -lm

#include "facaded_cnn_cuda.h"
#include <stdlib.h>
*/
import "C"
import (
	"errors"
	"runtime"
	"unsafe"
)

// ActivationType represents an activation function.
type ActivationType int

const (
	// Sigmoid activation function.
	Sigmoid ActivationType = iota
	// Tanh activation function.
	Tanh
	// ReLU activation function.
	ReLU
	// Linear activation function.
	Linear
)

func (a ActivationType) String() string {
	switch a {
	case Sigmoid:
		return "sigmoid"
	case Tanh:
		return "tanh"
	case ReLU:
		return "relu"
	case Linear:
		return "linear"
	default:
		return "unknown"
	}
}

// LossType represents a loss function.
type LossType int

const (
	// MSE is Mean Squared Error loss.
	MSE LossType = iota
	// CrossEntropy loss for classification.
	CrossEntropy
)

func (l LossType) String() string {
	switch l {
	case MSE:
		return "mse"
	case CrossEntropy:
		return "crossentropy"
	default:
		return "unknown"
	}
}

// Config holds the configuration for creating a CNN.
type Config struct {
	InputWidth       int
	InputHeight      int
	InputChannels    int
	ConvFilters      []int
	KernelSizes      []int
	PoolSizes        []int
	FCSizes          []int
	OutputSize       int
	HiddenActivation ActivationType
	OutputActivation ActivationType
	LossType         LossType
	LearningRate     float64
	GradientClip     float64
}

// CNN is a CUDA-accelerated Convolutional Neural Network.
type CNN struct {
	handle *C.CnnHandle
}

// getLastError returns the last error message from the C library.
func getLastError() string {
	cErr := C.cnn_get_last_error()
	if cErr == nil {
		return "unknown error"
	}
	return C.GoString(cErr)
}

// New creates a new CNN with the given configuration.
func New(config Config) (*CNN, error) {
	// Convert slices to C arrays
	convFilters := make([]C.int, len(config.ConvFilters))
	for i, v := range config.ConvFilters {
		convFilters[i] = C.int(v)
	}

	kernelSizes := make([]C.int, len(config.KernelSizes))
	for i, v := range config.KernelSizes {
		kernelSizes[i] = C.int(v)
	}

	poolSizes := make([]C.int, len(config.PoolSizes))
	for i, v := range config.PoolSizes {
		poolSizes[i] = C.int(v)
	}

	fcSizes := make([]C.int, len(config.FCSizes))
	for i, v := range config.FCSizes {
		fcSizes[i] = C.int(v)
	}

	cConfig := C.CnnConfig{
		input_width:       C.int(config.InputWidth),
		input_height:      C.int(config.InputHeight),
		input_channels:    C.int(config.InputChannels),
		conv_filters:      &convFilters[0],
		conv_filters_len:  C.int(len(convFilters)),
		kernel_sizes:      &kernelSizes[0],
		kernel_sizes_len:  C.int(len(kernelSizes)),
		pool_sizes:        &poolSizes[0],
		pool_sizes_len:    C.int(len(poolSizes)),
		fc_sizes:          &fcSizes[0],
		fc_sizes_len:      C.int(len(fcSizes)),
		output_size:       C.int(config.OutputSize),
		hidden_activation: C.CnnActivationType(config.HiddenActivation),
		output_activation: C.CnnActivationType(config.OutputActivation),
		loss_type:         C.CnnLossType(config.LossType),
		learning_rate:     C.double(config.LearningRate),
		gradient_clip:     C.double(config.GradientClip),
	}

	var handle *C.CnnHandle
	err := C.cnn_create(&cConfig, &handle)
	if err != C.CNN_SUCCESS {
		return nil, errors.New(getLastError())
	}

	cnn := &CNN{handle: handle}
	runtime.SetFinalizer(cnn, (*CNN).Close)
	return cnn, nil
}

// LoadFromJSON loads a CNN from a JSON file.
func LoadFromJSON(filename string) (*CNN, error) {
	cFilename := C.CString(filename)
	defer C.free(unsafe.Pointer(cFilename))

	var handle *C.CnnHandle
	err := C.cnn_load_from_json(cFilename, &handle)
	if err != C.CNN_SUCCESS {
		return nil, errors.New(getLastError())
	}

	cnn := &CNN{handle: handle}
	runtime.SetFinalizer(cnn, (*CNN).Close)
	return cnn, nil
}

// ImportFromONNX imports a CNN from an ONNX file.
func ImportFromONNX(filename string) (*CNN, error) {
	cFilename := C.CString(filename)
	defer C.free(unsafe.Pointer(cFilename))

	var handle *C.CnnHandle
	err := C.cnn_import_from_onnx(cFilename, &handle)
	if err != C.CNN_SUCCESS {
		return nil, errors.New(getLastError())
	}

	cnn := &CNN{handle: handle}
	runtime.SetFinalizer(cnn, (*CNN).Close)
	return cnn, nil
}

// Close releases the CNN resources.
func (c *CNN) Close() error {
	if c.handle != nil {
		C.cnn_destroy(c.handle)
		c.handle = nil
	}
	return nil
}

// Predict performs inference on input data.
func (c *CNN) Predict(input []float64) ([]float64, error) {
	if c.handle == nil {
		return nil, errors.New("CNN handle is nil")
	}

	outputSize := c.OutputSize()
	output := make([]float64, outputSize)

	err := C.cnn_predict(
		c.handle,
		(*C.double)(unsafe.Pointer(&input[0])),
		C.int(len(input)),
		(*C.double)(unsafe.Pointer(&output[0])),
		C.int(outputSize),
	)
	if err != C.CNN_SUCCESS {
		return nil, errors.New(getLastError())
	}

	return output, nil
}

// TrainStep performs a single training step.
func (c *CNN) TrainStep(input, target []float64) (float64, error) {
	if c.handle == nil {
		return 0, errors.New("CNN handle is nil")
	}

	var loss C.double
	err := C.cnn_train_step(
		c.handle,
		(*C.double)(unsafe.Pointer(&input[0])),
		C.int(len(input)),
		(*C.double)(unsafe.Pointer(&target[0])),
		C.int(len(target)),
		&loss,
	)
	if err != C.CNN_SUCCESS {
		return 0, errors.New(getLastError())
	}

	return float64(loss), nil
}

// SaveToJSON saves the model to a JSON file.
func (c *CNN) SaveToJSON(filename string) error {
	if c.handle == nil {
		return errors.New("CNN handle is nil")
	}

	cFilename := C.CString(filename)
	defer C.free(unsafe.Pointer(cFilename))

	err := C.cnn_save_to_json(c.handle, cFilename)
	if err != C.CNN_SUCCESS {
		return errors.New(getLastError())
	}

	return nil
}

// ExportToONNX exports the model to ONNX format.
func (c *CNN) ExportToONNX(filename string) error {
	if c.handle == nil {
		return errors.New("CNN handle is nil")
	}

	cFilename := C.CString(filename)
	defer C.free(unsafe.Pointer(cFilename))

	err := C.cnn_export_to_onnx(c.handle, cFilename)
	if err != C.CNN_SUCCESS {
		return errors.New(getLastError())
	}

	return nil
}

// InputWidth returns the input width.
func (c *CNN) InputWidth() int {
	return int(C.cnn_get_input_width(c.handle))
}

// InputHeight returns the input height.
func (c *CNN) InputHeight() int {
	return int(C.cnn_get_input_height(c.handle))
}

// InputChannels returns the number of input channels.
func (c *CNN) InputChannels() int {
	return int(C.cnn_get_input_channels(c.handle))
}

// OutputSize returns the output size (number of classes).
func (c *CNN) OutputSize() int {
	return int(C.cnn_get_output_size(c.handle))
}

// LearningRate returns the learning rate.
func (c *CNN) LearningRate() float64 {
	return float64(C.cnn_get_learning_rate(c.handle))
}

// SetLearningRate sets the learning rate.
func (c *CNN) SetLearningRate(lr float64) {
	C.cnn_set_learning_rate(c.handle, C.double(lr))
}

// GradientClip returns the gradient clipping threshold.
func (c *CNN) GradientClip() float64 {
	return float64(C.cnn_get_gradient_clip(c.handle))
}

// SetGradientClip sets the gradient clipping threshold.
func (c *CNN) SetGradientClip(clip float64) {
	C.cnn_set_gradient_clip(c.handle, C.double(clip))
}

// SetDropoutRate sets the dropout rate.
func (c *CNN) SetDropoutRate(rate float64) {
	C.cnn_set_dropout_rate(c.handle, C.double(rate))
}

// HiddenActivation returns the hidden layer activation type.
func (c *CNN) HiddenActivation() ActivationType {
	return ActivationType(C.cnn_get_hidden_activation(c.handle))
}

// OutputActivation returns the output layer activation type.
func (c *CNN) OutputActivation() ActivationType {
	return ActivationType(C.cnn_get_output_activation(c.handle))
}

// GetLossType returns the loss function type.
func (c *CNN) GetLossType() LossType {
	return LossType(C.cnn_get_loss_function(c.handle))
}

// UsesBatchNorm returns whether batch normalization is enabled.
func (c *CNN) UsesBatchNorm() bool {
	return C.cnn_uses_batch_norm(c.handle) != 0
}

// InitializeBatchNorm initializes batch normalization for all conv layers.
func (c *CNN) InitializeBatchNorm() {
	C.cnn_initialize_batch_norm(c.handle)
}

// Version returns the library version.
func Version() string {
	return C.GoString(C.cnn_version())
}
