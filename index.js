/**
 * @file
 * @ingroup CNN_Wrappers
 */
/**
 * CUDA-accelerated Convolutional Neural Network library.
 *
 * This package provides a high-performance CNN implementation using CUDA acceleration,
 * with a native Node.js API.
 *
 * @example
 * ```javascript
 * const { CNN, ActivationType, LossType } = require('facaded-cnn-cuda');
 *
 * // Create a CNN for MNIST-like image classification
 * const cnn = new CNN({
 *   inputWidth: 28,
 *   inputHeight: 28,
 *   inputChannels: 1,
 *   convFilters: [32, 64],
 *   kernelSizes: [3, 3],
 *   poolSizes: [2, 2],
 *   fcSizes: [128],
 *   outputSize: 10,
 *   hiddenActivation: ActivationType.ReLU,
 *   outputActivation: ActivationType.Linear,
 *   lossType: LossType.CrossEntropy,
 *   learningRate: 0.001,
 *   gradientClip: 5.0
 * });
 *
 * // Make a prediction
 * const input = new Array(784).fill(0);
 * const output = cnn.predict(input);
 * console.log('Predicted class:', output.indexOf(Math.max(...output)));
 * ```
 *
 * @module facaded-cnn-cuda
 */

const { platform, arch } = process;

let nativeBinding = null;
let loadError = null;

/**
 * Try to load the native binding for the current platform.
 */
function loadBinding() {
  const platformArch = `${platform}-${arch}`;
  
  // Try platform-specific binding first
  const bindingPaths = [
    // Local development build
    `./facaded-cnn-cuda.${platformArch}.node`,
    // Fallback to generic name
    './facaded-cnn-cuda.node',
    // Target directory builds
    `./target/release/libfacaded_cnn_cuda.node`,
    `./target/debug/libfacaded_cnn_cuda.node`,
  ];

  for (const bindingPath of bindingPaths) {
    try {
      nativeBinding = require(bindingPath);
      return;
    } catch (e) {
      // Continue trying other paths
    }
  }

  loadError = new Error(
    `Failed to load native binding for ${platformArch}. ` +
    `Make sure you have built the native module with: npm run build`
  );
}

loadBinding();

/**
 * Activation function types for neural network layers.
 * @enum {string}
 */
const ActivationType = {
  /** Sigmoid activation function */
  Sigmoid: 'Sigmoid',
  /** Hyperbolic tangent activation function */
  Tanh: 'Tanh',
  /** Rectified Linear Unit activation function */
  ReLU: 'ReLU',
  /** Linear (identity) activation function */
  Linear: 'Linear',
};

/**
 * Loss function types for training.
 * @enum {string}
 */
const LossType = {
  /** Mean Squared Error loss */
  MSE: 'MSE',
  /** Cross-Entropy loss (for classification) */
  CrossEntropy: 'CrossEntropy',
};

/**
 * Get the native CNN class.
 * @throws {Error} If the native binding failed to load
 * @returns {typeof import('./index').CNN}
 */
function getCNN() {
  if (loadError) {
    throw loadError;
  }
  return nativeBinding.CNN;
}

/**
 * Create new batch normalization parameters.
 * @returns {import('./index').BatchNormParams}
 */
function createBatchNormParams() {
  if (loadError) {
    throw loadError;
  }
  return nativeBinding.createBatchNormParams();
}

/**
 * Initialize batch normalization parameters for a given size.
 * @param {number} size - Number of channels/neurons
 * @returns {import('./index').BatchNormParams}
 */
function initializeBatchNormParams(size) {
  if (loadError) {
    throw loadError;
  }
  return nativeBinding.initializeBatchNormParams(size);
}

// Export everything
module.exports = {
  ActivationType,
  LossType,
  get CNN() {
    return getCNN();
  },
  createBatchNormParams,
  initializeBatchNormParams,
};

// Also support ES module style imports
module.exports.default = module.exports;
