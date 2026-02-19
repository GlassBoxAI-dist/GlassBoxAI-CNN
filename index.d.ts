/**
 * @file
 * @ingroup CNN_Internal_Logic
 */
/**
 * CUDA-accelerated Convolutional Neural Network library.
 *
 * @module facaded-cnn-cuda
 */

/**
 * Activation function types for neural network layers.
 */
export const enum ActivationType {
  /** Sigmoid activation function */
  Sigmoid = 'Sigmoid',
  /** Hyperbolic tangent activation function */
  Tanh = 'Tanh',
  /** Rectified Linear Unit activation function */
  ReLU = 'ReLU',
  /** Linear (identity) activation function */
  Linear = 'Linear',
}

/**
 * Loss function types for training.
 */
export const enum LossType {
  /** Mean Squared Error loss */
  MSE = 'MSE',
  /** Cross-Entropy loss (for classification) */
  CrossEntropy = 'CrossEntropy',
}

/**
 * Batch normalization parameters for a layer.
 */
export interface BatchNormParams {
  gamma: number[];
  beta: number[];
  runningMean: number[];
  runningVar: number[];
  epsilon: number;
  momentum: number;
}

/**
 * Options for creating a CNN.
 */
export interface CnnOptions {
  /** Width of input images */
  inputWidth: number;
  /** Height of input images */
  inputHeight: number;
  /** Number of input channels (e.g., 1 for grayscale, 3 for RGB) */
  inputChannels: number;
  /** Number of filters for each convolutional layer */
  convFilters: number[];
  /** Kernel size for each convolutional layer */
  kernelSizes: number[];
  /** Pooling size for each pooling layer */
  poolSizes: number[];
  /** Number of neurons for each fully-connected hidden layer */
  fcSizes: number[];
  /** Number of output classes */
  outputSize: number;
  /** Activation function for hidden layers (default: ReLU) */
  hiddenActivation?: ActivationType;
  /** Activation function for output layer (default: Linear) */
  outputActivation?: ActivationType;
  /** Loss function for training (default: CrossEntropy) */
  lossType?: LossType;
  /** Learning rate for Adam optimizer (default: 0.001) */
  learningRate?: number;
  /** Gradient clipping threshold (default: 5.0) */
  gradientClip?: number;
}

/**
 * CUDA-accelerated Convolutional Neural Network.
 *
 * This class provides a complete CNN implementation with:
 * - Configurable convolutional, pooling, and fully-connected layers
 * - Adam optimizer for training
 * - Batch normalization support
 * - Model serialization to JSON and ONNX formats
 *
 * @example
 * ```typescript
 * import { CNN, ActivationType, LossType } from 'facaded-cnn-cuda';
 *
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
 * });
 *
 * const output = cnn.predict(new Array(784).fill(0));
 * ```
 */
export class CNN {
  /**
   * Creates a new CNN with the specified architecture.
   * @param options - Configuration options for the CNN
   */
  constructor(options: CnnOptions);

  /**
   * Performs inference on an input image.
   * @param imageData - Flattened input image data (size: width * height * channels)
   * @returns Softmax probabilities for each output class
   */
  predict(imageData: number[]): number[];

  /**
   * Performs a single training step with one sample.
   * @param imageData - Flattened input image data
   * @param target - One-hot encoded target labels
   * @returns The cross-entropy loss for this sample
   */
  trainStep(imageData: number[], target: number[]): number;

  /**
   * Saves the model to a JSON file.
   * @param filename - Path to the output JSON file
   */
  saveToJson(filename: string): void;

  /**
   * Loads a model from a JSON file.
   * @param filename - Path to the JSON file
   * @returns A new CNN instance with loaded weights
   */
  static loadFromJson(filename: string): CNN;

  /**
   * Exports the model to ONNX binary format.
   * @param filename - Path to the output ONNX file
   */
  exportToOnnx(filename: string): void;

  /**
   * Imports a model from ONNX binary format.
   * @param filename - Path to the ONNX file
   * @returns A new CNN instance with loaded weights
   */
  static importFromOnnx(filename: string): CNN;

  /** Width of input images */
  readonly inputWidth: number;

  /** Height of input images */
  readonly inputHeight: number;

  /** Number of input channels */
  readonly inputChannels: number;

  /** Number of output classes */
  readonly outputSize: number;

  /** Learning rate for Adam optimizer */
  learningRate: number;

  /** Gradient clipping threshold */
  gradientClip: number;

  /** Hidden layer activation type */
  readonly hiddenActivation: string;

  /** Output layer activation type */
  readonly outputActivation: string;

  /** Loss function type */
  readonly lossFunction: string;

  /** Convolutional filter counts */
  readonly convFilters: number[];

  /** Kernel sizes */
  readonly kernelSizes: number[];

  /** Pool sizes */
  readonly poolSizes: number[];

  /** Fully-connected layer sizes */
  readonly fcSizes: number[];

  /** Whether batch normalization is enabled */
  readonly usesBatchNorm: boolean;

  /**
   * Sets the dropout rate.
   * @param rate - Dropout rate between 0 and 1
   */
  setDropoutRate(rate: number): void;

  /**
   * Initializes batch normalization for all convolutional layers.
   */
  initializeBatchNorm(): void;

  /**
   * Applies batch normalization to input data for a specific layer.
   * @param input - Input data to normalize
   * @param layerIdx - Index of the layer
   * @param training - Whether the model is in training mode
   * @returns Normalized output data
   */
  applyBatchNorm(input: number[], layerIdx: number, training: boolean): number[];
}

/**
 * Creates new batch normalization parameters.
 * @returns Empty batch normalization parameters
 */
export function createBatchNormParams(): BatchNormParams;

/**
 * Initializes batch normalization parameters for a given size.
 * @param size - Number of channels/neurons
 * @returns Initialized batch normalization parameters
 */
export function initializeBatchNormParams(size: number): BatchNormParams;
