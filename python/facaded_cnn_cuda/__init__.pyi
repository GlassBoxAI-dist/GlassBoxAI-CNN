"""Type stubs for facaded_cnn_cuda."""

from typing import List, Optional

class ActivationType:
    """Activation function types for neural network layers."""

    @staticmethod
    def sigmoid() -> ActivationType:
        """Create a sigmoid activation type."""
        ...

    @staticmethod
    def tanh() -> ActivationType:
        """Create a tanh activation type."""
        ...

    @staticmethod
    def relu() -> ActivationType:
        """Create a ReLU activation type."""
        ...

    @staticmethod
    def linear() -> ActivationType:
        """Create a linear activation type."""
        ...

    @staticmethod
    def from_str(s: str) -> ActivationType:
        """Create an activation type from a string ('sigmoid', 'tanh', 'relu', 'linear')."""
        ...

    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...


class LossType:
    """Loss function types for training."""

    @staticmethod
    def mse() -> LossType:
        """Create a Mean Squared Error loss type."""
        ...

    @staticmethod
    def cross_entropy() -> LossType:
        """Create a Cross-Entropy loss type."""
        ...

    @staticmethod
    def from_str(s: str) -> LossType:
        """Create a loss type from a string ('mse', 'crossentropy')."""
        ...

    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...


class BatchNormParams:
    """Batch normalization parameters for a layer."""

    gamma: List[float]
    beta: List[float]
    running_mean: List[float]
    running_var: List[float]
    epsilon: float
    momentum: float

    def __init__(self) -> None:
        """Create new batch normalization parameters."""
        ...

    def initialize(self, size: int) -> None:
        """Initialize parameters for a layer with the given size."""
        ...


class CNN:
    """CUDA-accelerated Convolutional Neural Network.

    This class provides a complete CNN implementation with:
    - Configurable convolutional, pooling, and fully-connected layers
    - Adam optimizer for training
    - Batch normalization support
    - Model serialization to JSON and ONNX formats

    Example:
        >>> cnn = CNN(
        ...     input_width=28, input_height=28, input_channels=1,
        ...     conv_filters=[32, 64], kernel_sizes=[3, 3], pool_sizes=[2, 2],
        ...     fc_sizes=[128], output_size=10
        ... )
        >>> output = cnn.predict([0.0] * 784)
    """

    input_width: int
    input_height: int
    input_channels: int
    output_size: int
    learning_rate: float
    gradient_clip: float
    hidden_activation: str
    output_activation: str
    loss_function: str
    conv_filters: List[int]
    kernel_sizes: List[int]
    pool_sizes: List[int]
    fc_sizes: List[int]
    uses_batch_norm: bool

    def __init__(
        self,
        input_width: int,
        input_height: int,
        input_channels: int,
        conv_filters: List[int],
        kernel_sizes: List[int],
        pool_sizes: List[int],
        fc_sizes: List[int],
        output_size: int,
        hidden_activation: Optional[ActivationType] = None,
        output_activation: Optional[ActivationType] = None,
        loss_type: Optional[LossType] = None,
        learning_rate: float = 0.001,
        gradient_clip: float = 5.0,
    ) -> None:
        """Creates a new CNN with the specified architecture.

        Args:
            input_width: Width of input images
            input_height: Height of input images
            input_channels: Number of input channels (e.g., 1 for grayscale, 3 for RGB)
            conv_filters: List of filter counts for each convolutional layer
            kernel_sizes: List of kernel sizes for each convolutional layer
            pool_sizes: List of pooling sizes for each pooling layer
            fc_sizes: List of neuron counts for each fully-connected hidden layer
            output_size: Number of output classes
            hidden_activation: Activation function for hidden layers (default: ReLU)
            output_activation: Activation function for output layer (default: Linear)
            loss_type: Loss function for training (default: CrossEntropy)
            learning_rate: Learning rate for Adam optimizer (default: 0.001)
            gradient_clip: Gradient clipping threshold (default: 5.0)
        """
        ...

    def predict(self, image_data: List[float]) -> List[float]:
        """Performs inference on an input image.

        Args:
            image_data: Flattened input image data (size: width * height * channels)

        Returns:
            List of softmax probabilities for each output class
        """
        ...

    def train_step(self, image_data: List[float], target: List[float]) -> float:
        """Performs a single training step with one sample.

        Args:
            image_data: Flattened input image data
            target: One-hot encoded target labels

        Returns:
            The cross-entropy loss for this sample
        """
        ...

    def save_to_json(self, filename: str) -> None:
        """Saves the model to a JSON file.

        Args:
            filename: Path to the output JSON file
        """
        ...

    @staticmethod
    def load_from_json(filename: str) -> CNN:
        """Loads a model from a JSON file.

        Args:
            filename: Path to the JSON file

        Returns:
            A new CNN instance with loaded weights
        """
        ...

    def export_to_onnx(self, filename: str) -> None:
        """Exports the model to ONNX binary format.

        Args:
            filename: Path to the output ONNX file
        """
        ...

    @staticmethod
    def import_from_onnx(filename: str) -> CNN:
        """Imports a model from ONNX binary format.

        Args:
            filename: Path to the ONNX file

        Returns:
            A new CNN instance with loaded weights
        """
        ...

    def set_dropout_rate(self, rate: float) -> None:
        """Sets the dropout rate.

        Args:
            rate: Dropout rate between 0 and 1
        """
        ...

    def initialize_batch_norm(self) -> None:
        """Initializes batch normalization for all convolutional layers."""
        ...

    def apply_batch_norm(
        self, input: List[float], layer_idx: int, training: bool
    ) -> List[float]:
        """Applies batch normalization to input data for a specific layer.

        Args:
            input: Input data to normalize
            layer_idx: Index of the layer
            training: Whether the model is in training mode

        Returns:
            Normalized output data
        """
        ...
