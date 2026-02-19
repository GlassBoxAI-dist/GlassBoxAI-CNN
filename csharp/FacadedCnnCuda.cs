/**
 * @file
 * @ingroup CNN_Wrappers
 */
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

using System;
using System.Runtime.InteropServices;

namespace FacadedCnnCuda
{
    /// <summary>Error codes returned by the CNN library.</summary>
    public enum CnnError : int
    {
        Success = 0,
        NullPointer = 1,
        InvalidParameter = 2,
        CreationFailed = 3,
        PredictionFailed = 4,
        TrainingFailed = 5,
        SaveFailed = 6,
        LoadFailed = 7,
        ExportFailed = 8,
        ImportFailed = 9,
        BufferTooSmall = 10,
        Unknown = 255
    }

    /// <summary>Activation function types.</summary>
    public enum ActivationType : int
    {
        Sigmoid = 0,
        Tanh = 1,
        ReLU = 2,
        Linear = 3
    }

    /// <summary>Loss function types.</summary>
    public enum LossType : int
    {
        MSE = 0,
        CrossEntropy = 1
    }

    /// <summary>Configuration for creating a CNN.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct CnnConfig
    {
        public int InputWidth;
        public int InputHeight;
        public int InputChannels;
        public IntPtr ConvFilters;
        public int ConvFiltersLen;
        public IntPtr KernelSizes;
        public int KernelSizesLen;
        public IntPtr PoolSizes;
        public int PoolSizesLen;
        public IntPtr FcSizes;
        public int FcSizesLen;
        public int OutputSize;
        public ActivationType HiddenActivation;
        public ActivationType OutputActivation;
        public LossType Loss;
        public double LearningRate;
        public double GradientClip;
    }

    internal static class NativeMethods
    {
        private const string LibName = "facaded_cnn_cuda";

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern CnnError cnn_create(ref CnnConfig config, out IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void cnn_destroy(IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern CnnError cnn_predict(
            IntPtr handle,
            [In] double[] imageData, int imageLen,
            [Out] double[] output, int outputLen);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern CnnError cnn_train_step(
            IntPtr handle,
            [In] double[] imageData, int imageLen,
            [In] double[] target, int targetLen,
            out double outLoss);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern CnnError cnn_save_to_json(
            IntPtr handle,
            [MarshalAs(UnmanagedType.LPStr)] string filename);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern CnnError cnn_load_from_json(
            [MarshalAs(UnmanagedType.LPStr)] string filename,
            out IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern CnnError cnn_export_to_onnx(
            IntPtr handle,
            [MarshalAs(UnmanagedType.LPStr)] string filename);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern CnnError cnn_import_from_onnx(
            [MarshalAs(UnmanagedType.LPStr)] string filename,
            out IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int cnn_get_input_width(IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int cnn_get_input_height(IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int cnn_get_input_channels(IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int cnn_get_output_size(IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern double cnn_get_learning_rate(IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void cnn_set_learning_rate(IntPtr handle, double lr);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern double cnn_get_gradient_clip(IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void cnn_set_gradient_clip(IntPtr handle, double clip);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void cnn_set_dropout_rate(IntPtr handle, double rate);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern ActivationType cnn_get_hidden_activation(IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern ActivationType cnn_get_output_activation(IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern LossType cnn_get_loss_function(IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int cnn_uses_batch_norm(IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void cnn_initialize_batch_norm(IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr cnn_get_last_error();

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void cnn_clear_error();

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr cnn_version();

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr cnn_activation_to_string(ActivationType activation);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr cnn_loss_to_string(LossType loss);
    }

    /// <summary>Exception thrown when a CNN operation fails.</summary>
    public class CnnException : Exception
    {
        /// <summary>The error code from the native library.</summary>
        public CnnError Error { get; }

        public CnnException(CnnError error)
            : base(GetErrorMessage(error))
        {
            Error = error;
        }

        private static string GetErrorMessage(CnnError error)
        {
            IntPtr ptr = NativeMethods.cnn_get_last_error();
            if (ptr != IntPtr.Zero)
            {
                string msg = Marshal.PtrToStringAnsi(ptr);
                if (!string.IsNullOrEmpty(msg))
                    return msg;
            }
            return error.ToString();
        }
    }

    /// <summary>CUDA-accelerated Convolutional Neural Network.</summary>
    public class CNN : IDisposable
    {
        private IntPtr _handle;
        private bool _disposed;

        private CNN(IntPtr handle)
        {
            _handle = handle;
        }

        /// <summary>Creates a new CNN with the specified configuration.</summary>
        public CNN(
            int inputWidth, int inputHeight, int inputChannels,
            int[] convFilters, int[] kernelSizes, int[] poolSizes,
            int[] fcSizes, int outputSize,
            ActivationType hiddenActivation = ActivationType.ReLU,
            ActivationType outputActivation = ActivationType.Linear,
            LossType lossType = LossType.CrossEntropy,
            double learningRate = 0.001,
            double gradientClip = 5.0)
        {
            GCHandle convPin = GCHandle.Alloc(convFilters, GCHandleType.Pinned);
            GCHandle kernelPin = GCHandle.Alloc(kernelSizes, GCHandleType.Pinned);
            GCHandle poolPin = GCHandle.Alloc(poolSizes, GCHandleType.Pinned);
            GCHandle fcPin = GCHandle.Alloc(fcSizes, GCHandleType.Pinned);

            try
            {
                var config = new CnnConfig
                {
                    InputWidth = inputWidth,
                    InputHeight = inputHeight,
                    InputChannels = inputChannels,
                    ConvFilters = convPin.AddrOfPinnedObject(),
                    ConvFiltersLen = convFilters.Length,
                    KernelSizes = kernelPin.AddrOfPinnedObject(),
                    KernelSizesLen = kernelSizes.Length,
                    PoolSizes = poolPin.AddrOfPinnedObject(),
                    PoolSizesLen = poolSizes.Length,
                    FcSizes = fcPin.AddrOfPinnedObject(),
                    FcSizesLen = fcSizes.Length,
                    OutputSize = outputSize,
                    HiddenActivation = hiddenActivation,
                    OutputActivation = outputActivation,
                    Loss = lossType,
                    LearningRate = learningRate,
                    GradientClip = gradientClip
                };

                CnnError err = NativeMethods.cnn_create(ref config, out _handle);
                if (err != CnnError.Success)
                    throw new CnnException(err);
            }
            finally
            {
                convPin.Free();
                kernelPin.Free();
                poolPin.Free();
                fcPin.Free();
            }
        }

        /// <summary>Loads a CNN from a JSON file.</summary>
        public static CNN LoadFromJson(string filename)
        {
            CnnError err = NativeMethods.cnn_load_from_json(filename, out IntPtr handle);
            if (err != CnnError.Success)
                throw new CnnException(err);
            return new CNN(handle);
        }

        /// <summary>Imports a CNN from an ONNX file.</summary>
        public static CNN ImportFromOnnx(string filename)
        {
            CnnError err = NativeMethods.cnn_import_from_onnx(filename, out IntPtr handle);
            if (err != CnnError.Success)
                throw new CnnException(err);
            return new CNN(handle);
        }

        /// <summary>Performs inference on input data.</summary>
        public double[] Predict(double[] imageData)
        {
            ThrowIfDisposed();
            int outSize = OutputSize;
            double[] output = new double[outSize];

            CnnError err = NativeMethods.cnn_predict(
                _handle, imageData, imageData.Length, output, outSize);
            if (err != CnnError.Success)
                throw new CnnException(err);

            return output;
        }

        /// <summary>Performs a single training step.</summary>
        public double TrainStep(double[] imageData, double[] target)
        {
            ThrowIfDisposed();
            CnnError err = NativeMethods.cnn_train_step(
                _handle, imageData, imageData.Length,
                target, target.Length, out double loss);
            if (err != CnnError.Success)
                throw new CnnException(err);

            return loss;
        }

        /// <summary>Saves the model to a JSON file.</summary>
        public void SaveToJson(string filename)
        {
            ThrowIfDisposed();
            CnnError err = NativeMethods.cnn_save_to_json(_handle, filename);
            if (err != CnnError.Success)
                throw new CnnException(err);
        }

        /// <summary>Exports the model to ONNX format.</summary>
        public void ExportToOnnx(string filename)
        {
            ThrowIfDisposed();
            CnnError err = NativeMethods.cnn_export_to_onnx(_handle, filename);
            if (err != CnnError.Success)
                throw new CnnException(err);
        }

        /// <summary>Gets the input width.</summary>
        public int InputWidth
        {
            get { ThrowIfDisposed(); return NativeMethods.cnn_get_input_width(_handle); }
        }

        /// <summary>Gets the input height.</summary>
        public int InputHeight
        {
            get { ThrowIfDisposed(); return NativeMethods.cnn_get_input_height(_handle); }
        }

        /// <summary>Gets the number of input channels.</summary>
        public int InputChannels
        {
            get { ThrowIfDisposed(); return NativeMethods.cnn_get_input_channels(_handle); }
        }

        /// <summary>Gets the output size (number of classes).</summary>
        public int OutputSize
        {
            get { ThrowIfDisposed(); return NativeMethods.cnn_get_output_size(_handle); }
        }

        /// <summary>Gets or sets the learning rate.</summary>
        public double LearningRate
        {
            get { ThrowIfDisposed(); return NativeMethods.cnn_get_learning_rate(_handle); }
            set { ThrowIfDisposed(); NativeMethods.cnn_set_learning_rate(_handle, value); }
        }

        /// <summary>Gets or sets the gradient clipping threshold.</summary>
        public double GradientClip
        {
            get { ThrowIfDisposed(); return NativeMethods.cnn_get_gradient_clip(_handle); }
            set { ThrowIfDisposed(); NativeMethods.cnn_set_gradient_clip(_handle, value); }
        }

        /// <summary>Sets the dropout rate.</summary>
        public double DropoutRate
        {
            set { ThrowIfDisposed(); NativeMethods.cnn_set_dropout_rate(_handle, value); }
        }

        /// <summary>Gets the hidden activation type.</summary>
        public ActivationType HiddenActivation
        {
            get { ThrowIfDisposed(); return NativeMethods.cnn_get_hidden_activation(_handle); }
        }

        /// <summary>Gets the output activation type.</summary>
        public ActivationType OutputActivation
        {
            get { ThrowIfDisposed(); return NativeMethods.cnn_get_output_activation(_handle); }
        }

        /// <summary>Gets the loss function type.</summary>
        public LossType LossFunction
        {
            get { ThrowIfDisposed(); return NativeMethods.cnn_get_loss_function(_handle); }
        }

        /// <summary>Gets whether batch normalization is enabled.</summary>
        public bool UsesBatchNorm
        {
            get { ThrowIfDisposed(); return NativeMethods.cnn_uses_batch_norm(_handle) != 0; }
        }

        /// <summary>Initializes batch normalization for all convolutional layers.</summary>
        public void InitializeBatchNorm()
        {
            ThrowIfDisposed();
            NativeMethods.cnn_initialize_batch_norm(_handle);
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(CNN));
        }

        /// <summary>Releases the CNN resources.</summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (_handle != IntPtr.Zero)
                {
                    NativeMethods.cnn_destroy(_handle);
                    _handle = IntPtr.Zero;
                }
                _disposed = true;
            }
        }

        ~CNN()
        {
            Dispose(false);
        }
    }

    /// <summary>Library-level utility functions.</summary>
    public static class Library
    {
        /// <summary>Returns the library version string.</summary>
        public static string Version()
        {
            IntPtr ptr = NativeMethods.cnn_version();
            return Marshal.PtrToStringAnsi(ptr) ?? "unknown";
        }

        /// <summary>Gets the last error message from the library.</summary>
        public static string GetLastError()
        {
            IntPtr ptr = NativeMethods.cnn_get_last_error();
            if (ptr == IntPtr.Zero)
                return null;
            return Marshal.PtrToStringAnsi(ptr);
        }

        /// <summary>Clears the last error message.</summary>
        public static void ClearError()
        {
            NativeMethods.cnn_clear_error();
        }

        /// <summary>Converts an activation type to its string representation.</summary>
        public static string ActivationToString(ActivationType activation)
        {
            IntPtr ptr = NativeMethods.cnn_activation_to_string(activation);
            return Marshal.PtrToStringAnsi(ptr) ?? "unknown";
        }

        /// <summary>Converts a loss type to its string representation.</summary>
        public static string LossToString(LossType loss)
        {
            IntPtr ptr = NativeMethods.cnn_loss_to_string(loss);
            return Marshal.PtrToStringAnsi(ptr) ?? "unknown";
        }
    }
}
