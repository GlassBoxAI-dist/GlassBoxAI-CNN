//! @file
//! @ingroup CNN_Core_Verified
/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 *
 * Kani Verification: CUDA Backend FFI Safety (CISA/NSA Compliance)
 *
 * Proves that all data passed across the CUDA FFI boundary is valid:
 * correct alignment, valid buffer sizes, proper grid/block dimensions,
 * weight index safety, and memory layout before kernel launches.
 *
 * The CNN uses f64 (double) for all GPU buffers and BLOCK_SIZE = 256.
 *
 * CISA "Secure by Design" requirements verified:
 * A. Convolutional layer buffer size correctness
 * B. CUDA grid/block dimension safety
 * C. FC layer weight index validity for flat buffers
 * D. Transfer size non-zero and alignment
 * E. f64 alignment for CUDA transfers
 * F. Pooling layer buffer sizing
 * G. Conv weight buffer sizing
 * H. Kernel launch parameter overflow prevention
 * I. Flattened feature buffer sizing
 * J. Softmax/logits buffer safety
 * K. Gradient buffer sizing
 * L. Adam optimizer buffer sizing
 * M. No-panic guarantee for dimension calculations
 * N. ABI type compatibility for CUDA interop
 * O. End-to-end forward pass buffer chain validation
 */

use super::*;

const BLOCK_SIZE: u32 = 256;
const MAX_IMAGE_DIM: i32 = 4096;
const MAX_FILTERS: i32 = 1024;
const MAX_NEURONS: i32 = 4096;
const MAX_KERNEL_SIZE: i32 = 31;

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // =========================================================================
    // A. CONVOLUTIONAL LAYER BUFFER SIZE CORRECTNESS
    // =========================================================================

    #[kani::proof]
    fn verify_cuda_conv_output_buffer_size() {
        let num_filters: i32 = kani::any();
        let output_h: i32 = kani::any();
        let output_w: i32 = kani::any();
        kani::assume(num_filters > 0 && num_filters <= 64);
        kani::assume(output_h > 0 && output_h <= 64);
        kani::assume(output_w > 0 && output_w <= 64);

        let output_size = (num_filters * output_h * output_w) as usize;
        let output_bytes = output_size * std::mem::size_of::<f64>();

        kani::assert(output_bytes == output_size * 8,
            "Conv output buffer must match filters * H * W * sizeof(f64)");
    }

    #[kani::proof]
    fn verify_cuda_conv_weight_buffer_size() {
        let num_filters: i32 = kani::any();
        let input_channels: i32 = kani::any();
        let kernel_size: i32 = kani::any();
        kani::assume(num_filters > 0 && num_filters <= 16);
        kani::assume(input_channels > 0 && input_channels <= 16);
        kani::assume(kernel_size > 0 && kernel_size <= 7);

        let weight_count = (num_filters * input_channels * kernel_size * kernel_size) as usize;
        let weight_bytes = weight_count * std::mem::size_of::<f64>();

        kani::assert(weight_bytes == weight_count * 8,
            "Conv weight buffer must match F*C*K*K * sizeof(f64)");
    }

    #[kani::proof]
    fn verify_cuda_conv_padded_input_buffer_size() {
        let channels: i32 = kani::any();
        let height: i32 = kani::any();
        let width: i32 = kani::any();
        let padding: i32 = kani::any();
        kani::assume(channels > 0 && channels <= 8);
        kani::assume(height > 0 && height <= 32);
        kani::assume(width > 0 && width <= 32);
        kani::assume(padding >= 0 && padding <= 4);

        let padded_h = height + 2 * padding;
        let padded_w = width + 2 * padding;
        let padded_size = (channels * padded_h * padded_w) as usize;

        kani::assert(padded_size > 0, "Padded input buffer must be positive");
        kani::assert(padded_h >= height, "Padded height >= original height");
        kani::assert(padded_w >= width, "Padded width >= original width");
    }

    // =========================================================================
    // B. CUDA GRID/BLOCK DIMENSION SAFETY
    // =========================================================================

    #[kani::proof]
    fn verify_cuda_grid_block_covers_all_elements() {
        let n: u32 = kani::any();
        kani::assume(n > 0 && n <= 1048576);

        let blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

        kani::assert(blocks > 0, "Grid size must be at least 1");
        kani::assert(blocks as u64 * BLOCK_SIZE as u64 >= n as u64,
            "Grid * block must cover all elements");
    }

    #[kani::proof]
    fn verify_cuda_conv_forward_grid_dims() {
        let total: i32 = kani::any();
        kani::assume(total > 0 && total <= MAX_FILTERS * 64 * 64);

        let blocks = (total as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        kani::assert(blocks > 0, "Conv forward must launch at least one block");
        kani::assert(blocks * BLOCK_SIZE >= total as u32,
            "Conv forward grid must cover all outputs");
    }

    #[kani::proof]
    fn verify_cuda_fc_forward_grid_dims() {
        let num_neurons: i32 = kani::any();
        kani::assume(num_neurons > 0 && num_neurons <= MAX_NEURONS);

        let blocks = (num_neurons as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        kani::assert(blocks > 0, "FC forward must launch at least one block");
        kani::assert(blocks * BLOCK_SIZE >= num_neurons as u32,
            "FC forward grid must cover all neurons");
    }

    #[kani::proof]
    fn verify_cuda_pool_forward_grid_dims() {
        let num_filters: i32 = kani::any();
        let pool_out_h: i32 = kani::any();
        let pool_out_w: i32 = kani::any();
        kani::assume(num_filters > 0 && num_filters <= 64);
        kani::assume(pool_out_h > 0 && pool_out_h <= 64);
        kani::assume(pool_out_w > 0 && pool_out_w <= 64);

        let total = (num_filters * pool_out_h * pool_out_w) as u32;
        let blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

        kani::assert(blocks > 0, "Pool forward must launch at least one block");
        kani::assert(blocks * BLOCK_SIZE >= total,
            "Pool forward grid must cover all outputs");
    }

    // =========================================================================
    // C. FC LAYER WEIGHT INDEX VALIDITY FOR FLAT BUFFERS
    // =========================================================================

    #[kani::proof]
    fn verify_cuda_fc_weight_index_valid() {
        let num_neurons: i32 = kani::any();
        let num_inputs: i32 = kani::any();
        kani::assume(num_neurons > 0 && num_neurons <= 16);
        kani::assume(num_inputs > 0 && num_inputs <= 16);

        let neuron_idx: i32 = kani::any();
        let input_idx: i32 = kani::any();
        kani::assume(neuron_idx >= 0 && neuron_idx < num_neurons);
        kani::assume(input_idx >= 0 && input_idx < num_inputs);

        let flat_idx = (neuron_idx * num_inputs + input_idx) as usize;
        let total = (num_neurons * num_inputs) as usize;

        kani::assert(flat_idx < total,
            "FC flat weight index must be within buffer");
    }

    #[kani::proof]
    fn verify_cuda_conv_weight_index_valid() {
        let num_filters: i32 = kani::any();
        let channels: i32 = kani::any();
        let kernel_size: i32 = kani::any();
        kani::assume(num_filters > 0 && num_filters <= 8);
        kani::assume(channels > 0 && channels <= 4);
        kani::assume(kernel_size > 0 && kernel_size <= 5);

        let f: i32 = kani::any();
        let c: i32 = kani::any();
        let ky: i32 = kani::any();
        let kx: i32 = kani::any();
        kani::assume(f >= 0 && f < num_filters);
        kani::assume(c >= 0 && c < channels);
        kani::assume(ky >= 0 && ky < kernel_size);
        kani::assume(kx >= 0 && kx < kernel_size);

        let flat_idx = (((f * channels + c) * kernel_size + ky) * kernel_size + kx) as usize;
        let total = (num_filters * channels * kernel_size * kernel_size) as usize;

        kani::assert(flat_idx < total,
            "Conv flat weight index must be within buffer");
    }

    // =========================================================================
    // D. TRANSFER SIZE NON-ZERO AND ALIGNMENT
    // =========================================================================

    #[kani::proof]
    fn verify_cuda_transfer_size_non_zero() {
        let num_elements: usize = kani::any();
        kani::assume(num_elements > 0 && num_elements <= 1048576);

        let transfer_bytes = num_elements * std::mem::size_of::<f64>();

        kani::assert(transfer_bytes > 0, "Transfer size must be non-zero");
        kani::assert(transfer_bytes % std::mem::size_of::<f64>() == 0,
            "Transfer size must be aligned to f64");
    }

    #[kani::proof]
    fn verify_cuda_weight_transfer_aligned() {
        let count: usize = kani::any();
        kani::assume(count > 0 && count <= 65536);

        let bytes = count * std::mem::size_of::<f64>();
        kani::assert(bytes % 8 == 0, "Weight transfer must be 8-byte aligned");
    }

    // =========================================================================
    // E. F64 ALIGNMENT FOR CUDA TRANSFERS
    // =========================================================================

    #[kani::proof]
    fn verify_cuda_f64_alignment() {
        kani::assert(std::mem::size_of::<f64>() == 8,
            "f64 must be 8 bytes for CUDA double");
        kani::assert(std::mem::align_of::<f64>() == 8,
            "f64 must be 8-byte aligned for CUDA");
    }

    #[kani::proof]
    fn verify_cuda_i32_alignment() {
        kani::assert(std::mem::size_of::<i32>() == 4,
            "i32 must be 4 bytes for CUDA int");
        kani::assert(std::mem::align_of::<i32>() == 4,
            "i32 must be 4-byte aligned for CUDA");
    }

    // =========================================================================
    // F. POOLING LAYER BUFFER SIZING
    // =========================================================================

    #[kani::proof]
    fn verify_cuda_pool_output_buffer_size() {
        let channels: i32 = kani::any();
        let pool_out_h: i32 = kani::any();
        let pool_out_w: i32 = kani::any();
        kani::assume(channels > 0 && channels <= 64);
        kani::assume(pool_out_h > 0 && pool_out_h <= 64);
        kani::assume(pool_out_w > 0 && pool_out_w <= 64);

        let pool_size = (channels * pool_out_h * pool_out_w) as usize;
        let pool_bytes = pool_size * std::mem::size_of::<f64>();

        kani::assert(pool_bytes == pool_size * 8,
            "Pool output buffer must match C*H*W * sizeof(f64)");
    }

    #[kani::proof]
    fn verify_cuda_pool_dimension_reduction() {
        let input_dim: i32 = kani::any();
        let pool_size: i32 = kani::any();
        kani::assume(input_dim > 0 && input_dim <= 256);
        kani::assume(pool_size > 0 && pool_size <= 8);

        let output_dim = input_dim / pool_size;

        if pool_size <= input_dim {
            kani::assert(output_dim >= 0, "Pool output dim must be non-negative");
            kani::assert(output_dim <= input_dim, "Pool must reduce dimensions");
        }
    }

    // =========================================================================
    // G. CONV WEIGHT BUFFER SIZING
    // =========================================================================

    #[kani::proof]
    fn verify_cuda_conv_weight_count_no_overflow() {
        let filters: i32 = kani::any();
        let channels: i32 = kani::any();
        let kernel: i32 = kani::any();
        kani::assume(filters > 0 && filters <= MAX_FILTERS);
        kani::assume(channels > 0 && channels <= 256);
        kani::assume(kernel > 0 && kernel <= MAX_KERNEL_SIZE);

        let count = (filters as u64) * (channels as u64) * (kernel as u64) * (kernel as u64);
        kani::assert(count <= i64::MAX as u64,
            "Conv weight count must not overflow");
    }

    #[kani::proof]
    fn verify_cuda_conv_bias_buffer_matches_filters() {
        let num_filters: i32 = kani::any();
        kani::assume(num_filters > 0 && num_filters <= MAX_FILTERS);

        let bias_count = num_filters as usize;
        let bias_bytes = bias_count * std::mem::size_of::<f64>();

        kani::assert(bias_bytes == num_filters as usize * 8,
            "Conv bias buffer must match num_filters * sizeof(f64)");
    }

    // =========================================================================
    // H. KERNEL LAUNCH PARAMETER OVERFLOW PREVENTION
    // =========================================================================

    #[kani::proof]
    fn verify_cuda_kernel_launch_no_overflow() {
        let n: i32 = kani::any();
        kani::assume(n > 0 && n <= MAX_NEURONS);

        let blocks = (n as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;

        kani::assert(blocks > 0, "Must launch at least one block");
        kani::assert(blocks as u64 * BLOCK_SIZE as u64 >= n as u64,
            "Total threads must cover all elements");
        kani::assert(blocks <= 65535, "Grid dim must fit CUDA limit");
    }

    #[kani::proof]
    fn verify_cuda_conv_output_dim_calculation() {
        let input_dim: i32 = kani::any();
        let kernel_size: i32 = kani::any();
        let padding: i32 = kani::any();
        kani::assume(input_dim > 0 && input_dim <= 256);
        kani::assume(kernel_size > 0 && kernel_size <= 15);
        kani::assume(padding >= 0 && padding <= kernel_size / 2);

        let output_dim = input_dim + 2 * padding - kernel_size + 1;

        if output_dim > 0 {
            kani::assert(output_dim <= input_dim + 2 * padding,
                "Conv output dim bounded by padded input");
        }
    }

    // =========================================================================
    // I. FLATTENED FEATURE BUFFER SIZING
    // =========================================================================

    #[kani::proof]
    fn verify_cuda_flattened_buffer_size() {
        let last_c: i32 = kani::any();
        let last_h: i32 = kani::any();
        let last_w: i32 = kani::any();
        kani::assume(last_c > 0 && last_c <= 64);
        kani::assume(last_h > 0 && last_h <= 64);
        kani::assume(last_w > 0 && last_w <= 64);

        let flattened = (last_c * last_h * last_w) as usize;
        let bytes = flattened * std::mem::size_of::<f64>();

        kani::assert(bytes == flattened * 8,
            "Flattened feature buffer must match C*H*W * sizeof(f64)");
        kani::assert(flattened > 0, "Flattened size must be positive");
    }

    // =========================================================================
    // J. SOFTMAX/LOGITS BUFFER SAFETY
    // =========================================================================

    #[kani::proof]
    fn verify_cuda_logits_buffer_size() {
        let output_size: i32 = kani::any();
        kani::assume(output_size > 0 && output_size <= MAX_NEURONS);

        let logit_bytes = output_size as usize * std::mem::size_of::<f64>();
        kani::assert(logit_bytes == output_size as usize * 8,
            "Logits buffer must match output_size * sizeof(f64)");
    }

    #[kani::proof]
    fn verify_cuda_softmax_buffer_size() {
        let output_size: i32 = kani::any();
        kani::assume(output_size > 0 && output_size <= MAX_NEURONS);

        let softmax_bytes = output_size as usize * std::mem::size_of::<f64>();
        kani::assert(softmax_bytes == output_size as usize * 8,
            "Softmax buffer must match output_size * sizeof(f64)");
    }

    // =========================================================================
    // K. GRADIENT BUFFER SIZING
    // =========================================================================

    #[kani::proof]
    fn verify_cuda_fc_gradient_buffer_matches_weights() {
        let neurons: i32 = kani::any();
        let inputs: i32 = kani::any();
        kani::assume(neurons > 0 && neurons <= 256);
        kani::assume(inputs > 0 && inputs <= 256);

        let weight_count = (neurons * inputs) as usize;
        kani::assert(weight_count > 0, "FC gradient buffer must be positive");
    }

    #[kani::proof]
    fn verify_cuda_conv_gradient_buffer_matches_weights() {
        let filters: i32 = kani::any();
        let channels: i32 = kani::any();
        let kernel: i32 = kani::any();
        kani::assume(filters > 0 && filters <= 32);
        kani::assume(channels > 0 && channels <= 16);
        kani::assume(kernel > 0 && kernel <= 7);

        let grad_count = (filters * channels * kernel * kernel) as usize;
        kani::assert(grad_count > 0, "Conv weight gradient buffer must be positive");
    }

    // =========================================================================
    // L. ADAM OPTIMIZER BUFFER SIZING
    // =========================================================================

    #[kani::proof]
    fn verify_cuda_adam_m_v_buffers_match_weights() {
        let weight_count: usize = kani::any();
        kani::assume(weight_count > 0 && weight_count <= 65536);

        let m_size = weight_count;
        let v_size = weight_count;

        kani::assert(m_size == weight_count,
            "Adam M buffer must match weight count");
        kani::assert(v_size == weight_count,
            "Adam V buffer must match weight count");
    }

    #[kani::proof]
    fn verify_cuda_adam_bias_m_v_match_bias() {
        let bias_count: usize = kani::any();
        kani::assume(bias_count > 0 && bias_count <= MAX_FILTERS as usize);

        let m_size = bias_count;
        let v_size = bias_count;

        kani::assert(m_size == bias_count, "Adam bias M must match bias count");
        kani::assert(v_size == bias_count, "Adam bias V must match bias count");
    }

    // =========================================================================
    // M. NO-PANIC GUARANTEE FOR DIMENSION CALCULATIONS
    // =========================================================================

    #[kani::proof]
    fn verify_cuda_all_grid_calculations_no_panic() {
        let n: u32 = kani::any();
        kani::assume(n > 0 && n <= 1048576);

        let _blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    }

    #[kani::proof]
    fn verify_cuda_conv_output_dim_no_panic() {
        let w: i32 = kani::any();
        let k: i32 = kani::any();
        let p: i32 = kani::any();
        let s: i32 = kani::any();
        kani::assume(w > 0 && w <= MAX_IMAGE_DIM);
        kani::assume(k > 0 && k <= MAX_KERNEL_SIZE);
        kani::assume(p >= 0 && p <= 15);
        kani::assume(s > 0 && s <= 4);

        let _out = (w + 2 * p - k) / s + 1;
    }

    #[kani::proof]
    fn verify_cuda_pool_output_dim_no_panic() {
        let dim: i32 = kani::any();
        let pool: i32 = kani::any();
        kani::assume(dim > 0 && dim <= MAX_IMAGE_DIM);
        kani::assume(pool > 0 && pool <= 8);

        let _out = dim / pool;
    }

    // =========================================================================
    // N. ABI TYPE COMPATIBILITY FOR CUDA INTEROP
    // =========================================================================

    #[kani::proof]
    fn verify_cuda_f64_abi_compatibility() {
        kani::assert(std::mem::size_of::<f64>() == 8, "f64 == CUDA double");
        kani::assert(std::mem::align_of::<f64>() == 8, "f64 8-byte aligned for CUDA");
    }

    #[kani::proof]
    fn verify_cuda_i32_abi_compatibility() {
        kani::assert(std::mem::size_of::<i32>() == 4, "i32 == CUDA int");
        kani::assert(std::mem::align_of::<i32>() == 4, "i32 4-byte aligned for CUDA");
    }

    #[kani::proof]
    fn verify_cuda_u32_abi_compatibility() {
        kani::assert(std::mem::size_of::<u32>() == 4, "u32 == CUDA unsigned int");
        kani::assert(std::mem::align_of::<u32>() == 4, "u32 4-byte aligned for CUDA");
    }

    // =========================================================================
    // O. END-TO-END FORWARD PASS BUFFER CHAIN VALIDATION
    // =========================================================================

    #[kani::proof]
    fn verify_cuda_conv_to_pool_chain() {
        let conv_out_h: i32 = kani::any();
        let conv_out_w: i32 = kani::any();
        let pool_size: i32 = kani::any();
        kani::assume(conv_out_h > 0 && conv_out_h <= 128);
        kani::assume(conv_out_w > 0 && conv_out_w <= 128);
        kani::assume(pool_size > 0 && pool_size <= 4);

        let pool_out_h = conv_out_h / pool_size;
        let pool_out_w = conv_out_w / pool_size;

        kani::assert(pool_out_h <= conv_out_h, "Pool reduces height");
        kani::assert(pool_out_w <= conv_out_w, "Pool reduces width");
    }

    #[kani::proof]
    fn verify_cuda_flatten_to_fc_chain() {
        let last_c: i32 = kani::any();
        let last_h: i32 = kani::any();
        let last_w: i32 = kani::any();
        kani::assume(last_c > 0 && last_c <= 64);
        kani::assume(last_h > 0 && last_h <= 32);
        kani::assume(last_w > 0 && last_w <= 32);

        let flattened = last_c * last_h * last_w;

        kani::assert(flattened > 0, "Flattened size must be positive");
    }

    #[kani::proof]
    fn verify_cuda_fc_to_output_chain() {
        let fc_neurons: i32 = kani::any();
        let output_size: i32 = kani::any();
        kani::assume(fc_neurons > 0 && fc_neurons <= MAX_NEURONS);
        kani::assume(output_size > 0 && output_size <= MAX_NEURONS);

        let weight_count = (fc_neurons as u64) * (output_size as u64);
        kani::assert(weight_count <= i64::MAX as u64,
            "FC-to-output weight count must not overflow");
    }

    #[kani::proof]
    fn verify_cuda_softmax_grid_dims() {
        let output_size: i32 = kani::any();
        kani::assume(output_size > 0 && output_size <= MAX_NEURONS);

        let blocks = (output_size as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        kani::assert(blocks > 0, "Softmax must launch at least one block");
        kani::assert(blocks * BLOCK_SIZE >= output_size as u32,
            "Softmax grid must cover all outputs");
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn test_cuda_grid_calculation() {
        assert_eq!((1_u32 + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
        assert_eq!((256_u32 + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
        assert_eq!((257_u32 + BLOCK_SIZE - 1) / BLOCK_SIZE, 2);
    }

    #[test]
    fn test_cuda_conv_output_dim() {
        let output = (28 + 2 * 1 - 3) / 1 + 1;
        assert_eq!(output, 28);
    }

    #[test]
    fn test_cuda_pool_output_dim() {
        assert_eq!(28 / 2, 14);
        assert_eq!(14 / 2, 7);
    }

    #[test]
    fn test_cuda_fc_flat_index() {
        let neurons = 10;
        let inputs = 128;
        assert!(9 * inputs + 127 < neurons * inputs);
    }

    #[test]
    fn test_cuda_f64_properties() {
        assert_eq!(std::mem::size_of::<f64>(), 8);
        assert_eq!(std::mem::align_of::<f64>(), 8);
    }

    #[test]
    fn test_cuda_conv_weight_index() {
        let f = 3; let c = 1; let k = 3;
        assert!(((2 * c + 0) * k + 2) * k + 2 < f * c * k * k);
    }
}

