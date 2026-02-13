# Kani Verification Test Suite - CISA Hardening Compliance

This document describes the Kani formal verification tests implemented for the Facaded CNN project to meet CISA "Secure by Design" standards.

## Installation

```bash
# Install Kani verifier
cargo install --locked kani-verifier

# Setup Kani (downloads required toolchain)
kani setup
```

## Running Tests

### CLI Version
```bash
cd facaded_rust_cuda
cargo kani
```

### GUI Version
```bash
cd facaded_rust_cuda/gui
cargo kani
```

## Verification Categories

The test suite covers 19 security verification categories:

### 1. Strict Bound Checks
- `verify_parse_args_bounds` - CLI argument parsing bounds
- `verify_conv_filter_indexing` - Convolutional filter array access
- `verify_output_indexing` - Output tensor indexing
- `verify_dataset_indexing` - Dataset sample access (GUI)
- `verify_image_buffer_indexing` - Image buffer access (GUI)

### 2. Pointer Validity Proofs
- `verify_slice_to_ptr_validity` - Slice-to-pointer conversions
- `verify_vec_ptr_validity` - Vector pointer validity (GUI)

### 3. No-Panic Guarantee
- `verify_activation_type_no_panic` - ActivationType enum handling
- `verify_loss_type_no_panic` - LossType enum handling
- `verify_command_parsing_no_panic` - Command enum handling (CLI)
- `verify_progress_calculation_no_panic` - Progress percentage (GUI)

### 4. Integer Overflow Prevention
- `verify_weight_size_no_overflow` - Weight buffer size calculation
- `verify_output_dimension_no_overflow` - Output dimension calculation
- `verify_adam_timestep_no_overflow` - Adam optimizer timestep
- `verify_image_size_no_overflow` - Image buffer size (GUI)
- `verify_sample_count_no_overflow` - Dataset sample count (GUI)

### 5. Division-by-Zero Exclusion
- `verify_launch_config_no_div_zero` - CUDA launch configuration
- `verify_pooling_stride_no_div_zero` - Pooling stride calculation
- `verify_batch_average_no_div_zero` - Batch average loss
- `verify_accuracy_no_div_zero` - Accuracy calculation (GUI)
- `verify_progress_no_div_zero` - Progress calculation (GUI)

### 6. Global State Consistency
- `verify_is_training_state_consistency` - Training mode state
- `verify_stop_requested_state` - Stop training flag (GUI)
- `verify_network_exists_state` - Network initialization state (GUI)

### 7. Deadlock-Free Logic
- `verify_no_lock_hierarchy_violation` - Arc reference counting
- `verify_no_recursive_borrow` - Pin borrow semantics (GUI)

### 8. Input Sanitization Bounds
- `verify_conv_layer_iteration_bounded` - Conv layer loop bounds
- `verify_fc_layer_iteration_bounded` - FC layer loop bounds
- `verify_epoch_iteration_bounded` - Training epoch bounds
- `verify_sample_loop_bounded` - Sample iteration bounds (GUI)
- `verify_class_loop_bounded` - Class iteration bounds (GUI)

### 9. Result Coverage Audit
- `verify_result_handling_coverage` - Result<T, E> handling
- `verify_network_creation_result_handled` - Network creation (GUI)
- `verify_train_step_result_handled` - Training step results (GUI)

### 10. Memory Leak Prevention
- `verify_vec_allocation_bounds` - Vector allocation limits
- `verify_dataset_clear_releases_memory` - Dataset cleanup (GUI)
- `verify_image_vec_bounded_allocation` - Image allocation (GUI)

### 11. Constant-Time Execution
- `verify_activation_constant_time` - ReLU timing analysis
- `verify_argmax_no_secret_timing` - Argmax comparison (GUI)

### 12. State Machine Integrity
- `verify_training_state_transitions` - Training state changes
- `verify_training_state_machine` - Train prerequisites (GUI)
- `verify_prediction_requires_network` - Prediction prerequisites (GUI)

### 13. Enum Exhaustion
- `verify_activation_type_exhaustive` - All ActivationType variants
- `verify_loss_type_exhaustive` - All LossType variants
- `verify_command_exhaustive` - All Command variants (CLI)
- `verify_activation_enum_exhaustive` - GUI activation handling
- `verify_loss_enum_exhaustive` - GUI loss handling

### 14. Floating-Point Sanity
- `verify_relu_no_nan` - ReLU output validity
- `verify_softmax_denominator_positive` - Softmax denominator
- `verify_gradient_clipping` - Gradient clip bounds
- `verify_softmax_output_valid` - Softmax probability (GUI)
- `verify_loss_finite` - Loss value finiteness (GUI)
- `verify_radius_calculation_valid` - Synthetic data radius (GUI)

### 15. Resource Limit Compliance
- `verify_weight_allocation_limit` - Weight memory budget
- `verify_fc_layer_allocation_limit` - FC layer memory budget
- `verify_output_buffer_size_limit` - Output buffer memory
- `verify_dataset_size_limit` - Dataset memory budget (GUI)
- `verify_training_log_bounded` - Log size limit (GUI)
- `verify_gui_property_limits` - Property value bounds (GUI)

### 16. FFI C Boundary Safety
Located in `src/kani/ffi_c_boundary.rs` — 43 Kani proofs covering:
- Signed-to-unsigned conversion safety (c_int → usize)
- Output buffer overflow prevention
- NaN/Infinity parameter rejection (learning rate, gradient clip, dropout)
- Enum variant validation (ActivationType, LossType)
- CnnConfig array length bounds
- Error string NUL-byte sanitization
- No-panic guarantee for all validators
- ABI type compatibility (f64, i32)
- Input array NaN/Inf detection
- Resource limits at FFI boundary
- Setter value validation
- End-to-end pipeline validation (predict, train, create)

### 17. CUDA Backend FFI Safety
Located in `src/kani/ffi_cuda_boundary.rs` — 42 Kani proofs + unit tests across 15 categories (A-O):
- Conv/Pool/FC layer buffer size correctness
- CUDA grid/block dimension safety for all kernel types
- Weight index validity for flat buffers (FC and Conv 4D)
- Transfer size non-zero and alignment (f64/8-byte)
- Pooling layer buffer sizing and dimension reduction
- Kernel launch parameter overflow prevention
- Flattened feature, softmax/logits, gradient buffer sizing
- Adam optimizer M/V buffer sizing
- End-to-end forward pass buffer chain (conv→pool→flatten→FC→output)

### 18. OpenCL Backend FFI Safety
Located in `src/kani/ffi_opencl_boundary.rs` — 42 Kani proofs + unit tests across 15 categories (A-O):
- Conv/Pool/FC layer buffer size for clCreateBuffer
- OpenCL global/local work size safety (divisible, covers all items)
- Weight index validity for flat OpenCL buffers
- Transfer alignment for clEnqueueRead/WriteBuffer
- Work group size power-of-two property
- Same buffer chain validation as CUDA, adapted for OpenCL

### 19. Polyglot FFI Safety
Located in `src/kani/ffi_polyglot.rs` — 48 Kani proofs + unit tests across 15 categories (A-O):
- CnnConfig struct field validation (width, height, channels, output, LR, gradient clip)
- CnnError enum repr(C) ABI compatibility
- CnnActivationType/CnnLossType enum roundtrip safety
- Handle lifecycle safety (create/destroy null checks)
- Null pointer rejection across all C API functions
- String parameter NUL-termination safety
- Output buffer capacity contracts
- Thread-local error storage safety
- Version string NUL-termination
- Batch norm flag bool mapping
- Config array pointer validation
- ABI layout for repr(C) types (c_int, c_double, pointer)
- End-to-end polyglot call chains (create, predict, train, setter)

## Expected Results

When Kani verification passes, you should see output similar to:

```
SUMMARY:
 ** 0 of N proofs failed
```

Each proof function should complete with `VERIFICATION:- SUCCESSFUL`.

## Notes

- Kani proofs use symbolic execution to explore all possible input states
- The `#[kani::unwind(N)]` attribute bounds loop unrolling for termination
- `kani::any()` generates symbolic values that Kani explores exhaustively
- `kani::assume()` constrains symbolic values to valid ranges

## License

MIT License - Copyright (c) 2025 Matthew Abbott
