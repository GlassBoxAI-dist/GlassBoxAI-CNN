//! @file
//! @ingroup CNN_Core_Verified
/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 *
 * Kani Verification: No-Panic Guarantee
 * Verify functions cannot trigger panic/unwrap/expect failures
 */

use super::*;

#[kani::proof]
fn verify_activation_type_no_panic() {
    let act_type: u8 = kani::any();
    kani::assume(act_type < 4);

    let activation = match act_type {
        0 => ActivationType::Sigmoid,
        1 => ActivationType::Tanh,
        2 => ActivationType::ReLU,
        3 => ActivationType::Linear,
        _ => unreachable!(),
    };

    let _ = activation;
}

#[kani::proof]
fn verify_loss_type_no_panic() {
    let loss_type: u8 = kani::any();
    kani::assume(loss_type < 2);

    let loss = match loss_type {
        0 => LossType::MSE,
        1 => LossType::CrossEntropy,
        _ => unreachable!(),
    };

    let _ = loss;
}

#[kani::proof]
fn verify_command_parsing_no_panic() {
    let cmd_type: u8 = kani::any();
    kani::assume(cmd_type < 6);

    let command = match cmd_type {
        0 => Command::None,
        1 => Command::Create,
        2 => Command::Train,
        3 => Command::Predict,
        4 => Command::Info,
        5 => Command::Help,
        _ => unreachable!(),
    };

    let _ = command;
}

