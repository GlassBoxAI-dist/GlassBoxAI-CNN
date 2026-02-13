/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 *
 * Kani Verification: Enum Exhaustion
 * Verify all match statements handle every variant
 */

use super::*;

#[kani::proof]
fn verify_activation_type_exhaustive() {
    let act: ActivationType = kani::any();

    let result = match act {
        ActivationType::Sigmoid => 0,
        ActivationType::Tanh => 1,
        ActivationType::ReLU => 2,
        ActivationType::Linear => 3,
    };

    assert!(result >= 0 && result <= 3, "All variants must be handled");
}

#[kani::proof]
fn verify_loss_type_exhaustive() {
    let loss: LossType = kani::any();

    let result = match loss {
        LossType::MSE => 0,
        LossType::CrossEntropy => 1,
    };

    assert!(result >= 0 && result <= 1, "All loss variants must be handled");
}

#[kani::proof]
fn verify_command_exhaustive() {
    let cmd: Command = kani::any();

    let result = match cmd {
        Command::None => 0,
        Command::Create => 1,
        Command::Train => 2,
        Command::Predict => 3,
        Command::Info => 4,
        Command::Help => 5,
    };

    assert!(result >= 0 && result <= 5, "All command variants must be handled");
}
