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

use facaded_cnn_cuda::{
    Command, ConvolutionalNeuralNetworkCUDA,
    activation_to_str, loss_to_str, parse_activation, parse_command, parse_loss,
};
use serde::{Deserialize, Serialize};
use std::env;
use std::fs::File;
use std::io::BufReader;

#[derive(Serialize, Deserialize)]
struct ModelInfoJson {
    input_width: i32,
    input_height: i32,
    input_channels: i32,
    output_size: i32,
    learning_rate: f64,
    gradient_clip: f64,
    activation: String,
    output_activation: String,
    loss_type: String,
}

fn parse_int_list(s: &str) -> Vec<i32> {
    s.split(',')
        .filter_map(|t| t.trim().parse().ok())
        .collect()
}

fn get_arg_value(args: &[String], arg: &str, default: &str) -> String {
    let search_key = format!("{}=", arg);
    for i in 0..args.len() {
        if args[i].starts_with(&search_key) {
            return args[i][search_key.len()..].to_string();
        }
        if args[i] == arg && i + 1 < args.len() {
            return args[i + 1].clone();
        }
    }
    default.to_string()
}

fn print_help() {
    println!("Commands:");
    println!("  create       Create a new CNN model and save to JSON");
    println!("  train        Train an existing model with data from JSON");
    println!("  predict      Make predictions with a trained model from JSON");
    println!("  info         Display model information from JSON");
    println!("  export-onnx  Export model to ONNX binary format");
    println!("  import-onnx  Import model from ONNX binary format");
    println!("  help         Show this help message");
    println!();
    println!("Create Options:");
    println!("  --input-w=N            Input width (required)");
    println!("  --input-h=N            Input height (required)");
    println!("  --input-c=N            Input channels (required)");
    println!("  --conv=N,N,...         Conv filters (required)");
    println!("  --kernels=N,N,...      Kernel sizes (required)");
    println!("  --pools=N,N,...        Pool sizes (required)");
    println!("  --fc=N,N,...           FC layer sizes (required)");
    println!("  --output=N             Output layer size (required)");
    println!("  --save=FILE.json       Save model to JSON file (required)");
    println!("  --lr=VALUE             Learning rate (default: 0.001)");
    println!("  --hidden-act=TYPE      sigmoid|tanh|relu|linear (default: relu)");
    println!("  --output-act=TYPE      sigmoid|tanh|relu|linear (default: linear)");
    println!("  --loss=TYPE            mse|crossentropy (default: mse)");
    println!("  --clip=VALUE           Gradient clipping (default: 5.0)");
    println!("  --batch-norm           Enable batch normalization");
    println!();
    println!("Train Options:");
    println!("  --model=FILE.json      Load model from JSON file (required)");
    println!("  --data=FILE.csv        Training data CSV file (required)");
    println!("  --epochs=N             Number of epochs (required)");
    println!("  --save=FILE.json       Save trained model to JSON (required)");
    println!("  --batch-size=N         Batch size (default: 32)");
    println!();
    println!("Predict Options:");
    println!("  --model=FILE.json      Load model from JSON file (required)");
    println!("  --data=FILE.csv        Input data CSV file (required)");
    println!("  --output=FILE.csv      Save predictions to CSV file (required)");
    println!();
    println!("Info Options:");
    println!("  --model=FILE.json      Load model from JSON file (required)");
    println!();
    println!("Export ONNX Options:");
    println!("  --model=FILE.json      Load model from JSON file (required)");
    println!("  --output=FILE.onnx     Save model to ONNX binary file (required)");
    println!();
    println!("Import ONNX Options:");
    println!("  --input=FILE.onnx      Load model from ONNX binary file (required)");
    println!("  --save=FILE.json       Save model to JSON file (required)");
    println!();
    println!("Examples:");
    println!("  cnn create --input-w=28 --input-h=28 --input-c=1 --conv=32,64 --kernels=3,3 --pools=2,2 --fc=128 --output=10 --save=model.json");
    println!("  cnn create --input-w=28 --input-h=28 --input-c=1 --conv=32,64 --kernels=3,3 --pools=2,2 --fc=128 --output=10 --batch-norm --save=model.json");
    println!("  cnn train --model=model.json --data=data.csv --epochs=50 --save=model_trained.json");
    println!("  cnn predict --model=model_trained.json --data=test.csv --output=predictions.csv");
    println!("  cnn info --model=model.json");
    println!("  cnn export-onnx --model=model.json --output=model.onnx");
    println!("  cnn import-onnx --input=model.onnx --save=model.json");
}

fn print_model_info(model_file: &str) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::open(model_file)?;
    let reader = BufReader::new(file);
    let model: ModelInfoJson = serde_json::from_reader(reader)?;

    println!();
    println!("=================================================================");
    println!("  Model Information:  {}", model_file);
    println!("=================================================================");
    println!();
    println!("Architecture:");
    println!("Input: {}x{}x{}", model.input_width, model.input_height, model.input_channels);
    println!("Output size: {}", model.output_size);
    println!();
    println!("Training Parameters:");
    println!("Learning rate: {:.6}", model.learning_rate);
    println!("Gradient clip: {:.2}", model.gradient_clip);
    println!("activation: {}", model.activation);
    println!("output_activation: {}", model.output_activation);
    println!("loss_type: {}", model.loss_type);
    println!();

    Ok(())
}

fn handle_create(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let save_file = get_arg_value(args, "--save", "");
    let input_w_str = get_arg_value(args, "--input-w", "");
    let input_h_str = get_arg_value(args, "--input-h", "");
    let input_c_str = get_arg_value(args, "--input-c", "");
    let conv_filters = get_arg_value(args, "--conv", "");
    let kernels = get_arg_value(args, "--kernels", "");
    let pools = get_arg_value(args, "--pools", "");
    let fc_layers = get_arg_value(args, "--fc", "");
    let output_size_str = get_arg_value(args, "--output", "");

    if save_file.is_empty() {
        eprintln!("Error: --save argument is required for create command");
        return Ok(());
    }
    if input_w_str.is_empty() {
        eprintln!("Error: --input-w argument is required for create command");
        return Ok(());
    }
    if input_h_str.is_empty() {
        eprintln!("Error: --input-h argument is required for create command");
        return Ok(());
    }
    if input_c_str.is_empty() {
        eprintln!("Error: --input-c argument is required for create command");
        return Ok(());
    }
    if conv_filters.is_empty() {
        eprintln!("Error: --conv argument is required for create command");
        return Ok(());
    }
    if kernels.is_empty() {
        eprintln!("Error: --kernels argument is required for create command");
        return Ok(());
    }
    if pools.is_empty() {
        eprintln!("Error: --pools argument is required for create command");
        return Ok(());
    }
    if fc_layers.is_empty() {
        eprintln!("Error: --fc argument is required for create command");
        return Ok(());
    }
    if output_size_str.is_empty() {
        eprintln!("Error: --output argument is required for create command");
        return Ok(());
    }

    let input_w: i32 = input_w_str.parse()?;
    let input_h: i32 = input_h_str.parse()?;
    let input_c: i32 = input_c_str.parse()?;
    let output_size: i32 = output_size_str.parse()?;

    let hidden_act_str = get_arg_value(args, "--hidden-act", "relu");
    let output_act_str = get_arg_value(args, "--output-act", "linear");
    let loss_str = get_arg_value(args, "--loss", "mse");
    let lr: f64 = get_arg_value(args, "--lr", "0.001").parse()?;
    let clip: f64 = get_arg_value(args, "--clip", "5.0").parse()?;
    let use_batch_norm = args.iter().any(|a| a == "--batch-norm");

    let conv_filter_vec = parse_int_list(&conv_filters);
    let kernel_vec = parse_int_list(&kernels);
    let pool_vec = parse_int_list(&pools);
    let fc_vec = parse_int_list(&fc_layers);

    let hidden_act = parse_activation(&hidden_act_str);
    let output_act = parse_activation(&output_act_str);
    let loss_type = parse_loss(&loss_str);

    println!("Creating CNN model...");
    println!("  Input: {}x{}x{}", input_w, input_h, input_c);
    println!("  Conv filters: {:?}", conv_filter_vec);
    println!("  Kernel sizes: {:?}", kernel_vec);
    println!("  Pool sizes: {:?}", pool_vec);
    println!("  FC layers: {:?}", fc_vec);
    println!("  Output size: {}", output_size);
    println!("  Hidden activation: {}", activation_to_str(hidden_act));
    println!("  Output activation: {}", activation_to_str(output_act));
    println!("  Loss function: {}", loss_to_str(loss_type));
    println!("  Learning rate: {:.6}", lr);
    println!("  Gradient clip: {:.2}", clip);
    println!("  Batch normalization: {}", if use_batch_norm { "enabled" } else { "disabled" });

    let mut cnn = ConvolutionalNeuralNetworkCUDA::new(
        input_w, input_h, input_c,
        &conv_filter_vec, &kernel_vec, &pool_vec, &fc_vec,
        output_size, hidden_act, output_act,
        loss_type, lr, clip,
    )?;

    if use_batch_norm {
        cnn.initialize_batch_norm();
    }

    cnn.save_to_json(&save_file)?;
    println!("Created CNN model");
    println!("Model saved to: {}", save_file);

    Ok(())
}

fn handle_train(args: &[String]) {
    let model_file = get_arg_value(args, "--model", "");
    let data_file = get_arg_value(args, "--data", "");
    let epochs_str = get_arg_value(args, "--epochs", "");
    let save_file = get_arg_value(args, "--save", "");
    let batch_size: i32 = get_arg_value(args, "--batch-size", "32").parse().unwrap_or(32);

    if model_file.is_empty() {
        eprintln!("Error: --model argument is required for train command");
        return;
    }
    if data_file.is_empty() {
        eprintln!("Error: --data argument is required for train command");
        return;
    }
    if epochs_str.is_empty() {
        eprintln!("Error: --epochs argument is required for train command");
        return;
    }
    if save_file.is_empty() {
        eprintln!("Error: --save argument is required for train command");
        return;
    }

    let epochs: i32 = epochs_str.parse().unwrap_or(0);

    println!("Training model...");
    println!("  Model: {}", model_file);
    println!("  Data: {}", data_file);
    println!("  Epochs: {}", epochs);
    println!("  Batch size: {}", batch_size);
    println!("  Save to: {}", save_file);
    println!();
    println!("Training not fully implemented in this CLI demo.");
    println!("To implement training:");
    println!("  1. Load CSV data from {}", data_file);
    println!("  2. Load model from {}", model_file);
    println!("  3. Run training loop with train_step() for {} epochs", epochs);
    println!("  4. Save updated model to {}", save_file);
    println!();
    println!("See the library API for complete training implementation.");
}

fn handle_predict(args: &[String]) {
    let model_file = get_arg_value(args, "--model", "");
    let data_file = get_arg_value(args, "--data", "");
    let output_file = get_arg_value(args, "--output", "");

    if model_file.is_empty() {
        eprintln!("Error: --model argument is required for predict command");
        return;
    }
    if data_file.is_empty() {
        eprintln!("Error: --data argument is required for predict command");
        return;
    }
    if output_file.is_empty() {
        eprintln!("Error: --output argument is required for predict command");
        return;
    }

    println!("Making predictions...");
    println!("  Model: {}", model_file);
    println!("  Data: {}", data_file);
    println!("  Output: {}", output_file);
    println!();
    println!("Prediction not fully implemented in this CLI demo.");
    println!("To implement prediction:");
    println!("  1. Load model from {}", model_file);
    println!("  2. Load input data from CSV file: {}", data_file);
    println!("  3. Run predict() on each input");
    println!("  4. Save predictions to CSV: {}", output_file);
    println!();
    println!("See the library API for complete prediction implementation.");
}

fn handle_export_onnx(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let model_file = get_arg_value(args, "--model", "");
    let output_file = get_arg_value(args, "--output", "");

    if model_file.is_empty() {
        eprintln!("Error: --model argument is required for export-onnx command");
        return Ok(());
    }
    if output_file.is_empty() {
        eprintln!("Error: --output argument is required for export-onnx command");
        return Ok(());
    }

    println!("Exporting model to ONNX format...");
    println!("  Model: {}", model_file);
    println!("  Output: {}", output_file);

    let cnn = ConvolutionalNeuralNetworkCUDA::load_from_json(&model_file)?;
    cnn.export_to_onnx(&output_file)?;
    println!("Model exported to: {}", output_file);

    Ok(())
}

fn handle_import_onnx(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let input_file = get_arg_value(args, "--input", "");
    let save_file = get_arg_value(args, "--save", "");

    if input_file.is_empty() {
        eprintln!("Error: --input argument is required for import-onnx command");
        return Ok(());
    }
    if save_file.is_empty() {
        eprintln!("Error: --save argument is required for import-onnx command");
        return Ok(());
    }

    println!("Importing model from ONNX format...");
    println!("  Input: {}", input_file);
    println!("  Save to: {}", save_file);

    let cnn = ConvolutionalNeuralNetworkCUDA::import_from_onnx(&input_file)?;
    cnn.save_to_json(&save_file)?;

    println!("Model imported and saved to: {}", save_file);

    Ok(())
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_help();
        return;
    }

    if args[1] == "--help" || args[1] == "-h" {
        print_help();
        return;
    }

    let cmd = parse_command(&args[1]);

    match cmd {
        Command::Help => {
            print_help();
        }
        Command::Info => {
            let model_file = get_arg_value(&args, "--model", "");
            if model_file.is_empty() {
                eprintln!("Error: --model argument required for info command");
                return;
            }
            if let Err(e) = print_model_info(&model_file) {
                eprintln!("Error: {}", e);
            }
        }
        Command::Create => {
            if let Err(e) = handle_create(&args) {
                eprintln!("Error: {}", e);
            }
        }
        Command::Train => {
            handle_train(&args);
        }
        Command::Predict => {
            handle_predict(&args);
        }
        Command::ExportOnnx => {
            if let Err(e) = handle_export_onnx(&args) {
                eprintln!("Error: {}", e);
            }
        }
        Command::ImportOnnx => {
            if let Err(e) = handle_import_onnx(&args) {
                eprintln!("Error: {}", e);
            }
        }
        Command::None => {
            eprintln!("Unknown command: '{}'", args[1]);
            eprintln!("Run 'cnn help' for usage information");
        }
    }
}
