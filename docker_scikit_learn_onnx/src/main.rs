use ort::session;
use ndarray::Array2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create session
    let session = session::Session::builder()?
        .with_optimization_level(session::builder::GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file("toy_model_generation/rf_model.onnx")?;

    // Create input data - adjust dimensions to match your 4 features
    let input_array: Array2<f32> = Array2::from_shape_vec(
        (2, 4), // 2 samples, 4 features
        vec![1.2, 0.5, 3.4, 2.0, 2.3, 1.1, 4.1, 1.5] // Example values
    )?;

    // Get input name from model
    let input_name = session.inputs[0].name.clone();
    
    // Run inference
    let outputs =     session.run(ort::inputs![input_name => input_array]?)?;


    // Get the first output
    let output = outputs[0].try_extract_tensor::<f32>()?;
    println!("Predictions: {:?}", output);

    Ok(())
}