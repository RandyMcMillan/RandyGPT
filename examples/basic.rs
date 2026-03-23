use candle_core::{Device, Tensor, DType};
use candle_nn::{/*Linear, */Module, VarMap, VarBuilder};

fn main() -> Result<(), candle_core::Error> {
    // 1. Select the device (CPU or GPU if available)
    let device = Device::Cpu;

    // 2. Create a Variable Map to manage our weights/biases
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    // 3. Define a Linear Layer: 10 input features -> 5 output features
    // This automatically initializes weights and biases
    let in_features = 10;
    let out_features = 5;
    let linear_layer = candle_nn::linear(in_features, out_features, vb.pp("layer1"))?;

    // 4. Create dummy input data (Batch size of 2, 10 features each)
    let input = Tensor::randn(0f32, 1f32, (2, 10), &device)?;

    // 5. Run the forward pass
    let output = linear_layer.forward(&input)?;

    println!("Input shape:  {:?}", input.shape());
    println!("Output shape: {:?}", output.shape());
    println!("Output Tensor:\n{}", output);

    Ok(())
}
