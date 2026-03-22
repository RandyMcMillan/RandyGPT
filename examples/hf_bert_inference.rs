use anyhow::Result;
use candle_core::{Device, Tensor, DType};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

fn main() -> Result<()> {
    // 1. Setup Device (CPU or CUDA)
    let device = Device::Cpu;

    // 2. Fetch model files from Hugging Face Hub
    let api = Api::new()?;
    let repo = api.repo(Repo::new(
        "sentence-transformers/all-MiniLM-L6-v2".to_string(),
        RepoType::Model,
    ));

    let config_filename = repo.get("config.json")?;
    let tokenizer_filename = repo.get("tokenizer.json")?;
    let weights_filename = repo.get("model.safetensors")?;

    // 3. Load Config and Tokenizer
    let config = std::fs::read_to_string(config_filename)?;
    let config: Config = serde_json::from_str(&config)?;
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(anyhow::Error::msg)?;

    // 4. Load Weights using VarBuilder
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights_filename], DType::F32, &device)?
    };

    // 5. Initialize the Model
    let model = BertModel::load(vb, &config)?; 

    // 6. Prepare Input
    let sentences = vec!["The weather is lovely today."];
    let tokens = tokenizer
        .encode_batch(sentences, true)
        .map_err(anyhow::Error::msg)?;
    
    let token_ids = tokens
        .iter()
        .map(|v| v.get_ids().to_vec())
        .collect::<Vec<_>>();
    let token_ids = Tensor::new(token_ids, &device)?;
    
    let token_type_ids = token_ids.zeros_like()?;

    // 7. Run Inference
    let embeddings = model.forward(&token_ids, &token_type_ids, None)?;
    
    println!("Embeddings shape: {:?}", embeddings.shape());
    // Typically you would perform mean pooling here to get a single sentence vector
    
    Ok(())
}