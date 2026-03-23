use anyhow::Result;
use candle_core::{Device, Tensor, DType, error::Result as CResult};
use candle_nn::ops::softmax; // Added this import
use candle_nn::VarBuilder; // Re-added
use candle_transformers::models::qwen3_moe::{Config, Model as QwenModel};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;
use rand::SeedableRng; // Added this import
use std::io::{Write, BufWriter};
use std::path::PathBuf; // Re-added




// Model and file constants
const MODEL_ID: &str       = "Qwen/Qwen3-Coder-Next"; // Changed to original model ID

const SAFETENSORS_INDEX_FILE: &str = "model.safetensors.index.json";
const TOKENIZER_FILE: &str = "tokenizer.json";
const CONFIG_FILE: &str    = "config.json";

fn main() -> Result<()> {
    let device = Device::Cpu;

    // 1. Fetch model files from Hugging Face Hub
    println!("Fetching model files from Hugging Face Hub...");
    let api = Api::new()?;
    let repo = api.repo(Repo::new(MODEL_ID.to_string(), RepoType::Model));

    let config_path    = repo.get(CONFIG_FILE)?;
    let tokenizer_path = repo.get(TOKENIZER_FILE)?;
    let index_path     = repo.get(SAFETENSORS_INDEX_FILE)?;

    let safetensors_paths: Vec<PathBuf> = (1..=40)
        .map(|i| {
            let filename = format!("model-{:05}-of-00040.safetensors", i);
            repo.get(&filename).map_err(anyhow::Error::from) // Apply map_err here
        })
        .collect::<Result<Vec<PathBuf>, anyhow::Error>>()?;

    println!("Model files downloaded: Config: {:?}, Tokenizer: {:?}, Safetensors Index: {:?}, Shards: {} files",
        config_path, tokenizer_path, index_path, safetensors_paths.len());

    // 2. Load Tokenizer
    println!("Loading tokenizer...");
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;
    let eos_token = tokenizer.token_to_id("<|endoftext|>").unwrap_or(0);
    println!("Tokenizer loaded. EOS token ID: {}", eos_token);

    // 3. Load Config
    println!("Loading model config...");
    let config_str = std::fs::read_to_string(config_path)?;
    let config: Config = serde_json::from_str(&config_str)?;
    println!("Config loaded: {:?}", config);

    // 4. Load Model Weights (Safetensors)
    println!("Loading model weights from safetensors: {:?}...", safetensors_paths);
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&safetensors_paths, DType::BF16, &device)? // Assuming BF16 as Qwen default
    };
    let mut model = QwenModel::new(&config, vb)?;
    println!("Model loaded.");

    // 5. Text Generation Loop
    let prompt = "def fibonacci(n):";
    let mut tokens = tokenizer.encode(prompt, true).map_err(anyhow::Error::msg)?.get_ids().to_vec();
    let mut buffer = BufWriter::new(std::io::stdout());

    println!("\n--- Generating text from prompt: \"{}\" ---", prompt);
    write!(buffer, "{}", prompt)?;

    let mut new_tokens = Vec::new();
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let max_gen_tokens = 200;
    let temperature = 0.8;
    let top_k = 0.9;

    for i in 0..max_gen_tokens {
        let input_tensor = Tensor::new(&tokens[..], &device)?.unsqueeze(0)?;
        let logits = model.forward(&input_tensor, tokens.len())?;
        let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;

        let next_token = sample_token(&logits, temperature, top_k, &mut rng)?;

        if next_token == eos_token {
            println!("
[EOS]");
            break;
        }

        tokens.push(next_token);
        new_tokens.push(next_token);

        let token_str = tokenizer.decode(&[next_token], true).map_err(anyhow::Error::msg)?;
        write!(buffer, "{}", token_str)?;
        buffer.flush()?;

        if i == max_gen_tokens - 1 {
            println!("
[MAX TOKENS REACHED]");
        }
    }
    println!("
--- Generation Finished ---");

    Ok(())
}

// Helper function for sampling the next token
fn sample_token(logits: &Tensor, temperature: f32, top_k: f32, rng: &mut impl rand::Rng) -> CResult<u32> {
    use candle_core::bail;
    use rand::distr::Distribution; // Added for `sample` method
    use rand::distr::weighted::WeightedIndex;

    let logits_v: Vec<f32> = logits.to_vec1()?;
    let softmax_temp = if temperature > 0. {
        let temp_tensor = Tensor::new(temperature, logits.device())?;
        softmax(&logits.div(&temp_tensor)?, 0)?
    } else {
        let mut argmax = 0;
        for i in 1..logits_v.len() {
            if logits_v[i] > logits_v[argmax] {
                argmax = i;
            }
        }
        let mut one_hot = vec![0f32; logits_v.len()];
        one_hot[argmax] = 1f32;
        Tensor::new(one_hot.as_slice(), logits.device())?
    };

    let probabilities: Vec<f32> = softmax_temp.to_vec1()?;

    let mut pairs: Vec<(f32, u32)> = probabilities
        .iter()
        .enumerate()
        .map(|(i, &p)| (p, i as u32))
        .collect();

    // Apply Top-K filtering
    if top_k > 0.0 {
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let k_idx = (pairs.len() as f32 * top_k) as usize;
        let threshold = pairs[k_idx].0;
        pairs.retain(|(p, _)| *p >= threshold);
    }

    if pairs.is_empty() {
        bail!("No tokens to sample from after filtering.");
    }

    let (weights, tokens): (Vec<f32>, Vec<u32>) = pairs.into_iter().unzip();
    let dist = WeightedIndex::new(&weights).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    Ok(tokens[dist.sample(rng)])
}