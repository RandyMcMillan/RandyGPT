mod checkpoint;
mod config;
mod forward;
mod metal;
mod model;
mod ops;
mod optimizer;
mod rng;
mod serve;
mod tokenizer;
mod train;
mod cli; // New import

use std::fs::File;
use std::io::{BufRead, BufReader, Read as _, Write as _};
use std::path::Path;

use checkpoint::{load_checkpoint, load_checkpoint_cpu, load_checkpoint_v2, load_checkpoint_v3};
use config::*;
use metal::METAL_DEVICE;
use model::{CandleModel, GPTModel};
use optimizer::GpuAdamState;
use rng::Rng;
use tokenizer::Tokenizer;
use train::{estimate_loss, generate, /*generate_cpu, */generate_cpu_streaming, train, train_candle};
use log::{debug, info, warn, error}; // Keep this here


fn load_training_data(path: &str) -> std::io::Result<String> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut text = String::new();
    for line in reader.lines() {
        text.push_str(&line?);
        text.push('\n');
    }
    Ok(text)
}

/// Build the list of valid batch start positions: windows of BLOCK_SIZE+1 tokens
/// that don't cross a document boundary (<|eos|> = token id 1).
///
/// Returns an empty Vec (caller uses random fallback) when:
/// - no <|eos|> separators exist, OR
/// - boundaries are sparse enough that <1% of windows are excluded
///   (avoids building a 200-300MB Vec that is essentially [0..N])
fn build_valid_starts(data: &[usize]) -> Vec<usize> {
    use crate::config::BLOCK_SIZE;
    if data.len() <= unsafe { BLOCK_SIZE } + 1 {
        return Vec::new();
    }
    let eos_id = 1usize; // <|eos|> is always vocab index 1
    let eos_positions: Vec<usize> = data.iter().enumerate()
        .filter(|(_, &t)| t == eos_id)
        .map(|(i, _)| i)
        .collect();
    if eos_positions.is_empty() {
        return Vec::new();
    }
    // Each boundary excludes at most BLOCK_SIZE windows. If the total excluded
    // fraction is <1%, the memory cost of storing valid_starts outweighs the benefit.
    let max_excluded = eos_positions.len() * unsafe { BLOCK_SIZE };
    let total_windows = data.len() - unsafe { BLOCK_SIZE } - 1;
    if max_excluded * 100 < total_windows {
        return Vec::new(); // <1% exclusion — not worth 200MB+ allocation
    }
    (0..total_windows)
        .filter(|&s| {
            let lo = eos_positions.partition_point(|&p| p < s);
            lo >= eos_positions.len() || eos_positions[lo] >= s + unsafe { BLOCK_SIZE }
        })
        .collect()
}


fn main() -> std::io::Result<()> {
    env_logger::init();
    debug!("Logger initialized.");

    let cli = cli::parse_args(); // Parse arguments using the new cli module

    // Load RandyGPT.toml config
    let randy_gpt_config = config::load_config(cli.config_path_arg.as_deref())
        .unwrap_or_else(|e| {
            eprintln!("Error loading configuration: {}", e);
            std::process::exit(1);
        });

    unsafe {
        // Apply TOML overrides, or use default feature values
        if let Some(n_embd) = randy_gpt_config.n_embd {
            N_EMBD = n_embd;
        }
        if let Some(n_head) = randy_gpt_config.n_head {
            N_HEAD = n_head;
        }
        if let Some(n_layer) = randy_gpt_config.n_layer {
            N_LAYER = n_layer;
        }
        if let Some(block_size) = randy_gpt_config.block_size {
            BLOCK_SIZE = block_size;
        }
        if let Some(max_vocab) = randy_gpt_config.max_vocab {
            MAX_VOCAB = max_vocab;
        }
        if let Some(batch_size) = randy_gpt_config.batch_size {
            BATCH_SIZE = batch_size;
        }
        if let Some(bpe_vocab_path) = randy_gpt_config.bpe_vocab_path {
            BPE_VOCAB_PATH = bpe_vocab_path;
        } else if BPE_VOCAB_PATH.is_empty() {
            BPE_VOCAB_PATH = "vocab.json".to_string(); // Default if not in TOML and not already set
        }

        // Re-calculate derived constants after potential overrides
        HEAD_DIM = N_EMBD / N_HEAD;
        MLP_DIM = 4 * N_EMBD;

        // Apply CLI model preset overrides (these take precedence over TOML)
        if cli.model_xs {
            N_EMBD = MODEL_XS_N_EMBD;
            N_HEAD = MODEL_XS_N_HEAD;
            N_LAYER = MODEL_XS_N_LAYER;
            BATCH_SIZE = MODEL_XS_BATCH_SIZE;
            debug!("CLI override: Applied --model-xs preset.");
        } else if cli.model_s {
            N_EMBD = MODEL_S_N_EMBD;
            N_HEAD = MODEL_S_N_HEAD;
            N_LAYER = MODEL_S_N_LAYER;
            BATCH_SIZE = MODEL_S_BATCH_SIZE;
            debug!("CLI override: Applied --model-s preset.");
        } else if cli.model_ds {
            N_EMBD = MODEL_DS_N_EMBD;
            N_HEAD = MODEL_DS_N_HEAD;
            N_LAYER = MODEL_DS_N_LAYER;
            BATCH_SIZE = MODEL_DS_BATCH_SIZE;
            debug!("CLI override: Applied --model-ds preset.");
        } else if cli.model_m {
            N_EMBD = MODEL_M_N_EMBD;
            N_HEAD = MODEL_M_N_HEAD;
            N_LAYER = MODEL_M_N_LAYER;
            BATCH_SIZE = MODEL_M_BATCH_SIZE;
            debug!("CLI override: Applied --model-m preset.");
        } else if cli.model_l {
            N_EMBD = MODEL_L_N_EMBD;
            N_HEAD = MODEL_L_N_HEAD;
            N_LAYER = MODEL_L_N_LAYER;
            BATCH_SIZE = MODEL_L_BATCH_SIZE;
            debug!("CLI override: Applied --model-l preset.");
        } else if cli.model_deep {
            N_EMBD = MODEL_DEEP_N_EMBD;
            N_HEAD = MODEL_DEEP_N_HEAD;
            N_LAYER = MODEL_DEEP_N_LAYER;
            BATCH_SIZE = MODEL_DEEP_BATCH_SIZE;
            debug!("CLI override: Applied --model-deep preset.");
        } else if cli.model_xl {
            N_EMBD = MODEL_XL_N_EMBD;
            N_HEAD = MODEL_XL_N_HEAD;
            N_LAYER = MODEL_XL_N_LAYER;
            BATCH_SIZE = MODEL_XL_BATCH_SIZE;
            debug!("CLI override: Applied --model-xl preset.");
        }

        // Re-calculate derived constants after CLI overrides as well
        HEAD_DIM = N_EMBD / N_HEAD;
        MLP_DIM = 4 * N_EMBD;
    }

    ctrlc_tiny::init_ctrlc().expect("Error setting Ctrl-C handler");
    // ── CLI arguments ─────────────────────────────────────────────────
    // Usage: randygpt [--iters N] [--resume [path]]
    debug!("CLI arguments: {:?}", cli);
    let iterations = cli.iterations;
    let mut resume_path: Option<String> = cli.resume_path;
    let lr_override:     Option<f32> = cli.lr_override;
    let min_lr_override: Option<f32> = cli.min_lr_override;
    let bpe_vocab_size:  Option<usize> = cli.bpe_vocab_size;
    let generate_mode:   bool = cli.generate_mode;
    let generate_prompts: Vec<String> = cli.generate_prompts;
    let serve_mode:      bool = cli.serve_mode;
    let serve_addr:      Option<String> = cli.serve_addr;
    let api_key:         Option<String> = cli.api_key;
    let train_file:              String = cli.train_file;
    let vocab_path:              String = cli.vocab_path;
    let checkpoint_prefix_arg:   Option<String> = cli.checkpoint_prefix_arg;
    let fine_tune:               bool = cli.fine_tune;
    let gen_max_tokens:         usize = cli.gen_max_tokens;
    let gen_temperature:         f32  = cli.gen_temperature;
    let gen_top_k:               f32  = cli.gen_top_k;
    
    // These need to be mutable here for the training loop
    let mut lr     = lr_override.unwrap_or(unsafe { LEARNING_RATE });
    let mut min_lr = min_lr_override.unwrap_or(unsafe { MIN_LEARNING_RATE });

    // Derive checkpoint prefix: explicit --checkpoint > stem of --train-file > "checkpoint"
    let checkpoint_prefix = checkpoint_prefix_arg.unwrap_or_else(|| {
        let stem = std::path::Path::new(&train_file)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("checkpoint");
        // train.txt → checkpoint, train_rust.txt → checkpoint_rust
        if stem == "train" {
            debug!("Derived checkpoint prefix: checkpoint (from default)");
            "checkpoint".to_string()
        } else {
            let prefix = format!("checkpoint_{}", stem.trim_start_matches("train_"));
            debug!("Derived checkpoint prefix: {} (from train file)", prefix);
            prefix
        }
    });

    // Resolve deferred --resume (bare flag with no path)
    if resume_path.as_deref() == Some("__default__") {
        let best = format!("{}_best.bin", checkpoint_prefix);
        let cur  = format!("{}.bin",      checkpoint_prefix);
        if Path::new(&best).exists() {
            resume_path = Some(best.clone());
            debug!("Resolved deferred --resume to best checkpoint: {}", best);
        } else {
            resume_path = Some(cur.clone());
            debug!("Resolved deferred --resume to current checkpoint: {}", cur);
        }
    }

    // --generate implies --resume if no explicit --resume given
    if generate_mode && resume_path.is_none() {
        let best = format!("{}_best.bin", checkpoint_prefix);
        let cur  = format!("{}.bin",      checkpoint_prefix);
        if Path::new(&best).exists() {
            resume_path = Some(best.clone());
            debug!("--generate: implied --resume from best checkpoint: {}", best);
        } else if Path::new(&cur).exists() {
            resume_path = Some(cur.clone());
            debug!("--generate: implied --resume from current checkpoint: {}", cur);
        } else {
            eprintln!("Error: --generate requires a checkpoint file. Train first or specify --resume <path>.");
            error!("No checkpoint found for --generate. Exiting.");
            return Ok(());
        }
    }

    // --serve implies --resume if no explicit --resume given
    if serve_mode && resume_path.is_none() {
        let best = format!("{}_best.bin", checkpoint_prefix);
        let cur  = format!("{}.bin",      checkpoint_prefix);
        if Path::new(&best).exists() {
            resume_path = Some(best.clone());
            debug!("--serve: implied --resume from best checkpoint: {}", best);
        } else if Path::new(&cur).exists() {
            resume_path = Some(cur.clone());
            debug!("--serve: implied --resume from current checkpoint: {}", cur);
        } else {
            eprintln!("Error: --serve requires a checkpoint file. Train first or specify --resume <path>.");
            error!("No checkpoint found for --serve. Exiting.");
            return Ok(());
        }
    }

    let ckpt_bin = format!("{}.bin", checkpoint_prefix);
    debug!("Checking for existing checkpoint: {}", ckpt_bin);
    if !generate_mode && resume_path.is_none() && Path::new(&ckpt_bin).exists() {
        eprintln!("Found {} — use --resume to continue from it, or delete it to start fresh.", ckpt_bin);
        info!("Found existing checkpoint file without --resume: {}", ckpt_bin);
    }
    if lr_override.is_some() || min_lr_override.is_some() {
        println!("LR override: {} → {}", lr, min_lr);
        debug!("Learning rate override active: {} -> {}", lr, min_lr);
    }

    let mut model_size_name = "Custom".to_string();
    unsafe {
        // Dynamic model name based on current parameters
        if N_EMBD == MODEL_XS_N_EMBD && N_HEAD == MODEL_XS_N_HEAD && N_LAYER == MODEL_XS_N_LAYER && BATCH_SIZE == MODEL_XS_BATCH_SIZE {
            model_size_name = "XS (~0.86M)".to_string();
        } else if N_EMBD == MODEL_S_N_EMBD && N_HEAD == MODEL_S_N_HEAD && N_LAYER == MODEL_S_N_LAYER && BATCH_SIZE == MODEL_S_BATCH_SIZE {
            model_size_name = "S (~1.6M)".to_string();
        } else if N_EMBD == MODEL_DS_N_EMBD && N_HEAD == MODEL_DS_N_HEAD && N_LAYER == MODEL_DS_N_LAYER && BATCH_SIZE == MODEL_DS_BATCH_SIZE {
            model_size_name = "DS (~2.78M)".to_string();
        } else if N_EMBD == MODEL_M_N_EMBD && N_HEAD == MODEL_M_N_HEAD && N_LAYER == MODEL_M_N_LAYER && BATCH_SIZE == MODEL_M_BATCH_SIZE {
            model_size_name = "M (~2.7M)".to_string();
        } else if N_EMBD == MODEL_L_N_EMBD && N_HEAD == MODEL_L_N_HEAD && N_LAYER == MODEL_L_N_LAYER && BATCH_SIZE == MODEL_L_BATCH_SIZE {
            model_size_name = "L (~4.82M)".to_string();
        } else if N_EMBD == MODEL_DEEP_N_EMBD && N_HEAD == MODEL_DEEP_N_HEAD && N_LAYER == MODEL_DEEP_N_LAYER && BATCH_SIZE == MODEL_DEEP_BATCH_SIZE {
            model_size_name = "Deep (~7.5M)".to_string();
        } else if N_EMBD == MODEL_XL_N_EMBD && N_HEAD == MODEL_XL_N_HEAD && N_LAYER == MODEL_XL_N_LAYER && BATCH_SIZE == MODEL_XL_BATCH_SIZE {
            model_size_name = "XL (~10.8M)".to_string();
        }
        debug!("Selected model size: {}", model_size_name);
        println!("=== Enhanced randyGPT ===");
        println!("Model: {} — {} layers, {} heads, {}-dim", model_size_name, N_LAYER, N_HEAD, N_EMBD);
        println!("Block size: {}, Vocab size: up to {}", BLOCK_SIZE, MAX_VOCAB);
        println!();
    }

    let mut rng = Rng::new(1337);
    debug!("RNG initialized with seed 1337.");

    // ── Generate-only: skip training data, just load tokenizer ──────
    if generate_mode {
        debug!("Entering generate mode.");
        // Force Metal init now so the banner prints before any generation output.
        let _ = METAL_DEVICE.is_some();

        let tokenizer = if let Some(_target) = bpe_vocab_size {
            if Path::new(&vocab_path).exists() {
                println!("Loading BPE vocab from {}...", vocab_path);
                debug!("Attempting to load BPE vocab from {} in generate mode.", vocab_path);
                let t = Tokenizer::load_bpe(&vocab_path)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
                println!("Loaded BPE vocab ({} tokens)", t.vocab_size);
                debug!("Successfully loaded BPE vocab with {} tokens.", t.vocab_size);
                t
            } else {
                error!("BPE vocab file not found at {}.", vocab_path);
                return Err(std::io::Error::new(std::io::ErrorKind::NotFound,
                    format!("No {} found. Train a model first before using --generate.", vocab_path)));
            }
        } else {
            error!("--generate requires BPE mode. Char-level generate needs training data.");
            return Err(std::io::Error::new(std::io::ErrorKind::Other,
                "--generate requires BPE mode (--bpe N). Char-level generate needs training data."));
        };

        println!("Vocabulary size: {}", tokenizer.vocab_size);
        debug!("Initializing GPTModel with vocab size: {}", tokenizer.vocab_size);
        println!();

        // Load model + checkpoint
        let mut model = GPTModel::new(tokenizer.vocab_size, &mut rng);
        let param_count = model.wte.len() + model.wpe.len() + model.lm_head.len()
            + unsafe { N_LAYER } * (
                model.layers[0].wq.len() + model.layers[0].wk.len()
                + model.layers[0].wv.len() + model.layers[0].wo.len()
                + model.layers[0].fc1.len() + model.layers[0].fc2.len()
            );
        println!("Parameters: ~{:.2}M", param_count as f32 / 1_000_000.0);
        debug!("GPTModel initialized with ~{:.2}M parameters.", param_count as f32 / 1_000_000.0);

        if let Some(ref path) = resume_path {
            println!("Loading checkpoint: {}...", path);
            debug!("Attempting to load checkpoint from {} in generate mode.", path);
            load_checkpoint_cpu(path, &mut model)
                .map_err(|e| {
                    error!("Failed to load checkpoint {}: {}", path, e);
                    std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
                })?;
            debug!("Successfully loaded checkpoint: {}", path);
        } else {
            error!("No checkpoint path provided for --generate.");
            return Err(std::io::Error::new(std::io::ErrorKind::NotFound,
                "No checkpoint found for --generate. Train a model first."));
        }

        let prompts: Vec<&str> = if generate_prompts.is_empty() {
            debug!("Using default generation prompts.");
            vec!["The ", "Once upon a time", "He said", "She walked into the room", "Chapter 3"]
        } else {
            debug!("Using custom generation prompts: {:?}", generate_prompts);
            generate_prompts.iter().map(|s| s.as_str()).collect()
        };
        println!("=== Generation Mode ===");
        println!("Checkpoint: {}", resume_path.as_deref().unwrap_or("?"));
        println!();
        for prompt in &prompts {
            println!("────────────────────────────────────");
            println!("Prompt: \"{}\"", prompt);
            println!("────────────────────────────────────");
            generate_cpu_streaming(&model, &tokenizer, prompt, gen_max_tokens, gen_temperature, gen_top_k, &mut rng);
            println!();
            println!();
        }
        return Ok(());
    }

    // ── Serve mode: load tokenizer + model, run HTTP server ─────────────
    if serve_mode {
        debug!("Entering serve mode.");
        let addr = serve_addr.unwrap_or_else(|| "0.0.0.0:8080".to_string());

        let tokenizer = if let Some(_target) = bpe_vocab_size {
            if Path::new(&vocab_path).exists() {
                println!("Loading BPE vocab from {}...", vocab_path);
                debug!("Attempting to load BPE vocab from {} in serve mode.", vocab_path);
                let t = Tokenizer::load_bpe(&vocab_path)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
                println!("Loaded BPE vocab ({} tokens)", t.vocab_size);
                debug!("Successfully loaded BPE vocab with {} tokens.", t.vocab_size);
                t
            } else {
                error!("BPE vocab file not found at {}.", vocab_path);
                return Err(std::io::Error::new(std::io::ErrorKind::NotFound,
                    format!("No {} found. Train first or specify --bpe.", vocab_path)));
            }
        } else {
            error!("--serve requires BPE mode. Using default tokenizer requires training data.");
            return Err(std::io::Error::new(std::io::ErrorKind::Other,
                "--serve requires BPE mode (--bpe N). Use --bpe with a trained vocab.json."));
        };

        println!("Vocabulary size: {}", tokenizer.vocab_size);
        debug!("Initializing GPTModel with vocab size: {} in serve mode.", tokenizer.vocab_size);
        println!();

        let mut model = GPTModel::new(tokenizer.vocab_size, &mut rng);
        let param_count = model.wte.len() + model.wpe.len() + model.lm_head.len()
            + unsafe { N_LAYER } * (
                model.layers[0].wq.len() + model.layers[0].wk.len()
                + model.layers[0].wv.len() + model.layers[0].wo.len()
                + model.layers[0].fc1.len() + model.layers[0].fc2.len()
            );
        println!("Parameters: ~{:.2}M", param_count as f32 / 1_000_000.0);
        debug!("GPTModel initialized with ~{:.2}M parameters in serve mode.", param_count as f32 / 1_000_000.0);
        println!();

        if let Some(ref path) = resume_path {
            println!("Loading checkpoint: {}...", path);
            debug!("Attempting to load checkpoint from {} in serve mode.", path);
            load_checkpoint_cpu(path, &mut model)
                .map_err(|e| {
                    error!("Failed to load checkpoint {}: {}", path, e);
                    std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
                })?;
            debug!("Successfully loaded checkpoint: {}", path);
        } else {
            error!("No checkpoint path provided for --serve.");
            return Err(std::io::Error::new(std::io::ErrorKind::NotFound,
                "No checkpoint found for --serve. Train a model first."));
        }

        let model_name = format!("randygpt-{}L-{}H-{}D", unsafe { N_LAYER }, unsafe { N_HEAD }, unsafe { N_EMBD });
        debug!("Starting HTTP server at {} for model: {}", addr, model_name);

        serve::run_server(&addr, &model, &tokenizer, &model_name, api_key.as_deref());
        return Ok(());
    }

    // ── Load training data + tokenizer + tokens ─────────────────────────
    // Memory optimization: if we have both tokens.bin and vocab.json cached,
    // skip loading the raw training text entirely (saves ~110MB for large corpora).
    let token_cache_path = format!("{}.tokens.bin", train_file);
    let have_token_cache = Path::new(&token_cache_path).exists();
    let have_bpe_vocab   = bpe_vocab_size.is_some() && Path::new(unsafe { BPE_VOCAB_PATH.as_str() }).exists();
    debug!("Token cache path: {}, exists: {}", token_cache_path, have_token_cache);
    debug!("BPE vocab path: {}, exists: {}", unsafe { BPE_VOCAB_PATH.as_str() }, have_bpe_vocab);

    let (tokenizer, data, val_data) = if have_token_cache && have_bpe_vocab {
        // Fast path: load vocab + cached tokens, skip raw text entirely
        println!("Loading BPE vocab from {}...", unsafe { BPE_VOCAB_PATH.as_str() });
        debug!("Fast path: Loading BPE vocab from {}.", unsafe { BPE_VOCAB_PATH.as_str() });
        let tokenizer = Tokenizer::load_bpe(unsafe { BPE_VOCAB_PATH.as_str() })
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        println!("Loaded BPE vocab ({} tokens)", tokenizer.vocab_size);
        debug!("Fast path: Loaded BPE vocab with {} tokens.", tokenizer.vocab_size);

        println!("Loading cached tokens from {}...", token_cache_path);
        debug!("Fast path: Loading cached tokens from {}.", token_cache_path);
        let mut f = File::open(&token_cache_path)?;
        let mut buf = Vec::new();
        f.read_to_end(&mut buf)?;
        let data_all: Vec<usize> = buf.chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as usize)
            .collect();
        drop(buf); // free raw bytes immediately
        println!("Loaded {} cached tokens", data_all.len());
        debug!("Fast path: Loaded {} cached tokens.", data_all.len());

        let val_split = (data_all.len() * 9) / 10;
        let data     = data_all[..val_split].to_vec();
        let val_data = data_all[val_split..].to_vec();
        println!("Tokens: {} train, {} val (skipped {} — using cache)",
            data.len(), val_data.len(), train_file);
        debug!("Fast path: {} train tokens, {} validation tokens. Skipped raw text.", data.len(), val_data.len());
        // data_all dropped here when it goes out of scope

        (tokenizer, data, val_data)
    } else {
        // Full path: load text, build/load tokenizer, tokenize, cache
        debug!("Full path: Token cache or BPE vocab not available. Loading raw text and processing.");
        let training_text = if Path::new(&train_file).exists() {
            println!("Loading training data from {}...", train_file);
            debug!("Loading training data from {}.", train_file);
            load_training_data(&train_file)?
        } else {
            println!("No {} found. Using default sample data.", train_file);
            warn!("Training file {} not found. Using default sample data.", train_file);
            concat!(
                "The quick brown fox jumps over the lazy dog. ",
                "Rust is a systems programming language. ",
                "Machine learning models learn from data. ",
                "Transformers use attention mechanisms. ",
                "GPT stands for Generative Pre-trained Transformer. ",
                "Neural networks are inspired by the human brain. ",
                "Deep learning is a subset of machine learning. "
            ).to_string()
        };
        println!("Training data size: {} characters", training_text.len());
        debug!("Training data loaded, size: {} characters.", training_text.len());

        let tokenizer = if let Some(target) = bpe_vocab_size {
            if Path::new(unsafe { BPE_VOCAB_PATH.as_str() }).exists() {
                println!("Loading BPE vocab from {}...", unsafe { BPE_VOCAB_PATH.as_str() });
                debug!("Attempting to load BPE vocab from {}.", unsafe { BPE_VOCAB_PATH.as_str() });
                match Tokenizer::load_bpe(unsafe { BPE_VOCAB_PATH.as_str() }) {
                    Ok(t)  => { println!("Loaded BPE vocab ({} tokens)", t.vocab_size); debug!("Loaded BPE vocab with {} tokens.", t.vocab_size); t }
                    Err(e) => {
                        eprintln!("Failed to load {}: {}. Retraining...", unsafe { BPE_VOCAB_PATH.as_str() }, e);
                        warn!("Failed to load BPE vocab {}: {}. Retraining.", unsafe { BPE_VOCAB_PATH.as_str() }, e);
                        let t = Tokenizer::from_text_bpe(&training_text, target);
                        t.save_bpe(unsafe { BPE_VOCAB_PATH.as_str() })?;
                        println!("BPE vocab ({} tokens) saved to {}", t.vocab_size, unsafe { BPE_VOCAB_PATH.as_str() });
                        debug!("Retrained and saved BPE vocab with {} tokens to {}.", t.vocab_size, unsafe { BPE_VOCAB_PATH.as_str() });
                        t
                    }
                }
            } else {
                println!("Training BPE tokenizer (target vocab: {})...", target);
                info!("Training new BPE tokenizer with target vocab: {}.", target);
                let t = Tokenizer::from_text_bpe(&training_text, target);
                t.save_bpe(unsafe { BPE_VOCAB_PATH.as_str() })?;
                println!("BPE vocab ({} tokens) saved to {}", t.vocab_size, unsafe { BPE_VOCAB_PATH.as_str() });
                debug!("New BPE vocab ({} tokens) saved to {}.", t.vocab_size, unsafe { BPE_VOCAB_PATH.as_str() });
                t
            }
        } else {
            debug!("Using character-level tokenizer.");
            Tokenizer::from_text(&training_text)
        };

        let data_all = if have_token_cache {
            println!("Loading cached tokens from {}...", token_cache_path);
            debug!("Loading cached tokens from {}.", token_cache_path);
            let mut f = File::open(&token_cache_path)?;
            let mut buf = Vec::new();
            f.read_to_end(&mut buf)?;
            let tokens: Vec<usize> = buf.chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as usize)
                .collect();
            println!("Loaded {} cached tokens", tokens.len());
            debug!("Loaded {} cached tokens.", tokens.len());
            tokens
        } else {
            println!("Tokenizing text ({} chars)...", training_text.len());
            debug!("Tokenizing text with {} characters.", training_text.len());
            let tokens = tokenizer.encode(&training_text);
            let mut f = File::create(&token_cache_path)?;
            for &t in &tokens {
                f.write_all(&(t as u32).to_le_bytes())?;
            }
            println!("Saved token cache to {} ({:.1}MB)",
                token_cache_path, (tokens.len() * 4) as f64 / 1_048_576.0);
            debug!("Saved token cache to {} ({:.1}MB) with {} tokens.", token_cache_path, (tokens.len() * 4) as f64 / 1_048_576.0, tokens.len());
            tokens
        };
        // training_text dropped here — frees ~110MB for large corpora

        let val_split = (data_all.len() * 9) / 10;
        let data     = data_all[..val_split].to_vec();
        let val_data = data_all[val_split..].to_vec();
        println!("Tokenized to {} tokens ({} train, {} val)",
            data_all.len(), data.len(), val_data.len());
        debug!("Tokenized to {} total tokens ({} train, {} val).", data_all.len(), data.len(), val_data.len());
        // data_all dropped here

        (tokenizer, data, val_data)
    };

    println!("Vocabulary size: {}", tokenizer.vocab_size);
    debug!("Final vocabulary size: {}", tokenizer.vocab_size);

    // Build document-boundary-aware valid start positions (empty = use random fallback)
    let valid_starts     = build_valid_starts(&data);
    let val_valid_starts = build_valid_starts(&val_data);
    debug!("Built {} valid train start positions.", valid_starts.len());
    debug!("Built {} valid validation start positions.", val_valid_starts.len());
    if !valid_starts.is_empty() {
        let pct = 100.0 * valid_starts.len() as f64
            / data.len().saturating_sub(unsafe { crate::config::BLOCK_SIZE } + 1) as f64;
        println!("Doc-boundary sampling: {} valid train windows ({:.1}% of total)",
            valid_starts.len(), pct);
        debug!("Document-boundary sampling: {} valid train windows ({:.1}% of total).", valid_starts.len(), pct);
    }
    println!();

    // ── Initialize model ──────────────────────────────────────────────
    println!("Initializing model...");
    debug!("Initializing GPTModel for training.");
    let mut model = GPTModel::new(tokenizer.vocab_size, &mut rng);

    let param_count = model.wte.len() + model.wpe.len() + model.lm_head.len()
        + unsafe { N_LAYER } * (
            model.layers[0].wq.len() + model.layers[0].wk.len()
            + model.layers[0].wv.len() + model.layers[0].wo.len()
            + model.layers[0].fc1.len() + model.layers[0].fc2.len()
        );
    println!("Parameters: ~{:.2}M", param_count as f32 / 1_000_000.0);
    debug!("GPTModel initialized with ~{:.2}M parameters.", param_count as f32 / 1_000_000.0);
    println!();

    // ── Force Metal init so we know which path to take ───────────────
    let use_metal = METAL_DEVICE.is_some();
    debug!("Metal device available: {}", use_metal);
    if use_metal {
        println!("Metal GPU: enabled — training via Candle autograd");
        debug!("Metal GPU enabled.");
    } else {
        println!("Metal GPU: unavailable — training on CPU (BLAS)");
        debug!("Metal GPU unavailable, falling back to CPU (BLAS).");
    }
    println!();

    // ── Resume from checkpoint ────────────────────────────────────────
    // On Metal: try RGPT0003, then RGPT0002 (moments reset), then RGPT0001.
    // On CPU:   RGPT0001 only.
    //
    // candle_resume holds (CandleModel, GpuAdamState, iter, step, best_loss)
    // if an RGPT0003 checkpoint was successfully loaded; otherwise None and
    // the model weights are available in the CPU `model` variable.
    let mut candle_resume: Option<(CandleModel, GpuAdamState, usize, usize, f32)> = None;

    let (iter_start, step_start, best_loss_start) = if let Some(ref ckpt) = resume_path {
        debug!("Attempting to resume from checkpoint: {}", ckpt);
        let result: std::io::Result<(usize, usize, f32)> = if use_metal {
            debug!("Resuming on Metal. Trying RGPT0003, RGPT0002, then RGPT0001.");
            let device = METAL_DEVICE.as_ref().unwrap();

            // Try RGPT0003 (full GPU state)
            let r3 = {
                debug!("Attempting to load RGPT0003 checkpoint.");
                let mut cm = CandleModel::from_gpt(&model, device)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
                let vars = cm.all_vars();
                let mut opt = GpuAdamState::new(&vars)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
                load_checkpoint_v3(ckpt, &mut cm, &mut opt).map(|(it, st, bl)| {
                    candle_resume = Some((cm, opt, it, st, bl));
                    (it, st, bl)
                })
            };

            if r3.is_ok() {
                debug!("Successfully loaded RGPT0003 checkpoint.");
                r3
            } else {
                warn!("Failed to load RGPT0003 checkpoint. Error: {:?}. Trying RGPT0002.", r3.err());
                // Try RGPT0002 (weights only, moments reset to zero)
                let r2 = {
                    debug!("Attempting to load RGPT0002 checkpoint.");
                    let mut cm = CandleModel::from_gpt(&model, device)
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
                    load_checkpoint_v2(ckpt, &mut cm).map(|(it, st, bl)| {
                        let vars = cm.all_vars();
                        let opt = GpuAdamState::new(&vars)
                            .expect("GpuAdamState init failed");
                        candle_resume = Some((cm, opt, it, st, bl));
                        (it, st, bl)
                    })
                };
                if r2.is_ok() { debug!("Successfully loaded RGPT0002 checkpoint."); r2 } else {
                    warn!("Failed to load RGPT0002 checkpoint. Error: {:?}. Trying RGPT0001 (CPU compatible).", r2.err());
                    debug!("Attempting to load RGPT0001 checkpoint on Metal after RGPT0002 failure.");
                    load_checkpoint(ckpt, &mut model)
                }
            }
        } else {
            debug!("Resuming on CPU. Loading RGPT0001 checkpoint.");
            load_checkpoint(ckpt, &mut model)
        };

        match result {
            Ok((it, st, bl)) => {
                if fine_tune {
                    println!("✓ Loaded weights from '{}' (fine-tune: iter/step/best reset)", ckpt);
                    info!("Fine-tune enabled: Loaded weights from {} but reset iter/step/best loss.", ckpt);
                    println!();
                    (0, 0, f32::INFINITY)
                } else {
                    println!("✓ Resumed from '{}' — iter {}, step {}, best loss {:.4}", ckpt, it, st, bl);
                    info!("Resumed from {} — iter {}, step {}, best loss {:.4}.", ckpt, it, st, bl);
                    println!();
                    (it, st, bl)
                }
            }
            Err(e) => {
                eprintln!("Error loading checkpoint '{}': {}", ckpt, e);
                error!("Error loading checkpoint {}: {}. Starting from scratch.", ckpt, e);
                eprintln!("Starting from scratch instead.");
                (0, 0, f32::INFINITY)
            }
        }
    } else {
        debug!("No resume path provided. Starting from scratch.");
        (0, 0, f32::INFINITY)
    };

    // ── Sync resumed weights to CPU model ──────────────────────────
    // When resuming on Metal, the loaded weights live in the CandleModel.
    // Sync them back to `model` now so estimate_loss / generate sees the
    // actual checkpoint state, not freshly-initialized random weights.
    if let Some((ref cm, _, _, _, _)) = candle_resume {
        debug!("Syncing resumed weights from CandleModel to CPU model.");
        model = cm.to_gpt()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
    }

    if iter_start >= iterations {
        println!("Already at iteration {} (target {}). Nothing to train.", iter_start, iterations);
        info!("Training not needed. Already at iteration {} (target {}).", iter_start, iterations);
        println!("Increase --iters to continue training.");
        return Ok(());
    }

    // ── Initial loss estimate ─────────────────────────────────────────
    println!("Estimating initial loss...");
    debug!("Estimating initial training loss.");
    let initial_loss     = estimate_loss(&model, &data, &valid_starts, 50, &mut rng);
    debug!("Initial training loss: {:.4}", initial_loss);
    let initial_val_loss = estimate_loss(&model, &val_data, &val_valid_starts, 50, &mut rng);
    debug!("Initial validation loss: {:.4}", initial_val_loss);
    println!("Initial loss: {:.4} | Val: {:.4} (ppl {:.1})",
        initial_loss, initial_val_loss, initial_val_loss.exp());
    println!();


    // ── Train ─────────────────────────────────────────────────────────
    if use_metal {
        debug!("Starting Metal GPU training via Candle autograd.");
        let device = METAL_DEVICE.as_ref().unwrap();
        let (mut candle_model, mut opt) = if let Some((cm, o, _, _, _)) = candle_resume {
            debug!("Resuming CandleModel and optimizer from checkpoint.");
            if fine_tune {
                // Keep weights, discard moments — stale Gutenberg moments cause NaN on new domain
                let vars = cm.all_vars();
                let fresh_opt = GpuAdamState::new(&vars)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
                println!("Fine-tune: optimizer moments reset to zero.");
                info!("Fine-tune: optimizer moments reset to zero for Metal training.");
                (cm, fresh_opt)
            } else {
                debug!("Continuing Metal training with existing optimizer state.");
                (cm, o)
            }
        } else {
            debug!("Initializing new CandleModel and optimizer for Metal training.");
            let cm = CandleModel::from_gpt(&model, device)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
            let vars = cm.all_vars();
            let o = GpuAdamState::new(&vars)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
            (cm, o)
        };
        // Sync step_t so bias correction starts correctly
        opt.step_t = step_start;
        debug!("Optimizer step_t synced to: {}", opt.step_t);
        train_candle(&mut candle_model, &mut opt, &data, &val_data,
            &valid_starts, &val_valid_starts,
            iterations, &mut rng,
            iter_start, step_start, best_loss_start, lr, min_lr,
            &checkpoint_prefix);
        debug!("Metal training complete. Converting CandleModel back to GPTModel.");
        model = candle_model.to_gpt()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
    } else {
        debug!("Starting CPU (BLAS) training.");
        train(&mut model, &data, &val_data, &valid_starts, &val_valid_starts,
            iterations, &mut rng,
            iter_start, step_start, best_loss_start, lr, min_lr,
            &checkpoint_prefix);
        debug!("CPU training complete.");
    }

    // ── Final loss estimate ───────────────────────────────────────────
    println!("Estimating final loss...");
    debug!("Estimating final training loss.");
    let final_loss     = estimate_loss(&model, &data, &valid_starts, 50, &mut rng);
    debug!("Final training loss: {:.4}", final_loss);
    let final_val_loss = estimate_loss(&model, &val_data, &val_valid_starts, 50, &mut rng);
    debug!("Final validation loss: {:.4}", final_val_loss);
    println!("Final train loss: {:.4} (started {:.4})", final_loss, initial_loss);
    println!("Final val loss:   {:.4} (ppl {:.1}, started {:.4})",
        final_val_loss, final_val_loss.exp(), initial_val_loss);
    println!();

    // ── Generate samples ──────────────────────────────────────────────
    println!("=== Generation After Training ===");
    debug!("Generating samples after training.");
    for (prompt, max_tokens) in &[("ROMEO:", 100), ("To be or not to be", 100), ("Once upon a time", 100)] {
        println!("\nPrompt: \"{}\"", prompt);
        debug!("Generating sample for prompt: \"{}\" (max_tokens: {}).", prompt, max_tokens);
        let sample = generate(&model, &tokenizer, prompt, *max_tokens, 0.8, 0.9, &mut rng);
        println!("{}", sample);
        debug!("Generated sample: {}", sample);
    }

    Ok(())
}