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
    if data.len() <= BLOCK_SIZE + 1 {
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
    let max_excluded = eos_positions.len() * BLOCK_SIZE;
    let total_windows = data.len() - BLOCK_SIZE - 1;
    if max_excluded * 100 < total_windows {
        return Vec::new(); // <1% exclusion — not worth 200MB+ allocation
    }
    (0..total_windows)
        .filter(|&s| {
            let lo = eos_positions.partition_point(|&p| p < s);
            lo >= eos_positions.len() || eos_positions[lo] >= s + BLOCK_SIZE
        })
        .collect()
}

fn main() -> std::io::Result<()> {
    ctrlc_tiny::init_ctrlc().expect("Error setting Ctrl-C handler");
    // ── CLI arguments ─────────────────────────────────────────────────
    // Usage: randygpt [--iters N] [--resume [path]]
    let args: Vec<String> = std::env::args().collect();
    let mut iterations = MAX_ITERS;
    let mut resume_path: Option<String> = None;
    let mut lr_override:     Option<f32> = None;
    let mut min_lr_override: Option<f32> = None;
    let mut bpe_vocab_size:  Option<usize> = None;
    let mut generate_mode:   bool = false;
    let mut generate_prompts: Vec<String> = Vec::new();
    let mut serve_mode:      bool = false;
    let mut serve_addr:      Option<String> = None;
    let mut api_key:         Option<String> = None;
    let mut train_file:              String = "train.txt".to_string();
    let mut vocab_path:              String = BPE_VOCAB_PATH.to_string();
    let mut checkpoint_prefix_arg:   Option<String> = None;
    let mut fine_tune:               bool = false;
    let mut gen_max_tokens:         usize = 200;
    let mut gen_temperature:         f32  = 0.8;
    let mut gen_top_k:               f32  = 0.9;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--iters" => {
                i += 1;
                if i < args.len() {
                    iterations = args[i].parse().unwrap_or(MAX_ITERS);
                }
            }
            "--resume" => {
                if i + 1 < args.len() && !args[i + 1].starts_with("--") {
                    i += 1;
                    resume_path = Some(args[i].clone());
                } else {
                    // deferred: will use checkpoint_prefix once fully parsed
                    resume_path = Some("__default__".to_string());
                }
            }
            "--lr" => {
                i += 1;
                if i < args.len() {
                    lr_override = args[i].parse().ok();
                }
            }
            "--min-lr" => {
                i += 1;
                if i < args.len() {
                    min_lr_override = args[i].parse().ok();
                }
            }
            "--bpe" => {
                // --bpe        → use default BPE_VOCAB_SIZE
                // --bpe 3000   → use custom target vocab size
                if i + 1 < args.len() && !args[i + 1].starts_with("--") {
                    i += 1;
                    bpe_vocab_size = Some(args[i].parse().unwrap_or(BPE_VOCAB_SIZE));
                } else {
                    bpe_vocab_size = Some(BPE_VOCAB_SIZE);
                }
            }
            "--generate" => {
                // --generate                     → use default prompts
                // --generate "prompt1" "prompt2"  → use custom prompts
                generate_mode = true;
                while i + 1 < args.len() && !args[i + 1].starts_with("--") {
                    i += 1;
                    generate_prompts.push(args[i].clone());
                }
            }
            "--serve" => {
                // --serve               → listen on 0.0.0.0:8080
                // --serve 127.0.0.1:9000 → listen on custom address
                serve_mode = true;
                if i + 1 < args.len() && !args[i + 1].starts_with("--") {
                    i += 1;
                    serve_addr = Some(args[i].clone());
                } else {
                    serve_addr = Some("0.0.0.0:8080".to_string());
                }
            }
            "--api-key" => {
                i += 1;
                if i < args.len() {
                    api_key = Some(args[i].clone());
                }
            }
            "--train-file" => {
                i += 1;
                if i < args.len() {
                    train_file = args[i].clone();
                }
            }
            "--vocab" => {
                i += 1;
                if i < args.len() {
                    vocab_path = args[i].clone();
                }
            }
            "--checkpoint" => {
                i += 1;
                if i < args.len() {
                    // Strip .bin suffix if provided — we append it ourselves
                    checkpoint_prefix_arg = Some(args[i].trim_end_matches(".bin").to_string());
                }
            }
            "--fine-tune" => { fine_tune = true; }
            "--max-tokens" => {
                i += 1;
                if i < args.len() {
                    gen_max_tokens = args[i].parse().unwrap_or(200);
                }
            }
            "--temperature" => {
                i += 1;
                if i < args.len() {
                    gen_temperature = args[i].parse().unwrap_or(0.8);
                }
            }
            "--top-k" => {
                i += 1;
                if i < args.len() {
                    gen_top_k = args[i].parse().unwrap_or(0.9);
                }
            }
            "--help" | "-h" => {
                println!("randyGPT — tiny GPT language model\n");
                println!("USAGE:");
                println!("  randygpt [OPTIONS]\n");
                println!("TRAINING:");
                println!("  --iters N          Training iterations (default: {})", MAX_ITERS);
                println!("  --train-file PATH  Training text file (default: train.txt)");
                println!("  --vocab PATH       BPE vocab JSON file (default: vocab.json)");
                println!("  --checkpoint NAME  Checkpoint filename prefix (default: checkpoint)");
                println!("  --bpe [N]          Use BPE tokenizer, optional target vocab size (default: {}). If N is omitted, uses default BPE_VOCAB_SIZE.", BPE_VOCAB_SIZE);
                println!("  --resume [PATH]    Resume from checkpoint (default: <prefix>_best.bin, where <prefix> is from --checkpoint or train-file).");
                println!("  --fine-tune        Load weights only, reset iter/step/best val (for domain transfer)");
                println!("  --lr LR            Learning rate override");
                println!("  --min-lr LR        Minimum learning rate override\n");
                println!("INFERENCE:");
                println!("  --generate [PROMPT...]  Generate text from checkpoint");
                println!("  --max-tokens N          Tokens to generate per prompt (default: 200)");
                println!("  --temperature F         Sampling temperature (default: 0.8, lower=focused)");
                println!("  --top-k F               Top-k cumulative probability cutoff (default: 0.9)");
                println!("  --serve [ADDR]          Start HTTP server (default: 0.0.0.0:8080)");
                println!("  --api-key KEY           API key for server auth\n");
                println!("EXAMPLES:");
                println!("  randygpt --bpe --iters 10000");
                println!("  randygpt --bpe --train-file train_rust.txt --vocab vocab_rust.json --checkpoint checkpoint_rust --iters 5000");
                println!("  randygpt --bpe --resume --generate \"fn main\"");
                std::process::exit(0);
            }
            other => {
                if let Ok(n) = other.parse::<usize>() {
                    iterations = n;
                } else {
                    eprintln!("Unknown argument '{}'. Ignoring.", other);
                }
            }
        }
        i += 1;
    }
    let lr     = lr_override.unwrap_or(LEARNING_RATE);
    let min_lr = min_lr_override.unwrap_or(MIN_LEARNING_RATE);

    // Derive checkpoint prefix: explicit --checkpoint > stem of --train-file > "checkpoint"
    let checkpoint_prefix = checkpoint_prefix_arg.unwrap_or_else(|| {
        let stem = std::path::Path::new(&train_file)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("checkpoint");
        // train.txt → checkpoint, train_rust.txt → checkpoint_rust
        if stem == "train" {
            "checkpoint".to_string()
        } else {
            format!("checkpoint_{}", stem.trim_start_matches("train_"))
        }
    });

    // Resolve deferred --resume (bare flag with no path)
    if resume_path.as_deref() == Some("__default__") {
        let best = format!("{}_best.bin", checkpoint_prefix);
        let cur  = format!("{}.bin",      checkpoint_prefix);
        if Path::new(&best).exists() {
            resume_path = Some(best);
        } else {
            resume_path = Some(cur);
        }
    }

    // --generate implies --resume if no explicit --resume given
    if generate_mode && resume_path.is_none() {
        let best = format!("{}_best.bin", checkpoint_prefix);
        let cur  = format!("{}.bin",      checkpoint_prefix);
        if Path::new(&best).exists() {
            resume_path = Some(best);
        } else if Path::new(&cur).exists() {
            resume_path = Some(cur);
        } else {
            eprintln!("Error: --generate requires a checkpoint file. Train first or specify --resume <path>.");
            return Ok(());
        }
    }

    // --serve implies --resume if no explicit --resume given
    if serve_mode && resume_path.is_none() {
        let best = format!("{}_best.bin", checkpoint_prefix);
        let cur  = format!("{}.bin",      checkpoint_prefix);
        if Path::new(&best).exists() {
            resume_path = Some(best);
        } else if Path::new(&cur).exists() {
            resume_path = Some(cur);
        } else {
            eprintln!("Error: --serve requires a checkpoint file. Train first or specify --resume <path>.");
            return Ok(());
        }
    }

    let ckpt_bin = format!("{}.bin", checkpoint_prefix);
    if !generate_mode && resume_path.is_none() && Path::new(&ckpt_bin).exists() {
        eprintln!("Found {} — use --resume to continue from it, or delete it to start fresh.", ckpt_bin);
    }
    if lr_override.is_some() || min_lr_override.is_some() {
        println!("LR override: {} → {}", lr, min_lr);
    }

    let model_size_name = if cfg!(feature = "model-s")    { "S (~1.6M)"    }
                          else if cfg!(feature = "model-ds")   { "DS (~2.78M)"  }
                          else if cfg!(feature = "model-m")    { "M (~2.7M)"    }
                          else if cfg!(feature = "model-l")    { "L (~4.82M)"   }
                          else if cfg!(feature = "model-deep") { "Deep (~7.5M)" }
                          else if cfg!(feature = "model-xl")   { "XL (~10.8M)"  }
                          else                                 { "XS (~0.86M)"  };
    println!("=== Enhanced randyGPT ===");
    println!("Model: {} — {} layers, {} heads, {}-dim", model_size_name, N_LAYER, N_HEAD, N_EMBD);
    println!("Block size: {}, Vocab size: up to {}", BLOCK_SIZE, MAX_VOCAB);
    println!();

    let mut rng = Rng::new(1337);

    // ── Generate-only: skip training data, just load tokenizer ──────
    if generate_mode {
        // Force Metal init now so the banner prints before any generation output.
        let _ = METAL_DEVICE.is_some();

        let tokenizer = if let Some(_target) = bpe_vocab_size {
            if Path::new(&vocab_path).exists() {
                println!("Loading BPE vocab from {}...", vocab_path);
                let t = Tokenizer::load_bpe(&vocab_path)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
                println!("Loaded BPE vocab ({} tokens)", t.vocab_size);
                t
            } else {
                return Err(std::io::Error::new(std::io::ErrorKind::NotFound,
                    format!("No {} found. Train a model first before using --generate.", vocab_path)));
            }
        } else {
            return Err(std::io::Error::new(std::io::ErrorKind::Other,
                "--generate requires BPE mode (--bpe N). Char-level generate needs training data."));
        };

        println!("Vocabulary size: {}", tokenizer.vocab_size);
        println!();

        // Load model + checkpoint
        let mut model = GPTModel::new(tokenizer.vocab_size, &mut rng);
        let param_count = model.wte.len() + model.wpe.len() + model.lm_head.len()
            + N_LAYER * (
                model.layers[0].wq.len() + model.layers[0].wk.len()
                + model.layers[0].wv.len() + model.layers[0].wo.len()
                + model.layers[0].fc1.len() + model.layers[0].fc2.len()
            );
        println!("Parameters: ~{:.2}M", param_count as f32 / 1_000_000.0);

        if let Some(ref path) = resume_path {
            println!("Loading checkpoint: {}...", path);
            load_checkpoint_cpu(path, &mut model)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        } else {
            return Err(std::io::Error::new(std::io::ErrorKind::NotFound,
                "No checkpoint found for --generate. Train a model first."));
        }

        let prompts: Vec<&str> = if generate_prompts.is_empty() {
            vec!["The ", "Once upon a time", "He said", "She walked into the room", "Chapter 3"]
        } else {
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
        let addr = serve_addr.unwrap_or_else(|| "0.0.0.0:8080".to_string());

        let tokenizer = if let Some(_target) = bpe_vocab_size {
            if Path::new(&vocab_path).exists() {
                println!("Loading BPE vocab from {}...", vocab_path);
                let t = Tokenizer::load_bpe(&vocab_path)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
                println!("Loaded BPE vocab ({} tokens)", t.vocab_size);
                t
            } else {
                return Err(std::io::Error::new(std::io::ErrorKind::NotFound,
                    format!("No {} found. Train first or specify --bpe.", vocab_path)));
            }
        } else {
            return Err(std::io::Error::new(std::io::ErrorKind::Other,
                "--serve requires BPE mode (--bpe N). Use --bpe with a trained vocab.json."));
        };

        println!("Vocabulary size: {}", tokenizer.vocab_size);
        println!();

        let mut model = GPTModel::new(tokenizer.vocab_size, &mut rng);
        let param_count = model.wte.len() + model.wpe.len() + model.lm_head.len()
            + N_LAYER * (
                model.layers[0].wq.len() + model.layers[0].wk.len()
                + model.layers[0].wv.len() + model.layers[0].wo.len()
                + model.layers[0].fc1.len() + model.layers[0].fc2.len()
            );
        println!("Parameters: ~{:.2}M", param_count as f32 / 1_000_000.0);

        if let Some(ref path) = resume_path {
            println!("Loading checkpoint: {}...", path);
            load_checkpoint_cpu(path, &mut model)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        } else {
            return Err(std::io::Error::new(std::io::ErrorKind::NotFound,
                "No checkpoint found for --serve. Train a model first."));
        }

        let model_name = format!("randygpt-{}L-{}H-{}D", N_LAYER, N_HEAD, N_EMBD);



        serve::run_server(&addr, &model, &tokenizer, &model_name, api_key.as_deref());
        return Ok(());
    }

    // ── Load training data + tokenizer + tokens ─────────────────────────
    // Memory optimization: if we have both tokens.bin and vocab.json cached,
    // skip loading the raw training text entirely (saves ~110MB for large corpora).
    let token_cache_path = format!("{}.tokens.bin", train_file);
    let have_token_cache = Path::new(&token_cache_path).exists();
    let have_bpe_vocab   = bpe_vocab_size.is_some() && Path::new(&vocab_path).exists();

    let (tokenizer, data, val_data) = if have_token_cache && have_bpe_vocab {
        // Fast path: load vocab + cached tokens, skip raw text entirely
        println!("Loading BPE vocab from {}...", vocab_path);
        let tokenizer = Tokenizer::load_bpe(&vocab_path)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        println!("Loaded BPE vocab ({} tokens)", tokenizer.vocab_size);

        println!("Loading cached tokens from {}...", token_cache_path);
        let mut f = File::open(&token_cache_path)?;
        let mut buf = Vec::new();
        f.read_to_end(&mut buf)?;
        let data_all: Vec<usize> = buf.chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as usize)
            .collect();
        drop(buf); // free raw bytes immediately
        println!("Loaded {} cached tokens", data_all.len());

        let val_split = (data_all.len() * 9) / 10;
        let data     = data_all[..val_split].to_vec();
        let val_data = data_all[val_split..].to_vec();
        println!("Tokens: {} train, {} val (skipped {} — using cache)",
            data.len(), val_data.len(), train_file);
        // data_all dropped here when it goes out of scope

        (tokenizer, data, val_data)
    } else {
        // Full path: load text, build/load tokenizer, tokenize, cache
        let training_text = if Path::new(&train_file).exists() {
            println!("Loading training data from {}...", train_file);
            load_training_data(&train_file)?
        } else {
            println!("No {} found. Using default sample data.", train_file);
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

        let tokenizer = if let Some(target) = bpe_vocab_size {
            if Path::new(&vocab_path).exists() {
                println!("Loading BPE vocab from {}...", vocab_path);
                match Tokenizer::load_bpe(&vocab_path) {
                    Ok(t)  => { println!("Loaded BPE vocab ({} tokens)", t.vocab_size); t }
                    Err(e) => {
                        eprintln!("Failed to load {}: {}. Retraining...", vocab_path, e);
                        let t = Tokenizer::from_text_bpe(&training_text, target);
                        t.save_bpe(&vocab_path)?;
                        println!("BPE vocab ({} tokens) saved to {}", t.vocab_size, vocab_path);
                        t
                    }
                }
            } else {
                println!("Training BPE tokenizer (target vocab: {})...", target);
                let t = Tokenizer::from_text_bpe(&training_text, target);
                t.save_bpe(&vocab_path)?;
                println!("BPE vocab ({} tokens) saved to {}", t.vocab_size, vocab_path);
                t
            }
        } else {
            Tokenizer::from_text(&training_text)
        };

        let data_all = if have_token_cache {
            println!("Loading cached tokens from {}...", token_cache_path);
            let mut f = File::open(&token_cache_path)?;
            let mut buf = Vec::new();
            f.read_to_end(&mut buf)?;
            let tokens: Vec<usize> = buf.chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as usize)
                .collect();
            println!("Loaded {} cached tokens", tokens.len());
            tokens
        } else {
            println!("Tokenizing text ({} chars)...", training_text.len());
            let tokens = tokenizer.encode(&training_text);
            let mut f = File::create(&token_cache_path)?;
            for &t in &tokens {
                f.write_all(&(t as u32).to_le_bytes())?;
            }
            println!("Saved token cache to {} ({:.1}MB)",
                token_cache_path, (tokens.len() * 4) as f64 / 1_048_576.0);
            tokens
        };
        // training_text dropped here — frees ~110MB for large corpora

        let val_split = (data_all.len() * 9) / 10;
        let data     = data_all[..val_split].to_vec();
        let val_data = data_all[val_split..].to_vec();
        println!("Tokenized to {} tokens ({} train, {} val)",
            data_all.len(), data.len(), val_data.len());
        // data_all dropped here

        (tokenizer, data, val_data)
    };

    println!("Vocabulary size: {}", tokenizer.vocab_size);

    // Build document-boundary-aware valid start positions (empty = use random fallback)
    let valid_starts     = build_valid_starts(&data);
    let val_valid_starts = build_valid_starts(&val_data);
    if !valid_starts.is_empty() {
        let pct = 100.0 * valid_starts.len() as f64
            / data.len().saturating_sub(crate::config::BLOCK_SIZE + 1) as f64;
        println!("Doc-boundary sampling: {} valid train windows ({:.1}% of total)",
            valid_starts.len(), pct);
    }
    println!();

    // ── Initialize model ──────────────────────────────────────────────
    println!("Initializing model...");
    let mut model = GPTModel::new(tokenizer.vocab_size, &mut rng);

    let param_count = model.wte.len() + model.wpe.len() + model.lm_head.len()
        + N_LAYER * (
            model.layers[0].wq.len() + model.layers[0].wk.len()
            + model.layers[0].wv.len() + model.layers[0].wo.len()
            + model.layers[0].fc1.len() + model.layers[0].fc2.len()
        );
    println!("Parameters: ~{:.2}M", param_count as f32 / 1_000_000.0);
    println!();

    // ── Force Metal init so we know which path to take ───────────────
    let use_metal = METAL_DEVICE.is_some();
    if use_metal {
        println!("Metal GPU: enabled — training via Candle autograd");
    } else {
        println!("Metal GPU: unavailable — training on CPU (BLAS)");
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
        let result: std::io::Result<(usize, usize, f32)> = if use_metal {
            let device = METAL_DEVICE.as_ref().unwrap();

            // Try RGPT0003 (full GPU state)
            let r3 = {
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
                r3
            } else {
                // Try RGPT0002 (weights only, moments reset to zero)
                let r2 = {
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
                if r2.is_ok() { r2 } else { load_checkpoint(ckpt, &mut model) }
            }
        } else {
            load_checkpoint(ckpt, &mut model)
        };

        match result {
            Ok((it, st, bl)) => {
                if fine_tune {
                    println!("✓ Loaded weights from '{}' (fine-tune: iter/step/best reset)", ckpt);
                    println!();
                    (0, 0, f32::INFINITY)
                } else {
                    println!("✓ Resumed from '{}' — iter {}, step {}, best loss {:.4}", ckpt, it, st, bl);
                    println!();
                    (it, st, bl)
                }
            }
            Err(e) => {
                eprintln!("Error loading checkpoint '{}': {}", ckpt, e);
                eprintln!("Starting from scratch instead.");
                (0, 0, f32::INFINITY)
            }
        }
    } else {
        (0, 0, f32::INFINITY)
    };

    // ── Sync resumed weights to CPU model ──────────────────────────
    // When resuming on Metal, the loaded weights live in the CandleModel.
    // Sync them back to `model` now so estimate_loss / generate sees the
    // actual checkpoint state, not freshly-initialized random weights.
    if let Some((ref cm, _, _, _, _)) = candle_resume {
        model = cm.to_gpt()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
    }

    if iter_start >= iterations {
        println!("Already at iteration {} (target {}). Nothing to train.", iter_start, iterations);
        println!("Increase --iters to continue training.");
        return Ok(());
    }

    // ── Initial loss estimate ─────────────────────────────────────────
    println!("Estimating initial loss...");
    let initial_loss     = estimate_loss(&model, &data, &valid_starts, 50, &mut rng);
    let initial_val_loss = estimate_loss(&model, &val_data, &val_valid_starts, 50, &mut rng);
    println!("Initial loss: {:.4} | Val: {:.4} (ppl {:.1})",
        initial_loss, initial_val_loss, initial_val_loss.exp());
    println!();


    // ── Train ─────────────────────────────────────────────────────────
    if use_metal {
        let device = METAL_DEVICE.as_ref().unwrap();
        let (mut candle_model, mut opt) = if let Some((cm, o, _, _, _)) = candle_resume {
            if fine_tune {
                // Keep weights, discard moments — stale Gutenberg moments cause NaN on new domain
                let vars = cm.all_vars();
                let fresh_opt = GpuAdamState::new(&vars)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
                println!("Fine-tune: optimizer moments reset to zero.");
                (cm, fresh_opt)
            } else {
                (cm, o)
            }
        } else {
            let cm = CandleModel::from_gpt(&model, device)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
            let vars = cm.all_vars();
            let o = GpuAdamState::new(&vars)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
            (cm, o)
        };
        // Sync step_t so bias correction starts correctly
        opt.step_t = step_start;
        train_candle(&mut candle_model, &mut opt, &data, &val_data,
            &valid_starts, &val_valid_starts,
            iterations, &mut rng,
            iter_start, step_start, best_loss_start, lr, min_lr,
            &checkpoint_prefix);
        model = candle_model.to_gpt()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
    } else {
        train(&mut model, &data, &val_data, &valid_starts, &val_valid_starts,
            iterations, &mut rng,
            iter_start, step_start, best_loss_start, lr, min_lr,
            &checkpoint_prefix);
    }

    // ── Final loss estimate ───────────────────────────────────────────
    println!("Estimating final loss...");
    let final_loss     = estimate_loss(&model, &data, &valid_starts, 50, &mut rng);
    let final_val_loss = estimate_loss(&model, &val_data, &val_valid_starts, 50, &mut rng);
    println!("Final train loss: {:.4} (started {:.4})", final_loss, initial_loss);
    println!("Final val loss:   {:.4} (ppl {:.1}, started {:.4})",
        final_val_loss, final_val_loss.exp(), initial_val_loss);
    println!();

    // ── Generate samples ──────────────────────────────────────────────
    println!("=== Generation After Training ===");
    for (prompt, max_tokens) in &[("ROMEO:", 100), ("To be or not to be", 100), ("Once upon a time", 100)] {
        println!("\nPrompt: \"{}\"", prompt);
        let sample = generate(&model, &tokenizer, prompt, *max_tokens, 0.8, 0.9, &mut rng);
        println!("{}", sample);
    }

    Ok(())
}
