use crate::config::{MAX_ITERS, BPE_VOCAB_SIZE, BPE_VOCAB_PATH};
use log::{debug};

#[derive(Debug)]
pub struct Cli {
    pub iterations: usize,
    pub resume_path: Option<String>,
    pub lr_override: Option<f32>,
    pub min_lr_override: Option<f32>,
    pub bpe_vocab_size: Option<usize>,
    pub generate_mode: bool,
    pub generate_prompts: Vec<String>,
    pub serve_mode: bool,
    pub serve_addr: Option<String>,
    pub api_key: Option<String>,
    pub train_file: String,
    pub vocab_path: String,
    pub checkpoint_prefix_arg: Option<String>,
    pub fine_tune: bool,
    pub gen_max_tokens: usize,
    pub gen_temperature: f32,
    pub gen_top_k: f32,
    pub config_path_arg: Option<String>,
    pub model_xs: bool,
    pub model_s: bool,
    pub model_ds: bool,
    pub model_m: bool,
    pub model_l: bool,
    pub model_deep: bool,
    pub model_xl: bool,
}

impl Default for Cli {
    fn default() -> Self {
        Self {
            iterations: MAX_ITERS ,
            resume_path: None,
            lr_override: None,
            min_lr_override: None,
            bpe_vocab_size: None,
            generate_mode: false,
            generate_prompts: Vec::new(),
            serve_mode: false,
            serve_addr: None,
            api_key: None,
            train_file: "train.txt".to_string(),
            vocab_path: (*(&raw const BPE_VOCAB_PATH)).clone(),
            checkpoint_prefix_arg: None,
            fine_tune: false,
            gen_max_tokens: 200,
            gen_temperature: 0.8,
            gen_top_k: 0.9,
            config_path_arg: None,
            model_xs: false,
            model_s: false,
            model_ds: false,
            model_m: false,
            model_l: false,
            model_deep: false,
            model_xl: false,
        }
    }
}

pub fn parse_args() -> Cli {
    let args: Vec<String> = std::env::args().collect();
    let mut config_path_arg: Option<String> = None;

    let mut args_iter = args.into_iter().peekable();
    let mut filtered_args: Vec<String> = Vec::new();

    // First pass: extract --config and filter it out
    while let Some(arg) = args_iter.next() {
        if arg == "--config" {
            if let Some(config_val) = args_iter.next() {
                config_path_arg = Some(config_val.clone());
                debug!("Pre-parsed --config path: {:?}", config_path_arg);
            } else {
                eprintln!("Warning: --config argument provided without a value. Ignoring.");
            }
        } else {
            filtered_args.push(arg);
        }
    }

    let mut cli = Cli::default();
    cli.config_path_arg = config_path_arg; // Assign the pre-parsed config path

    // Second pass: parse remaining arguments
    let mut i = 1;
    while i < filtered_args.len() {
        match filtered_args[i].as_str() {
            "--iters" => {
                i += 1;
                if i < filtered_args.len() {
                    cli.iterations = filtered_args[i].parse().unwrap_or(MAX_ITERS );
                    debug!("Parsed --iters: {}", cli.iterations);
                }
            }
            "--resume" => {
                if i + 1 < filtered_args.len() && !filtered_args[i + 1].starts_with("--") {
                    i += 1;
                    cli.resume_path = Some(filtered_args[i].clone());
                    debug!("Parsed --resume path: {}", cli.resume_path.as_ref().unwrap());
                } else {
                    cli.resume_path = Some("__default__".to_string());
                    debug!("Parsed --resume with default path.");
                }
            }
            "--lr" => {
                i += 1;
                if i < filtered_args.len() {
                    cli.lr_override = filtered_args[i].parse().ok();
                    debug!("Parsed --lr: {:?}", cli.lr_override);
                }
            }
            "--min-lr" => {
                i += 1;
                if i < filtered_args.len() {
                    cli.min_lr_override = filtered_args[i].parse().ok();
                    debug!("Parsed --min-lr: {:?}", cli.min_lr_override);
                }
            }
            "--bpe" => {
                if i + 1 < filtered_args.len() && !filtered_args[i + 1].starts_with("--") {
                    i += 1;
                    cli.bpe_vocab_size = Some(filtered_args[i].parse().unwrap_or(BPE_VOCAB_SIZE ));
                    debug!("Parsed --bpe with custom size: {:?}", cli.bpe_vocab_size);
                } else {
                    cli.bpe_vocab_size = Some(BPE_VOCAB_SIZE );
                    debug!("Parsed --bpe with default size: {:?}", cli.bpe_vocab_size);
                }
            }
            "--generate" => {
                cli.generate_mode = true;
                debug!("Generate mode activated.");
                while i + 1 < filtered_args.len() && !filtered_args[i + 1].starts_with("--") {
                    i += 1;
                    cli.generate_prompts.push(filtered_args[i].clone());
                    debug!("Added generate prompt: {}", filtered_args[i]);
                }
            }
            "--serve" => {
                cli.serve_mode = true;
                debug!("Serve mode activated.");
                if i + 1 < filtered_args.len() && !filtered_args[i + 1].starts_with("--") {
                    i += 1;
                    cli.serve_addr = Some(filtered_args[i].clone());
                    debug!("Parsed --serve address: {:?}", cli.serve_addr);
                } else {
                    cli.serve_addr = Some("0.0.0.0:8080".to_string());
                    debug!("Parsed --serve with default address: {:?}", cli.serve_addr);
                }
            }
            "--api-key" => {
                i += 1;
                if i < filtered_args.len() {
                    cli.api_key = Some(filtered_args[i].clone());
                    debug!("Parsed --api-key (value omitted for security).");
                }
            }
            "--train-file" => {
                i += 1;
                if i < filtered_args.len() {
                    cli.train_file = filtered_args[i].clone();
                    debug!("Parsed --train-file: {}", cli.train_file);
                }
            }
            "--vocab" => {
                i += 1;
                if i < filtered_args.len() {
                    cli.vocab_path = filtered_args[i].clone();
                    debug!("Parsed --vocab: {}", cli.vocab_path);
                }
            }
            "--checkpoint" => {
                i += 1;
                if i < filtered_args.len() {
                    cli.checkpoint_prefix_arg = Some(filtered_args[i].trim_end_matches(".bin").to_string());
                    debug!("Parsed --checkpoint prefix: {:?}", cli.checkpoint_prefix_arg);
                }
            }
            "--fine-tune" => { cli.fine_tune = true; debug!("Parsed --fine-tune: true"); }
            "--model-xs" => { cli.model_xs = true; debug!("Parsed --model-xs."); }
            "--model-s" => { cli.model_s = true; debug!("Parsed --model-s."); }
            "--model-ds" => { cli.model_ds = true; debug!("Parsed --model-ds."); }
            "--model-m" => { cli.model_m = true; debug!("Parsed --model-m."); }
            "--model-l" => { cli.model_l = true; debug!("Parsed --model-l."); }
            "--model-deep" => { cli.model_deep = true; debug!("Parsed --model-deep."); }
            "--model-xl" => { cli.model_xl = true; debug!("Parsed --model-xl."); }
            "--max-tokens" => {
                i += 1;
                if i < filtered_args.len() {
                    cli.gen_max_tokens = filtered_args[i].parse().unwrap_or(200);
                    debug!("Parsed --max-tokens: {}", cli.gen_max_tokens);
                }
            }
            "--temperature" => {
                i += 1;
                if i < filtered_args.len() {
                    cli.gen_temperature = filtered_args[i].parse().unwrap_or(0.8);
                    debug!("Parsed --temperature: {}", cli.gen_temperature);
                }
            }
            "--top-k" => {
                i += 1;
                if i < filtered_args.len() {
                    cli.gen_top_k = filtered_args[i].parse().unwrap_or(0.9);
                    debug!("Parsed --top-k: {}", cli.gen_top_k);
                }
            }
            "--help" | "-h" => {
                println!("randyGPT — tiny GPT language model\n");
                println!("USAGE:");
                println!("  randygpt [OPTIONS]\n");
                println!("CONFIG:");
                println!("  --config PATH      Path to RandyGPT.toml config file (overrides defaults)");
                println!("\nMODEL PRESETS:");
                println!("  --model-xs         Apply extra-small model (116-dim, 4-head, 3-layer, batch 64)");
                println!("  --model-s          Apply small model (128-dim, 4-head, 8-layer, batch 64)");
                println!("  --model-ds         Apply deep-small model (128-dim, 4-head, 12-layer, batch 64)");
                println!("  --model-m          Apply medium model (192-dim, 6-head, 6-layer, batch 64)");
                println!("  --model-l          Apply large model (256-dim, 8-head, 8-layer, batch 64)");
                println!("  --model-deep       Apply deep model (192-dim, 6-head, 16-layer, batch 16)");
                println!("  --model-xl         Apply extra-large model (384-dim, 8-head, 8-layer, batch 64)");
                println!("\nTRAINING:");
                println!("  --iters N          Training iterations (default: {})", MAX_ITERS );
                println!("  --train-file PATH  Training text file (default: train.txt)");
                println!("  --vocab PATH       BPE vocab JSON file (default: {})", (*(&raw const BPE_VOCAB_PATH)).as_str());
                println!("  --checkpoint NAME  Checkpoint filename prefix (default: checkpoint)");
                println!("  --bpe [N]          Use BPE tokenizer, optional target vocab size (default: {})", BPE_VOCAB_SIZE );
                println!("                     If N is omitted, uses default BPE_VOCAB_SIZE.");
                println!("  --resume [PATH]    Resume from checkpoint (default: <prefix>_best.bin,");
                println!("                     where <prefix> is from --checkpoint or train-file).");
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
                if let Ok(n) = filtered_args[i].parse::<usize>() {
                    cli.iterations = n;
                    debug!("Parsed unrecognized numeric argument as iterations: {}", cli.iterations);
                } else {
                    eprintln!("Unknown argument '{}'. Ignoring.", other);
                    debug!("Unknown argument '{}'. Ignoring.", other);
                }
            }
        }
        i += 1;
    }
    cli
}
