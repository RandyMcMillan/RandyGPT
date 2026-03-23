use serde::Deserialize;
use std::fs::read_to_string;
use toml;

pub const DEFAULT_CONFIG_TOML: &[u8] = include_bytes!("RandyGPT.toml");

#[derive(Debug, Deserialize, Default)]
pub struct RandyGPTConfig {
    pub n_embd: Option<usize>,
    pub n_head: Option<usize>,
    pub n_layer: Option<usize>,
    pub block_size: Option<usize>,
    pub max_vocab: Option<usize>,
    pub batch_size: Option<usize>,
    pub bpe_vocab_path: Option<String>,
}

pub fn load_config() -> RandyGPTConfig {
    match read_to_string("src/RandyGPT.toml") {
        Ok(content) => match toml::from_str(&content) {
            Ok(config) => config,
            Err(e) => {
                eprintln!("Warning: Could not parse src/RandyGPT.toml from filesystem: {}. Attempting to use embedded default configuration.", e);
                // Fallback to embedded default
                match toml::from_str(std::str::from_utf8(DEFAULT_CONFIG_TOML).unwrap()) {
                    Ok(config) => config,
                    Err(e) => {
                        eprintln!("Error: Could not parse embedded default RandyGPT.toml: {}. Using empty default configuration.", e);
                        RandyGPTConfig::default()
                    }
                }
            }
        },
        Err(e) => {
            eprintln!("Warning: Could not read src/RandyGPT.toml from filesystem: {}. Attempting to use embedded default configuration.", e);
            // Fallback to embedded default
            match toml::from_str(std::str::from_utf8(DEFAULT_CONFIG_TOML).unwrap()) {
                Ok(config) => config,
                Err(e) => {
                    eprintln!("Error: Could not parse embedded default RandyGPT.toml: {}. Using empty default configuration.", e);
                    RandyGPTConfig::default()
                }
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Hyperparameters and global constants                               */
/* ------------------------------------------------------------------ */
//
// Model size presets — select at build time with:
//   cargo build --release --features model-xs    (~746K params)   — 116-dim, 4-head,  3-layer
//   cargo build --release --features model-s     (~1.6M params)   — 128-dim, 4-head,  8-layer
//   cargo build --release --features model-ds    (~2.78M params)  — 128-dim, 4-head, 12-layer (deep-s)
//   cargo build --release --features model-m     (~2.7M params, ~1100ms/iter)
//   cargo build --release --features model-l     (~4.82M params, ~1835ms/iter)
//   cargo build --release --features model-deep  (~7.5M params)   — 192-dim, 6-head, 16-layer
//   cargo build --release --features model-xl    (~10.8M params, ~4000ms/iter)
//
// All presets use BLOCK_SIZE=256; BATCH_SIZE varies per model.
// Checkpoints are NOT cross-compatible between sizes (different weight shapes).

// ── Architecture ──────────────────────────────────────────────────────────

pub static mut N_EMBD:  usize = 116;
pub static mut N_HEAD:  usize = 4;
pub static mut N_LAYER: usize = 3;

pub static mut BLOCK_SIZE: usize = 256;
pub static mut HEAD_DIM:   usize = 0;   // Calculated dynamically
pub static mut MLP_DIM:    usize = 0;   // Calculated dynamically
pub static mut MAX_VOCAB:  usize = 8192;   // raised for BPE (char-level uses ~117)

// ── BPE tokenizer ─────────────────────────────────────────────────────────
pub const BPE_VOCAB_SIZE: usize = 2000; // default target vocab for --bpe mode
pub static mut BPE_VOCAB_PATH: String  = String::new();

// ── Training ──────────────────────────────────────────────────────────────

// Per-model batch size defaults — smaller models have memory headroom for larger batches.
pub static mut BATCH_SIZE: usize = 64;

// Gradient accumulation — kept at 1 for all models.
// accum>1 causes Metal GPU stalls (system freezes ~10s, interrupts blocked).
pub const GRAD_ACCUM_STEPS: usize = 1;
pub const LEARNING_RATE: f32 = 1e-4;       // raised from 3e-5; ~4× faster convergence
pub const MIN_LEARNING_RATE: f32 = 1e-5;   // 10% of max (was 3e-6)
pub const WEIGHT_DECAY: f32 = 0.1;         // raised from 0.01; more regularization for small dataset
pub const DROPOUT_RATE: f32 = 0.1;
pub const BETA1: f32 = 0.9;
pub const BETA2: f32 = 0.999;
pub const EPSILON: f32 = 1e-8;
pub const MAX_ITERS: usize = 1000;
pub const EVAL_INTERVAL: usize = 25;
pub const GRAD_CLIP: f32 = 1.0;
// Early stopping: halt if val loss hasn't improved for this many eval intervals.
// Set to 0 to disable. E.g. patience=20 + EVAL_INTERVAL=25 → stops after
// 500 consecutive iters with no val improvement.
pub const EARLY_STOP_PATIENCE: usize = 30;
// ReduceLROnPlateau: when patience is exhausted, reduce max_lr by this factor
// and reset patience instead of stopping.  Repeats up to MAX_LR_REDUCTIONS
// times before giving up entirely.
pub const LR_REDUCTION_FACTOR: f32  = 0.5;
pub const MAX_LR_REDUCTIONS:   usize = 3;

// ── Metal ─────────────────────────────────────────────────────────────────

pub const USE_METAL: bool = false;
#[allow(dead_code)]
pub const CANDLE_TRAIN: bool = true; // use Candle autograd for training when Metal available
