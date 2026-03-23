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

#[cfg(feature = "model-xs")]
pub const N_EMBD:  usize = 116;
#[cfg(feature = "model-xs")]
pub const N_HEAD:  usize = 4;
#[cfg(feature = "model-xs")]
pub const N_LAYER: usize = 3;

#[cfg(feature = "model-s")]
pub const N_EMBD:  usize = 128;
#[cfg(feature = "model-s")]
pub const N_HEAD:  usize = 4;
#[cfg(feature = "model-s")]
pub const N_LAYER: usize = 8;

#[cfg(feature = "model-ds")]
pub const N_EMBD:  usize = 128;
#[cfg(feature = "model-ds")]
pub const N_HEAD:  usize = 4;
#[cfg(feature = "model-ds")]
pub const N_LAYER: usize = 12;

#[cfg(feature = "model-m")]
pub const N_EMBD:  usize = 192;
#[cfg(feature = "model-m")]
pub const N_HEAD:  usize = 6;
#[cfg(feature = "model-m")]
pub const N_LAYER: usize = 6;

#[cfg(feature = "model-deep")]
pub const N_EMBD:  usize = 192;
#[cfg(feature = "model-deep")]
pub const N_HEAD:  usize = 6;
#[cfg(feature = "model-deep")]
pub const N_LAYER: usize = 16;

#[cfg(feature = "model-l")]
pub const N_EMBD:  usize = 256;
#[cfg(feature = "model-l")]
pub const N_HEAD:  usize = 8;
#[cfg(feature = "model-l")]
pub const N_LAYER: usize = 6;

#[cfg(feature = "model-xl")]
pub const N_EMBD:  usize = 384;
#[cfg(feature = "model-xl")]
pub const N_HEAD:  usize = 8;
#[cfg(feature = "model-xl")]
pub const N_LAYER: usize = 8;

// Default (model-xs): 116-dim, 4-head, 3-layer — ~0.86M params  ← default
#[cfg(not(any(feature = "model-xs", feature = "model-s", feature = "model-ds", feature = "model-m", feature = "model-l", feature = "model-deep", feature = "model-xl")))]
pub const N_EMBD:  usize = 116;
#[cfg(not(any(feature = "model-xs", feature = "model-s", feature = "model-ds", feature = "model-m", feature = "model-l", feature = "model-deep", feature = "model-xl")))]
pub const N_HEAD:  usize = 4;
#[cfg(not(any(feature = "model-xs", feature = "model-s", feature = "model-ds", feature = "model-m", feature = "model-l", feature = "model-deep", feature = "model-xl")))]
pub const N_LAYER: usize = 3;

pub const BLOCK_SIZE: usize = 256;
pub const HEAD_DIM:   usize = N_EMBD / N_HEAD;
pub const MLP_DIM:    usize = 4 * N_EMBD;
pub const MAX_VOCAB:  usize = 8192;   // raised for BPE (char-level uses ~117)

// ── BPE tokenizer ─────────────────────────────────────────────────────────
pub const BPE_VOCAB_SIZE: usize = 2000; // default target vocab for --bpe mode
pub const BPE_VOCAB_PATH: &str  = "vocab.json";

// ── Training ──────────────────────────────────────────────────────────────

// Per-model batch size defaults — smaller models have memory headroom for larger batches.
#[cfg(feature = "model-xs")]
pub const BATCH_SIZE: usize = 64;
#[cfg(feature = "model-s")]
pub const BATCH_SIZE: usize = 64;
#[cfg(feature = "model-ds")]
pub const BATCH_SIZE: usize = 64;
#[cfg(feature = "model-m")]
pub const BATCH_SIZE: usize = 64;
#[cfg(feature = "model-l")]
pub const BATCH_SIZE: usize = 64;
#[cfg(feature = "model-deep")]
pub const BATCH_SIZE: usize = 16;
#[cfg(not(any(feature = "model-xs", feature = "model-s", feature = "model-ds", feature = "model-m", feature = "model-l", feature = "model-deep")))]
pub const BATCH_SIZE: usize = 64;

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
