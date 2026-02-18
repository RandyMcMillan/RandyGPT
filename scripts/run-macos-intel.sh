#!/bin/bash

# Method A: The standard Bash way
# We use '2>/dev/null' to hide errors if the command doesn't exist
NPROC=$(nproc 2>/dev/null || echo 1)

echo "Number of processors (Method A): $NPROC"

# Method B: Using your preferred expansion syntax (requires 2 steps)
RAW_COUNT=$(nproc 2>/dev/null)
NPROC_EXPANDED=${RAW_COUNT:-1}

echo "Number of processors (Method B): $NPROC_EXPANDED"

# Method C: macOS (Intel/Apple Silicon) specific alternative
# Since you're on a Mac (DeepSpaceMBPro), nproc isn't always available by default.
if ! command -v nproc &> /dev/null; then
    NPROC=$(sysctl -n hw.ncpu)
fi

echo "Final confirmed NPROC for your hardware: $NPROC"

# 1. Clean environment and STALE CACHE
# This prevents the "swap_remove" panic by ensuring old, tiny token
# files aren't loaded into the new 256-block-size architecture.
echo "Cleaning stale cache and checkpoints..."
rm -f tokens.bin vocab.json checkpoint.bin checkpoint_best.bin

# 2. Verify train.txt exists

git diff 4b825dc642cb6eb9a060e54bf8d69288fbee4904 HEAD > repo.patch

touch train.txt || true

FILE_SIZE=$(wc -c < "train.txt" 2>/dev/null || echo 0)
if [ "$FILE_SIZE" -lt 10000 ]; then
    echo "Data too small ($FILE_SIZE bytes). Appending Shakespeare for stability..."
    curl -L https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt >> train.txt || \
    cat repo.patch >> train.txt
fi

echo "Switching to Stable Rust..."
rustup install stable
rustup override set stable

# 3. Run randyGPT with BPE enabled
# We use 10 iterations just to trigger the file creation.
# Note: The 'vocab size' will be set to the default (up to 8192)
echo "Starting BPE Tokenizer training and model initialization..."
cargo build -j$NPROC && \
cargo run -j$NPROC -- --bpe --iters 10

# 4. Verbose verification of generated files
echo -e "\n--- File Check ---"
if [ -f "vocab.json" ]; then
    echo "✅ SUCCESS: vocab.json created (Size: $(du -h vocab.json | cut -f1))"
else
    echo "❌ ERROR: vocab.json not found."
fi

if [ -f "tokens.bin" ]; then
    echo "✅ SUCCESS: tokens.bin created (Size: $(du -h tokens.bin | cut -f1))"
else
    echo "❌ ERROR: tokens.bin not found."
fi

# 5. List the root directory to confirm
echo -e "\nCurrent Directory Listing:"
ls -F | grep -E '(.json|.bin|train.txt)'

# 3. Forced "Release-Type" Cargo.toml
# Optimized for Intel Iris Plus Graphics via [profile.dev]
echo "Configuring Cargo.toml for optimized Intel Mac performance..."
cat <<EOF > Cargo.toml
[package]
name = "randygpt"
version = "0.9.2"
edition = "2021"
default-run = "randygpt"

[features]
# Model size presets — select with: cargo build --release --features model-s
# Default (no feature) = model-l: 256-dim, 8-head, 6-layer, ~4.82M params
model-xs = []   # 116-dim, 4-head, 3-layer, ~746K params
model-s  = []   # 128-dim, 4-head, 8-layer, ~1.6M params
model-m  = []   # 192-dim, 6-head, 6-layer, ~2.7M params, ~1100ms/iter
model-deep = [] # 192-dim, 6-head, 16-layer, ~7.5M params — depth experiment
model-xl = []   # 384-dim, 8-head, 8-layer, ~10.8M params, ~4000ms/iter

[dependencies]
candle-core = { version = "0.9.2", features = ["metal"] }
candle-nn = "0.9.2"
ctrlc = "3.4"
lazy_static = "1.5.0"
rand = "0.8.0"
rayon = "1.8"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

[profile.dev]
opt-level = 3
lto = true
codegen-units = 1
debug = false

[profile.release]
opt-level = 3
lto = true
codegen-units = 1

[dev-dependencies]
code2prompt = "4.2.0"
EOF

cargo update

# 4. Run the model
# Model-L (~4.82M params) is the sweet spot for 1.5GB VRAM.
# Using --iters 10000 to allow for significant convergence.
echo "Starting training on Metal GPU..."
cargo run -j$NPROC -- --iters 10000 #--generate
