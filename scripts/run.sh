# 1. Clean environment
rm -f tokens.bin vocab.json checkpoint.bin checkpoint_best.bin train.txt

# 2. Download a high-quality dataset (approx 1MB)
curl -o train.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# 3. Ensure your Cargo.toml is set to force release-level optimizations
# (This fulfills your request for a copy-pasteable example)
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
EOF

# 4. Run the model
cargo run --features model-xl -- --iters 10000
