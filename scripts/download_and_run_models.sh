#!/bin/bash

# This script downloads necessary training data and demonstrates how to train and
# run randygpt models with different configurations.

set -euo pipefail

DATA_DIR="./data"
TRAIN_FILE="${DATA_DIR}/gutenberg_cleaned_v3.txt"
VOCAB_FILE="./vocab_v3.json"

echo "Ensuring data directory exists..."
mkdir -p "${DATA_DIR}"

# --- Download Training Data ---
if [ ! -f "${TRAIN_FILE}" ]; then
    echo "Downloading and cleaning Gutenberg dataset (this may take a while)..."
    ./scripts/download_gutenberg.sh
    # Assuming download_gutenberg.sh places output in data/gutenberg_cleaned_v3.txt
else
    echo "Gutenberg dataset already present at ${TRAIN_FILE}. Skipping download."
fi

if [ ! -f "${VOCAB_FILE}" ]; then
    echo "Building BPE vocabulary (this may take a while)..."
    python3 ./scripts/build_bpe_vocab.py --input "${TRAIN_FILE}" --output "${VOCAB_FILE}"
else
    echo "BPE vocabulary already present at ${VOCAB_FILE}. Skipping build."
fi

echo ""
echo "--- Running RandyGPT Models ---"
echo ""

# --- Helper function for running randygpt ---
run_randygpt() {
    MODEL_SIZE=$1
    CHECKPOINT_PREFIX="checkpoint_${MODEL_SIZE}"
    echo "=================================================="
    echo "Running RandyGPT with model size: ${MODEL_SIZE}"
    echo "=================================================="

    echo "Training ${MODEL_SIZE} model..."
    cargo run --release --features "${MODEL_SIZE}" -- --bpe --train-file "${TRAIN_FILE}" --vocab "${VOCAB_FILE}" --checkpoint "${CHECKPOINT_PREFIX}" --iters 1000 --min-lr 1e-5
    echo ""

    echo "Generating text with ${MODEL_SIZE} model..."
    cargo run --release --features "${MODEL_SIZE}" -- --bpe --vocab "${VOCAB_FILE}" --resume "${CHECKPOINT_PREFIX}_best.bin" --generate "The quick brown fox" --max-tokens 50
    echo ""
}

# --- Examples for different model sizes ---
# Note: Training each model takes time. Adjust --iters for quicker demonstration.
# For full training, remove --iters or set to a high value.

# Small model example
run_randygpt "model-s"

# Medium model example
# run_randygpt "model-m"

# Deep model example
# run_randygpt "model-deep"

echo "All specified models have been processed."
echo "To run other model sizes, uncomment the relevant lines in this script."
echo "You can also adjust --iters for longer/shorter training."

echo ""
echo "--- Running Hugging Face BERT Model Example ---"
echo "This example downloads the 'sentence-transformers/all-MiniLM-L6-v2' model from Hugging Face Hub (approx. 90MB)"
echo "and uses it to compute sentence embeddings."
echo ""

cargo run --example hf_bert_inference --release

echo ""
echo "Hugging Face BERT model example finished."

echo ""
echo "--- Running Hugging Face Qwen3-Coder-Next Model Example ---"
echo "This example downloads the 'Qwen/Qwen3-Coder-Next-GGUF' model from Hugging Face Hub (approx. 45GB for Q4_K_M)"
echo "and uses it to generate code based on a prompt."
echo "Note: This model is very large and may require significant RAM/VRAM."
echo ""

cargo run --example hf_qwen_inference --release

echo ""
echo "Hugging Face Qwen3-Coder-Next model example finished."
