#!/bin/bash

# Detect OS and install correct gcc compiler for x86_64-unknown-linux-gnu
if command -v brew &> /dev/null
then
    echo "macOS detected. Installing gcc-x86_64-linux-gnu via Homebrew..."
    brew install gcc-x86_64-linux-gnu
elif command -v apt-get &> /dev/null
then
    echo "Debian/Ubuntu detected. Installing gcc-x86-64-linux-gnu via apt-get..."
    sudo apt-get update
    sudo apt-get install -y gcc-x86-64-linux-gnu
else
    echo "Unsupported OS or package manager. Please install x86_64-linux-gnu-gcc manually."
    exit 1
fi

rustup target add x86_64-unknown-linux-gnu || true;
cargo build --target x86_64-unknown-linux-gnu
