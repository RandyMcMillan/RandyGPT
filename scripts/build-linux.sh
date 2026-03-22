#!/bin/bash

# Detect OS and install correct gcc compiler for x86_64-unknown-linux-gnu
if command -v brew &> /dev/null
then
    echo "macOS detected. Installing native gcc via Homebrew..."
    brew install gcc
    # Set the cross-compiler to the Homebrew-installed gcc
    GCC_PATH=$(brew --prefix gcc)
    export CC_x86_64_unknown_linux_gnu="${GCC_PATH}/bin/gcc-$(ls -1 ${GCC_PATH}/bin | grep -o 'gcc-[0-9][0-9]' | sort -r | head -n 1 | cut -d'-' -f2)"
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
