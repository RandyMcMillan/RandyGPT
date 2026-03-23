# Cross-Compilation with `cross`

This document provides instructions for using the `cross` tool to build and test this project for various target platforms, including macOS and Linux.

`cross` is a Rust cross-compilation tool that leverages Docker/Podman containers to provide a consistent build environment for different targets.

## Prerequisites

Before you begin, ensure you have the following installed:

*   **Rustup:** For managing Rust toolchains.
*   **Docker or Podman:** `cross` requires a container engine to operate.
    *   **macOS:** Install Docker Desktop (https://docs.docker.com/desktop/install/mac-install/) or Podman Desktop (https://podman-desktop.io/docs/quickstarts/mac-quickstart). Ensure the container engine is running.
    *   **Linux:** Install Docker (https://docs.docker.com/engine/install/) or Podman (https://podman.io/docs/installation).

## Installation of `cross`

Install the `cross` tool using cargo:

```bash
cargo install cross --git https://github.com/cross-rs/cross
```

## Cross-Compilation Targets

Here are some common targets you might want to build for:

*   **Linux (x86_64):** `x86_64-unknown-linux-gnu`
*   **Linux (ARM64):** `aarch64-unknown-linux-gnu`
*   **macOS (x86_64 Intel):** `x86_64-apple-darwin`
*   **macOS (ARM64 Apple Silicon):** `aarch64-apple-darwin`
    *   **Note:** Cross-compiling *to* macOS targets from a Linux host using `cross` is generally more complex and might not be fully supported or straightforward due to Apple's proprietary frameworks. It's usually easier to build natively on macOS. The `macos-matrix.yml` workflow is designed for native macOS builds.

## Building and Testing with `cross`

You can build and test your project for a specific target using the `cross` command.

### General Command Structure

```bash
cross build --target <TARGET_TRIPLE> --features <FEATURE_NAME> --verbose
cross test --target <TARGET_TRIPLE> --features <FEATURE_NAME> --verbose
```

*   Replace `<TARGET_TRIPLE>` with the desired target (e.g., `x86_64-unknown-linux-gnu`).
*   Replace `<FEATURE_NAME>` with the specific feature you want to enable (e.g., `model-l`). Use `default` or omit `--features` for the default build.

### Examples

**1. Build for Linux (x86_64) with default features:**

```bash
cross build --target x86_64-unknown-linux-gnu --verbose
cross test --target x86_64-unknown-linux-gnu --verbose
```

**2. Build for Linux (ARM64) with `model-s` feature:**

```bash
cross build --target aarch64-unknown-linux-gnu --features model-s --verbose
cross test --target aarch64-unknown-linux-gnu --features model-s --verbose
```

**3. Build for macOS (Intel) from a Linux host (if supported and configured):**
(This is generally not recommended; prefer native builds on macOS for macOS targets.)

```bash
# This would require a highly specialized cross-toolchain and might not work
cross build --target x86_64-apple-darwin --verbose
```

## Managing Target-Specific Dependencies (`Cross.toml`)

For targets that require additional system packages (like BLAS libraries for `aarch64-unknown-linux-gnu`), `cross` can be configured using a `Cross.toml` file in the project root.

Example `Cross.toml` (already present in this project for `aarch64-unknown-linux-gnu`):

```toml
[target.aarch64-unknown-linux-gnu]
pre-build = ["apt-get update && apt-get install -y libopenblas-dev"]
```

This ensures that `libopenblas-dev` is installed within the Docker container specifically for the `aarch64-unknown-linux-gnu` target before the build proceeds.
