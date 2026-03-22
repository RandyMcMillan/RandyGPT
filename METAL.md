# Metal (GPU Acceleration) in RandyGPT

RandyGPT leverages Apple's Metal framework for GPU acceleration through the `candle-core` library. This document provides an overview of Metal compatibility and how RandyGPT utilizes it.

## Metal Compatibility on macOS

Metal's availability and feature set depend on both your Mac's hardware (GPU) and the macOS version installed.

| Metal Version | macOS Requirement | Primary Hardware Support                                |
| :------------ | :---------------- | :------------------------------------------------------ |
| **Metal 1 & 2** | OS X El Capitan (10.11) or later | Most Macs introduced in **2012 or later**.           |
| **Metal 3**   | macOS Ventura (13) or later | Most Macs introduced in **2017 or later**.           |
| **Metal 4**   | macOS Tahoe (26) or later | Primarily **Apple Silicon** Macs (M1, M2, M3, etc.). Intel-based Macs generally do not support the full Metal 4 feature set. |

**Important Note:** Since macOS Mojave (10.14), a Metal-capable graphics card has been a strict requirement for the operating system itself.

## RandyGPT's Metal Initialization and Fallback

RandyGPT attempts to utilize Metal for accelerated computations if available. This is managed by the `METAL_DEVICE` lazy static in `src/metal.rs` and the `USE_METAL` configuration.

1.  **Detection**: The `build.rs` script dynamically sets `MACOSX_DEPLOYMENT_TARGET` to 13.0 (or a user-defined environment variable) to ensure compatibility with Metal 3 and newer macOS versions.
2.  **Initialization Attempt**: The application attempts to initialize a Metal device using `Device::metal_if_available(0)` from the `candle-core` library.
3.  **Graceful Fallback**:
    *   If a Metal device is successfully initialized, RandyGPT will use the Metal GPU for its computations, printing `✓ Metal GPU enabled on device: ...`.
    *   If `Device::metal_if_available(0)` fails (e.g., no Metal device found, or an error occurs during initialization), RandyGPT will gracefully fall back to using the CPU (BLAS) for computations, printing `⚠ Metal GPU unavailable: ...` or `⚠ Metal GPU unavailable (no Metal device found).`.
    *   The application is designed to function correctly regardless of Metal's availability.

## How to Check Your Mac's Metal Support

You can check your specific Mac's Metal capabilities:

1.  Press and hold the **Option** key and click the the **Apple menu ()** in the top-left corner of your screen.
2.  Select **System Information** from the dropdown menu.
3.  In the sidebar of the System Information window, navigate to **Graphics/Displays**.
4.  In the right-hand pane, look for **Metal Support** or **Metal Family**. This will show you the specific Metal version or feature set supported by your Mac's GPU.
