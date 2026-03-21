fn main() {
    #[cfg(target_os = "macos")]
    {
        // For macOS ARM (Apple Silicon)
        #[cfg(target_arch = "aarch64")]
        {
            println!("cargo:rustc-link-lib=framework=Accelerate");
        }
        // For macOS Intel
        #[cfg(target_arch = "x86_64")]
        {
            println!("cargo:rustc-link-lib=framework=Accelerate");
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        // Explicitly link MKL runtime library for non-macOS targets (e.g., Linux with MKL feature)
        println!("cargo:rustc-link-lib=mkl_rt");
    }
}
