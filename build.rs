fn main() {
    #[cfg(target_os = "macos")]
    {
        let deployment_target = std::env::var("MACOSX_DEPLOYMENT_TARGET")
            .unwrap_or_else(|_| "13.0".to_string());
        println!("cargo:rustc-env=MACOSX_DEPLOYMENT_TARGET={}", deployment_target);
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
        #[cfg(target_arch = "aarch64")]
        {
            // println!("cargo:rustc-link-lib=mkl_rt");
        }
        #[cfg(target_arch = "x86_64")]
        {
            println!("cargo:rustc-link-lib=openblas");
        }
    }
}
