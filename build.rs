fn main() {
    #[cfg(target_os = "macos")]
    {
        // Link against Apple's Accelerate framework for BLAS (cblas_sgemv, cblas_sger)
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
    #[cfg(not(target_os = "macos"))]
    {
    }
}
