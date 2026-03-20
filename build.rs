#[cfg(target_os = "macos")]
fn main() {
    // Link against Apple's Accelerate framework for BLAS (cblas_sgemv, cblas_sger)
    println!("cargo:rustc-link-lib=framework=Accelerate");
}
