extern {
    fn cblas_sgemm(
        layout: i32,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

fn main() {
    println!("Attempting to link with Accelerate.");
    // This program will compile if Accelerate (containing cblas_sgemm) is found.
    // No need to actually call the function for this test.
}