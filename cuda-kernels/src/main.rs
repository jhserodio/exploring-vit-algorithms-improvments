//! First model-independent GPU operation: exact FP32 residual tensor addition.
//! Native CUDA Rust Tile / CUDA 13.3+, isolated from the CPU reference.
//! This is an operation conformance probe, NOT a DINOv2 GPU forward or OSCC result.
use cutile::prelude::*;

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    fn residual_add<const B: i32>(
        out: &mut Tensor<f32, { [B] }>,
        left: &Tensor<f32, { [-1] }>,
        right: &Tensor<f32, { [-1] }>,
    ) {
        let a = load_tile_like(left, out);
        let b = load_tile_like(right, out);
        out.store(a + b);
    }
}

fn main() -> Result<(), Error> {
    // 1370 tokens × 768 dimensions: native DINOv2 Base hidden-state shape.
    const N: usize = 1370 * 768;
    const TILE: usize = 128;
    assert_eq!(N % TILE, 0, "CUDA tile partition must cover the input exactly");
    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    let a = api::ones::<f32>(&[N]);
    let b = api::ones::<f32>(&[N]);
    let z = api::zeros::<f32>(&[N]).partition([TILE]);
    let actual: Vec<f32> = kernels::residual_add(z, a, b)
        .first()
        .unpartition()
        .to_host_vec()
        .sync_on(&stream)?;
    let bad = actual.iter().filter(|&&x| (x - 2.0).abs() > 1e-6).count();
    if bad > 0 {
        eprintln!("CUDA-Rust operation parity FAIL: {bad} mismatches of {N}");
        std::process::exit(1);
    }
    println!("CUDA-Rust residual_add PASSED: {N} FP32 values, {N} / {TILE} tiles");
    println!("This validates one operation only; full DINOv2 CUDA inference is pending.");
    Ok(())
}
