//! CUDA Rust Tile operation conformance runner (NVIDIA cutile 0.5.0).
//!
//! These are reusable mathematical operators, not a DINOv2 forward.
//! GPU execution MUST fail if no compatible device or toolkit is available.
//! This program deliberately performs GPU work, with no CPU substitute.
use cutile::prelude::*;
use cutile::tensor::Reshape;
use std::sync::Arc;

#[path = "../../motor_inferencia/src/operation_contract.rs"]
mod operation_contract;
use operation_contract::{Device as ContractDevice, Kernel, KernelSpec, TensorSpec};

fn gpu_spec(kernel: Kernel, inputs: &[&[usize]], output: &[usize]) {
    let request = KernelSpec {
        kernel,
        inputs: inputs
            .iter()
            .map(|dims| TensorSpec::f32(dims, ContractDevice::Cuda))
            .collect(),
        output: TensorSpec::f32(output, ContractDevice::Cuda),
    };
    request
        .require_device(ContractDevice::Cuda)
        .expect("invalid GPU operation contract; no CPU fallback");
}

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    fn residual_add<const B: i32>(
        out: &mut Tensor<f32, { [B] }>,
        left: &Tensor<f32, { [-1] }>,
        right: &Tensor<f32, { [-1] }>,
    ) {
        out.store(left.load_like(out) + right.load_like(out));
    }

    /// Fused, stable FP32 softmax over the final axis (one full row per tile).
    #[cutile::entry()]
    fn row_softmax<const BM: i32, const BN: i32>(
        out: &mut Tensor<f32, { [BM, BN] }>,
        input: &Tensor<f32, { [-1, -1] }>,
    ) {
        let x: Tile<f32, { [BM, BN] }> = input.load_like(out);
        let max: Tile<f32, { [BM] }> = reduce_max(x, 1_i32);
        let max = max.reshape(shape![BM, 1]).broadcast(out.shape());
        let numerator = exp(x - max);
        let denominator: Tile<f32, { [BM] }> = reduce_sum(numerator, 1_i32);
        let denominator = denominator.reshape(shape![BM, 1]).broadcast(out.shape());
        out.store(true_div(numerator, denominator));
    }

    /// Tiled matrix multiplication (A[M,K] * B[K,N]) using GPU MMA.
    /// Shape requirements: M % BM == 0, N % BN == 0, K % BK == 0.
    #[cutile::entry()]
    fn matmul<E: ElementType, const BM: i32, const BN: i32, const BK: i32, const K: i32>(
        out: &mut Tensor<E, { [BM, BN] }>,
        left: &Tensor<E, { [-1, K] }>,
        right: &Tensor<E, { [K, -1] }>,
    ) {
        let lhs = left.partition(shape![BM, BK]);
        let rhs = right.partition(shape![BK, BN]);
        let row = program_id(0);
        let col = program_id(1);
        let mut value = load_tile_mut(out);
        for k in 0_i32..(K / BK) {
            value = mma(lhs.load([row, k]), rhs.load([k, col]), value);
        }
        out.store(value);
    }
}

fn residual_probe() -> Result<(), Error> {
    const N: usize = 1370 * 768;
    const TILE: usize = 128;
    assert_eq!(N % TILE, 0);
    gpu_spec(Kernel::ResidualAdd, &[&[N], &[N]], &[N]);
    let host_a: Arc<Vec<f32>> = Arc::new(
        (0..N).map(|i| ((i % 101) as f32 - 50.0) * 0.03125).collect(),
    );
    let host_b: Arc<Vec<f32>> = Arc::new(
        (0..N).map(|i| ((i % 71) as f32 - 35.0) * 0.0625).collect(),
    );
    let a = api::copy_host_vec_to_device(&host_a);
    let b = api::copy_host_vec_to_device(&host_b);
    let out = api::zeros::<f32>(&[N]).partition([TILE]);
    let (out, _, _) = kernels::residual_add(out, a, b).sync()?;
    let values: Vec<f32> = out.unpartition().to_host_vec().sync()?;
    assert_eq!(values.len(), N);
    for (i, value) in values.iter().enumerate() {
        assert!(value.is_finite() && (*value - (host_a[i] + host_b[i])).abs() <= 1e-6);
    }
    println!("PASS residual_add: {N} FP32 elements");
    Ok(())
}

fn softmax_probe() -> Result<(), Error> {
    const M: usize = 4;
    const N: usize = 128;
    gpu_spec(Kernel::SoftmaxLast, &[&[M, N]], &[M, N]);
    // Large positive logits ensure the fused kernel subtracts row maxima.
    let host: Arc<Vec<f32>> = Arc::new(
        (0..M * N)
            .map(|i| (i / N) as f32 * 10000.0 + ((i % N) as f32 - 64.0) * 0.125)
            .collect(),
    );
    let x = api::copy_host_vec_to_device(&host).sync()?.reshape(&[M, N])?;
    let out = api::zeros::<f32>(&[M, N]).partition([1, N]);
    let (out, _) = kernels::row_softmax(out, x).sync()?;
    let values: Vec<f32> = out.unpartition().to_host_vec().sync()?;
    assert_eq!(values.len(), M * N);
    for r in 0..M {
        let row_in = &host[r * N..(r + 1) * N];
        let row_out = &values[r * N..(r + 1) * N];
        let max = row_in.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let denominator: f32 = row_in.iter().map(|x| (x - max).exp()).sum();
        for (input, output) in row_in.iter().zip(row_out.iter()) {
            let expected = (input - max).exp() / denominator;
            assert!(output.is_finite() && (output - expected).abs() <= 1e-5);
        }
        let sum: f32 = row_out.iter().sum();
        assert!((sum - 1.0).abs() <= 1e-5);
    }
    println!("PASS fused_softmax: [{M},{N}] FP32, rows normalized");
    Ok(())
}

fn matmul_probe() -> Result<(), Error> {
    const M: usize = 32;
    const N: usize = 32;
    const K: usize = 64;
    const BM: usize = 16;
    const BN: usize = 16;
    const BK: usize = 8;
    gpu_spec(Kernel::Gemm, &[&[M, K], &[K, N]], &[M, N]);
    let host_x: Arc<Vec<f32>> = Arc::new(
        (0..M * K)
            .map(|idx| (((idx / K + idx % K) % 13) as i32 - 6) as f32 / 8.0)
            .collect(),
    );
    let host_y: Arc<Vec<f32>> = Arc::new(
        (0..K * N)
            .map(|idx| (((2 * (idx / N) + idx % N) % 11) as i32 - 5) as f32 / 8.0)
            .collect(),
    );
    let x = api::copy_host_vec_to_device(&host_x).sync()?.reshape(&[M, K])?;
    let y = api::copy_host_vec_to_device(&host_y).sync()?.reshape(&[K, N])?;
    let out = api::zeros::<f32>(&[M, N]).partition([BM, BN]);
    let generics = vec![
        "f32".to_string(),
        BM.to_string(),
        BN.to_string(),
        BK.to_string(),
        K.to_string(),
    ];
    let (out, _, _) = kernels::matmul(out, x, y).generics(generics).sync()?;
    let actual: Vec<f32> = out.unpartition().to_host_vec().sync()?;
    assert_eq!(actual.len(), M * N);
    for row in 0..M {
        for col in 0..N {
            let expected: f32 = (0..K)
                .map(|k| host_x[row * K + k] * host_y[k * N + col])
                .sum();
            let value = actual[row * N + col];
            assert!(value.is_finite() && (value - expected).abs() <= 1e-3);
        }
    }
    println!("PASS matmul: [{M},{K}] × [{K},{N}] = [{M},{N}] FP32");
    Ok(())
}

fn main() -> Result<(), Error> {
    match std::env::args().nth(1).as_deref() {
        None | Some("all") => {
            residual_probe()?;
            softmax_probe()?;
            matmul_probe()?;
        }
        Some("residual") => residual_probe()?,
        Some("softmax") => softmax_probe()?,
        Some("matmul") => matmul_probe()?,
        Some("--help") | Some("-h") => {
            println!("GPU-only conformance probes: [all|residual|softmax|matmul]");
            return Ok(());
        }
        Some(x) => {
            eprintln!("Unknown probe: {x}. Expected all|residual|softmax|matmul");
            std::process::exit(2);
        }
    };
    println!("Native CUDA Rust kernel probes passed. Full-model inference is NOT implemented.");
    Ok(())
}
