use candle_core::{Device, Result, Tensor};

fn main() -> Result<()> {
    println!("🔬 Iniciando motor de inferência ViT...");

    // Verifica se o Rust consegue enxergar e assumir o controle da RTX 5070 Ti (Index 0)
    let device = Device::new_cuda(0)?;
    println!("✅ GPU detectada e alocada com sucesso.");

    // Cria uma matriz 3x3 de números de ponto flutuante diretamente na VRAM
    let tensor = Tensor::zeros((3, 3), candle_core::DType::F32, &device)?;

    println!("✅ Tensor alocado fisicamente na GPU:");
    println!("{:?}", tensor);

    Ok(())
}
