//! DINOv2 CPU reference CLI. Native CUDA Rust is being developed separately.
//! Never claim CPU execution to be a CUDA inference result.
use anyhow::{Context, Result, bail};
use motor_inferencia::{ModelAdapter, dinov2::DinoV2Base, preprocess};
use std::{
    env,
    io::Write,
    path::{Path, PathBuf},
};

fn usage() -> &'static str {
    "Usage: motor_inferencia --image IMAGE --weights MODEL.SAFETENSORS (--cpu | --cuda) [--export-prefix PATH]\n\
     --cpu runs the independent Rust reference DINOv2 Base and exports CLS + spatial features.\n\
     --cuda is reserved for the forthcoming native CUDA Rust full-model runner; it fails explicitly.\n\
     No OSCC classification or segmentation head is trained or provided."
}

fn write_f32_le(path: &Path, values: &[f32]) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let mut writer = std::io::BufWriter::new(std::fs::File::create(path)?);
    for value in values {
        writer.write_all(&value.to_le_bytes())?;
    }
    writer.flush()?;
    Ok(())
}

fn main() -> Result<()> {
    let mut image: Option<PathBuf> = None;
    let mut weights: Option<PathBuf> = None;
    let mut export_prefix: Option<PathBuf> = None;
    let mut device: Option<&str> = None;
    let mut args = env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--image" => image = Some(PathBuf::from(args.next().context("--image requires path")?)),
            "--weights" => {
                weights = Some(PathBuf::from(
                    args.next().context("--weights requires path")?,
                ))
            }
            "--export-prefix" => {
                export_prefix = Some(PathBuf::from(
                    args.next().context("--export-prefix requires path")?,
                ))
            }
            "--cpu" | "--cuda" => {
                if device.is_some() {
                    bail!("choose exactly one device");
                }
                device = Some(if arg == "--cpu" { "cpu" } else { "cuda" });
            }
            "--help" | "-h" => {
                println!("{}", usage());
                return Ok(());
            }
            _ => bail!("unknown argument '{arg}'\n{}", usage()),
        }
    }
    match device.context("specify exactly one of --cpu or --cuda")? {
        "cpu" => (),
        "cuda" => bail!(
            "native CUDA Rust full-model inference is not implemented yet; \
             use --cpu for correctness, or run the separate cuda-kernels probe \
             on CUDA 13.3+. No silent CPU fallback."
        ),
        _ => unreachable!(),
    }

    let image_path = image.context(usage())?;
    let weights_path = weights.context(usage())?;
    let input = preprocess::open_and_preprocess(&image_path)?;
    println!("preprocessing profile: dinov2-base-native518-v1");
    println!("input shape: {:?}", input.shape);
    println!("execution: standalone Rust FP32 CPU reference");
    let model = DinoV2Base::load(&weights_path)?;
    println!("model: {}", model.model_id());
    let dense = model.forward_features(&input)?;
    println!(
        "CLS: [1,{}]; patches: [1,{},{}]; grid: [1,{},{},{}]",
        dense.hidden_dim,
        dense.grid_height * dense.grid_width,
        dense.hidden_dim,
        dense.hidden_dim,
        dense.grid_height,
        dense.grid_width
    );
    if let Some(prefix) = export_prefix {
        let root = prefix.to_string_lossy();
        write_f32_le(&PathBuf::from(format!("{root}.input.f32le")), &input.values)?;
        write_f32_le(
            &PathBuf::from(format!("{root}.embedding.f32le")),
            &dense.cls,
        )?;
        write_f32_le(
            &PathBuf::from(format!("{root}.hidden.f32le")),
            &dense.hidden_state_row_major(),
        )?;
        write_f32_le(
            &PathBuf::from(format!("{root}.spatial.f32le")),
            &dense.spatial_map,
        )?;
        println!("exported tensor fixtures for independent parity validation");
    }
    println!("CLS first 8 values: {:?}", &dense.cls[..8]);
    println!(
        "Research status: checkpoint reference parity and native CUDA GPU forward are pending."
    );
    Ok(())
}
