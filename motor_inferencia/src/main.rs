mod dinov2;
mod preprocess;

use anyhow::{Context, Result, bail};
use candle_core::Device;
use std::{env, path::PathBuf};

fn usage() -> &'static str {
    "Usage: cargo run --release [--features cuda] -- --image IMAGE --weights MODEL.SAFETENSORS (--cpu | --cuda)\n\
     DINOv2 Base/518 native-resolution feature extractor. Outputs a [1,768] CLS embedding.\n\
     NOTE: This is not an OSCC classifier or a scientifically validated reference yet."
}

fn main() -> Result<()> {
    let mut image: Option<PathBuf> = None;
    let mut weights: Option<PathBuf> = None;
    let mut requested_device: Option<&str> = None;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--image" => {
                image = Some(PathBuf::from(
                    args.next().context("--image requires a path")?,
                ))
            }
            "--weights" => {
                weights = Some(PathBuf::from(
                    args.next().context("--weights requires a path")?,
                ))
            }
            "--cpu" => {
                if requested_device.is_some() {
                    bail!("choose exactly one device");
                }
                requested_device = Some("cpu");
            }
            "--cuda" => {
                if requested_device.is_some() {
                    bail!("choose exactly one device");
                }
                requested_device = Some("cuda");
            }
            "--help" | "-h" => {
                println!("{}", usage());
                return Ok(());
            }
            _ => bail!("unknown argument '{arg}'\n{}", usage()),
        }
    }
    let image = image.context(usage())?;
    let weights = weights.context(usage())?;
    let device = match requested_device.context("specify exactly one of --cpu or --cuda")? {
        "cpu" => Device::Cpu,
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                Device::new_cuda(0).context("CUDA initialization failed")?
            }
            #[cfg(not(feature = "cuda"))]
            {
                bail!("CUDA is disabled; rebuild using --features cuda")
            }
        }
        _ => unreachable!(),
    };

    let input_cpu = preprocess::open_and_preprocess(&image)?;
    println!("preprocessing profile: dinov2-base-native518-v1");
    println!(
        "input shape: {:?}, dtype: {:?}",
        input_cpu.dims(),
        input_cpu.dtype()
    );
    let input = input_cpu.to_device(&device)?;
    let model = dinov2::DinoV2Base::load(&weights, &device)?;
    let embedding = model.forward_features(&input)?;
    let cpu_output = embedding.to_device(&Device::Cpu)?.to_vec2::<f32>()?;
    if cpu_output.iter().flatten().any(|v| !v.is_finite()) {
        bail!("non-finite feature values; inference is invalid");
    }
    println!(
        "output: embedding, shape {:?}, finite: true",
        embedding.dims()
    );
    println!("CLS embedding first 8 values: {:?}", &cpu_output[0][..8]);
    println!(
        "Research status: numerical equivalence to the Hugging Face reference remains to be validated."
    );
    Ok(())
}
