//! Reference preprocessing for the initial DINOv2 Base / 518-pixel experiment.
//!
//! This profile deliberately uses the checkpoint's native positional-embedding
//! resolution (518 x 518). It is NOT the Hugging Face default 224-pixel image
//! processor. This distinction must be recorded when comparing models.

use anyhow::{Result, bail};
use candle_core::{Device, Tensor};
use image::{DynamicImage, imageops};

pub const INPUT_SIZE: u32 = 518;
pub const RESIZE_SHORTEST: u32 = 592;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Convert RGB image to a batched normalized CHW FP32 CPU tensor.
///
/// Resize the shortest dimension to 592 using Catmull-Rom interpolation,
/// then center-crop 518 x 518. Caller explicitly moves the result to GPU.
/// Decoding and preprocessing are excluded from isolated inference timing.
pub fn preprocess(image: &DynamicImage) -> Result<Tensor> {
    let rgb = image.to_rgb8();
    let (width, height) = rgb.dimensions();
    if width == 0 || height == 0 {
        bail!("cannot preprocess empty image");
    }
    let scale = RESIZE_SHORTEST as f64 / width.min(height) as f64;
    let resized_width = ((width as f64 * scale).round() as u32).max(INPUT_SIZE);
    let resized_height = ((height as f64 * scale).round() as u32).max(INPUT_SIZE);
    let resized = imageops::resize(
        &rgb,
        resized_width,
        resized_height,
        imageops::FilterType::CatmullRom,
    );
    let left = (resized_width - INPUT_SIZE) / 2;
    let top = (resized_height - INPUT_SIZE) / 2;
    let cropped = imageops::crop_imm(&resized, left, top, INPUT_SIZE, INPUT_SIZE).to_image();

    // Channel-first order: all red pixels, then green, then blue.
    let plane = (INPUT_SIZE * INPUT_SIZE) as usize;
    let mut values = vec![0f32; 3 * plane];
    for (idx, pixel) in cropped.pixels().enumerate() {
        for channel in 0..3 {
            values[channel * plane + idx] =
                ((pixel[channel] as f32 / 255.0) - MEAN[channel]) / STD[channel];
        }
    }
    Ok(Tensor::from_vec(
        values,
        (1, 3, INPUT_SIZE as usize, INPUT_SIZE as usize),
        &Device::Cpu,
    )?)
}

pub fn open_and_preprocess(path: &std::path::Path) -> Result<Tensor> {
    let image = image::open(path)?;
    preprocess(&image)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;
    use image::{ImageBuffer, Rgb};

    #[test]
    fn known_rgb_values_and_chw_order() -> Result<()> {
        let color = Rgb([255, 0, 128]);
        let img = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(32, 32, color));
        let tensor = preprocess(&img)?;
        assert_eq!(tensor.dims(), &[1, 3, 518, 518]);
        assert_eq!(tensor.dtype(), DType::F32);
        let channels = tensor.squeeze(0)?.to_vec3::<f32>()?;
        let expected = [
            (1.0 - MEAN[0]) / STD[0],
            (0.0 - MEAN[1]) / STD[1],
            ((128.0 / 255.0) - MEAN[2]) / STD[2],
        ];
        for (channel, want) in expected.iter().enumerate() {
            for &(y, x) in &[(0, 0), (16, 16), (517, 517)] {
                assert!(
                    (channels[channel][y][x] - *want).abs() < 1e-5,
                    "channel {channel} [{y},{x}] differs"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn rectangular_source_has_fixed_output() -> Result<()> {
        let img = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(80, 35, Rgb([12, 24, 36])));
        assert_eq!(preprocess(&img)?.dims(), &[1, 3, 518, 518]);
        Ok(())
    }

    #[test]
    fn deterministic_fixture_repeats() -> Result<()> {
        // The plain-text fixture ships with the source tree: no external dataset.
        let bytes = include_bytes!("../tests/fixtures/pixel_fixture.ppm");
        let img = image::load_from_memory(bytes)?;
        let a = preprocess(&img)?.flatten_all()?.to_vec1::<f32>()?;
        let b = preprocess(&img)?.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(a, b);
        Ok(())
    }
}
