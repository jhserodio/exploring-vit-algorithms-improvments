//! Host-side deterministic RGB, CHW and geometric preprocessing.
//! Coordinates are relative to a processed tile, not an entire WSI.
use anyhow::{Result, bail, ensure};
use image::{DynamicImage, imageops};
use std::path::Path;

pub const INPUT_SIZE: u32 = 518;
pub const RESIZE_SHORTEST: u32 = 592;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

#[derive(Debug, Clone, PartialEq)]
pub struct CropGeometry {
    pub original_width: u32,
    pub original_height: u32,
    pub resized_width: u32,
    pub resized_height: u32,
    pub crop_left: u32,
    pub crop_top: u32,
    pub output_size: u32,
}

impl CropGeometry {
    /// Approximate source-image center corresponding to processed pixel center.
    pub fn processed_to_source(&self, x: f64, y: f64) -> (f64, f64) {
        let sx = ((x + self.crop_left as f64 + 0.5) * self.original_width as f64
            / self.resized_width as f64)
            - 0.5;
        let sy = ((y + self.crop_top as f64 + 0.5) * self.original_height as f64
            / self.resized_height as f64)
            - 0.5;
        (sx, sy)
    }

    pub fn source_to_processed(&self, x: f64, y: f64) -> (f64, f64) {
        let px = ((x + 0.5) * self.resized_width as f64 / self.original_width as f64)
            - 0.5
            - self.crop_left as f64;
        let py = ((y + 0.5) * self.resized_height as f64 / self.original_height as f64)
            - 0.5
            - self.crop_top as f64;
        (px, py)
    }

    /// Center of a spatial token in the decoded source image.
    pub fn patch_center(&self, row: usize, col: usize, patch: usize) -> Result<(f64, f64)> {
        ensure!(patch > 0, "zero-size patch");
        ensure!(
            (col + 1) * patch <= self.output_size as usize
                && (row + 1) * patch <= self.output_size as usize,
            "patch outside crop"
        );
        let x = col as f64 * patch as f64 + (patch as f64 - 1.0) / 2.0;
        let y = row as f64 * patch as f64 + (patch as f64 - 1.0) / 2.0;
        Ok(self.processed_to_source(x, y))
    }
}

#[derive(Debug, Clone)]
pub struct ImageTensor {
    pub shape: [usize; 4], // [1, 3, H, W]
    pub values: Vec<f32>,
    pub geometry: CropGeometry,
}

pub fn preprocess(image: &DynamicImage) -> Result<ImageTensor> {
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

    let plane = (INPUT_SIZE * INPUT_SIZE) as usize;
    let mut values = vec![0.0f32; plane * 3];
    for (idx, pixel) in cropped.pixels().enumerate() {
        for channel in 0..3 {
            values[channel * plane + idx] =
                (pixel[channel] as f32 / 255.0 - MEAN[channel]) / STD[channel];
        }
    }
    Ok(ImageTensor {
        shape: [1, 3, INPUT_SIZE as usize, INPUT_SIZE as usize],
        values,
        geometry: CropGeometry {
            original_width: width,
            original_height: height,
            resized_width,
            resized_height,
            crop_left: left,
            crop_top: top,
            output_size: INPUT_SIZE,
        },
    })
}

pub fn open_and_preprocess(path: &Path) -> Result<ImageTensor> {
    preprocess(&image::open(path)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};

    #[test]
    fn known_rgb_values_and_chw_order() -> Result<()> {
        let img = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(32, 32, Rgb([255, 0, 128])));
        let input = preprocess(&img)?;
        assert_eq!(input.shape, [1, 3, 518, 518]);
        let plane = 518 * 518;
        for (channel, expected) in [
            (1.0 - MEAN[0]) / STD[0],
            (0.0 - MEAN[1]) / STD[1],
            (128.0 / 255.0 - MEAN[2]) / STD[2],
        ]
        .iter()
        .enumerate()
        {
            for pos in [0, 16 * 518 + 16, 517 * 518 + 517] {
                assert!((input.values[channel * plane + pos] - expected).abs() < 1e-5);
            }
        }
        Ok(())
    }

    #[test]
    fn rectangular_source_has_fixed_output_and_invertible_geometry() -> Result<()> {
        let img = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(80, 35, Rgb([12, 24, 36])));
        let input = preprocess(&img)?;
        assert_eq!(input.shape, [1, 3, 518, 518]);
        let g = &input.geometry;
        let (sx, sy) = g.patch_center(5, 11, 14)?;
        let (px, py) = g.source_to_processed(sx, sy);
        assert!((px - (11. * 14. + 6.5)).abs() < 1e-8);
        assert!((py - (5. * 14. + 6.5)).abs() < 1e-8);
        Ok(())
    }

    #[test]
    fn synthetic_fixture_repeats() -> Result<()> {
        let bytes = include_bytes!("../tests/fixtures/pixel_fixture.ppm");
        let image = image::load_from_memory(bytes)?;
        assert_eq!(preprocess(&image)?.values, preprocess(&image)?.values);
        Ok(())
    }

    #[test]
    fn rejects_out_of_bounds_patch_centers() -> Result<()> {
        let img = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(16, 16, Rgb([0, 0, 0])));
        let input = preprocess(&img)?;
        assert!(input.geometry.patch_center(37, 0, 14).is_err());
        Ok(())
    }
}
