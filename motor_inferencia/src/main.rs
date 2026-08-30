use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use image::{ImageBuffer, Rgb, io::Reader as ImageReader};

fn vit_image_load(path: &str, device: &Device) -> Result<Tensor> {
    println!("Image load: {}", path);

    let img = ImageReader::open(path)?.decode()?;

    let img_resized = img.resize_exact(224, 224, image::imageops::FilterType::Triangle);
    let img_rgb = img_resized.to_rgb8();

    let (width, height) = img_rgb.dimensions();
    let channels = 3;

    let img_data: Vec<f32> = img_rgb
        .into_raw()
        .into_iter()
        .map(|p| p as f32 / 255.0)
        .collect();

    let tensor = Tensor::from_vec(img_data, (height as usize, width as usize, chanells));

    let tensor_chw = tensor.permute((2, 0, 1))?;

    let tensor_final = tensor_chw
        .unsqueeze(0)?
        .to_dtype(DType::F32)?
        .to_device(device)?;
}

fn main() -> Result<()> {
    let device = Device::new_cuda(0)?;
    println!("Activate GPU to vision process");

    let img_path("teste.jpg");

    let img_false: ImgBuffer<Rgb<u8>, Vec<u8>> = ImgBuffer::new(500, 500);
    img_false.save(img_path)?;

    let tensor_entry = vit_image_load(img_path, &device)?;

    println!("Success to Generate Img Tensor");
    println!("Expected Shape: [1, 3, 224, 224]");

    std::fs::remove_file(img_path)?;

    Ok(())
}
