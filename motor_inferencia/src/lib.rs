//! Model-agnostic spatial-feature contracts and CPU numerical reference.
//!
//! This crate is standalone. The experimental NVIDIA CUDA Rust
//! kernels live in a separate package to keep the CPU correctness gate portable.
pub mod dinov2;
pub mod preprocess;
pub mod spatial;

use anyhow::Result;
use preprocess::ImageTensor;
use spatial::DenseFeatures;

pub trait ModelAdapter {
    fn model_id(&self) -> &'static str;
    fn forward_features(&self, input: &ImageTensor) -> Result<DenseFeatures>;
}
