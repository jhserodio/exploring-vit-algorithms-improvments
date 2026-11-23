//! DINOv2 Base CPU correctness oracle, without a tensor framework.
//!
//! This intentionally prioritizes transparent numerical operations and spatial
//! token retention over CPU throughput. The dedicated NVIDIA CUDA Rust crate
//! implements GPU operations independently; full CUDA forward is a later gate.
//! Model: facebook/dinov2-base, original FP32 safetensors, 518px native grid.
use crate::{ModelAdapter, preprocess::ImageTensor, spatial::DenseFeatures};
use anyhow::{Context, Result, bail, ensure};
use ndarray::{Array2, Axis, s};
use safetensors::{Dtype, SafeTensors};
use std::{fs, path::Path};

pub const HIDDEN: usize = 768;
pub const HEADS: usize = 12;
pub const HEAD_DIM: usize = HIDDEN / HEADS;
pub const LAYERS: usize = 12;
pub const PATCH: usize = 14;
pub const IMAGE: usize = 518;
pub const GRID: usize = IMAGE / PATCH;
pub const PATCHES: usize = GRID * GRID;
pub const TOKENS: usize = PATCHES + 1;

fn tensor(st: &SafeTensors<'_>, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let t = st
        .tensor(name)
        .with_context(|| format!("missing checkpoint tensor {name}"))?;
    ensure!(
        t.shape() == shape,
        "tensor {name}: expected {shape:?}, got {:?}",
        t.shape()
    );
    ensure!(
        t.dtype() == Dtype::F32,
        "tensor {name}: FP32 required; got {:?}",
        t.dtype()
    );
    let data = t.data();
    ensure!(
        data.len() == shape.iter().product::<usize>() * 4,
        "bad byte length for {name}"
    );
    Ok(data
        .chunks(4)
        .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .collect())
}

#[derive(Debug)]
struct Linear {
    weight: Array2<f32>, // [out, in]
    bias: Vec<f32>,
}
impl Linear {
    fn load(st: &SafeTensors<'_>, prefix: &str, output: usize, input: usize) -> Result<Self> {
        let weight = tensor(st, &format!("{prefix}.weight"), &[output, input])?;
        let bias = tensor(st, &format!("{prefix}.bias"), &[output])?;
        Ok(Self {
            weight: Array2::from_shape_vec((output, input), weight)?,
            bias,
        })
    }

    fn forward(&self, input: &Array2<f32>) -> Result<Array2<f32>> {
        ensure!(
            input.ncols() == self.weight.ncols(),
            "linear input feature dimension mismatch"
        );
        let mut out = input.dot(&self.weight.t());
        for mut row in out.axis_iter_mut(Axis(0)) {
            for (value, bias) in row.iter_mut().zip(&self.bias) {
                *value += bias;
            }
        }
        Ok(out)
    }
}

#[derive(Debug)]
struct LayerNorm {
    weight: Vec<f32>,
    bias: Vec<f32>,
}
impl LayerNorm {
    fn load(st: &SafeTensors<'_>, prefix: &str, hidden: usize) -> Result<Self> {
        Ok(Self {
            weight: tensor(st, &format!("{prefix}.weight"), &[hidden])?,
            bias: tensor(st, &format!("{prefix}.bias"), &[hidden])?,
        })
    }

    fn forward(&self, input: &Array2<f32>) -> Result<Array2<f32>> {
        ensure!(
            input.ncols() == self.weight.len(),
            "LayerNorm shape mismatch"
        );
        let mut out = input.to_owned();
        for mut row in out.axis_iter_mut(Axis(0)) {
            let mean = row.iter().copied().sum::<f32>() / row.len() as f32;
            let variance = row.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / row.len() as f32;
            let inv_std = 1.0 / (variance + 1e-6).sqrt();
            for (c, value) in row.iter_mut().enumerate() {
                *value = (*value - mean) * inv_std * self.weight[c] + self.bias[c];
            }
        }
        Ok(out)
    }
}

#[derive(Debug)]
struct Attention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
}
impl Attention {
    fn load(st: &SafeTensors<'_>, prefix: &str) -> Result<Self> {
        let a = format!("{prefix}.attention.attention");
        Ok(Self {
            query: Linear::load(st, &format!("{a}.query"), HIDDEN, HIDDEN)?,
            key: Linear::load(st, &format!("{a}.key"), HIDDEN, HIDDEN)?,
            value: Linear::load(st, &format!("{a}.value"), HIDDEN, HIDDEN)?,
            output: Linear::load(
                st,
                &format!("{prefix}.attention.output.dense"),
                HIDDEN,
                HIDDEN,
            )?,
        })
    }

    fn forward(&self, input: &Array2<f32>) -> Result<Array2<f32>> {
        let q = self.query.forward(input)?;
        let k = self.key.forward(input)?;
        let v = self.value.forward(input)?;
        let mut concat = Array2::<f32>::zeros((input.nrows(), HIDDEN));
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        for head in 0..HEADS {
            let start = head * HEAD_DIM;
            let stop = start + HEAD_DIM;
            let qs = q.slice(s![.., start..stop]);
            let ks = k.slice(s![.., start..stop]);
            let vs = v.slice(s![.., start..stop]);
            let mut scores = qs.dot(&ks.t());
            scores.mapv_inplace(|x| x * scale);
            softmax_in_place(&mut scores);
            let context = scores.dot(&vs);
            concat.slice_mut(s![.., start..stop]).assign(&context);
        }
        self.output.forward(&concat)
    }
}

/// Stable scaled-dot-product softmax over the final dimension.
fn softmax_in_place(scores: &mut Array2<f32>) {
    for mut row in scores.axis_iter_mut(Axis(0)) {
        let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = row
            .iter_mut()
            .map(|v| {
                *v = (*v - maximum).exp();
                *v
            })
            .sum();
        for v in &mut row {
            *v /= sum;
        }
    }
}

#[derive(Debug)]
struct EncoderBlock {
    norm1: LayerNorm,
    attention: Attention,
    gamma1: Vec<f32>,
    norm2: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    gamma2: Vec<f32>,
}
fn add_scaled(base: &mut Array2<f32>, candidate: &Array2<f32>, gamma: &[f32]) -> Result<()> {
    ensure!(
        base.dim() == candidate.dim() && base.ncols() == gamma.len(),
        "residual shape mismatch"
    );
    for (mut left, right) in base
        .axis_iter_mut(Axis(0))
        .zip(candidate.axis_iter(Axis(0)))
    {
        for ((a, b), g) in left.iter_mut().zip(right.iter()).zip(gamma) {
            *a += b * g;
        }
    }
    Ok(())
}

impl EncoderBlock {
    fn load(st: &SafeTensors<'_>, index: usize) -> Result<Self> {
        let prefix = format!("encoder.layer.{index}");
        Ok(Self {
            norm1: LayerNorm::load(st, &format!("{prefix}.norm1"), HIDDEN)?,
            attention: Attention::load(st, &prefix)?,
            gamma1: tensor(st, &format!("{prefix}.layer_scale1.lambda1"), &[HIDDEN])?,
            norm2: LayerNorm::load(st, &format!("{prefix}.norm2"), HIDDEN)?,
            fc1: Linear::load(st, &format!("{prefix}.mlp.fc1"), HIDDEN * 4, HIDDEN)?,
            fc2: Linear::load(st, &format!("{prefix}.mlp.fc2"), HIDDEN, HIDDEN * 4)?,
            gamma2: tensor(st, &format!("{prefix}.layer_scale2.lambda1"), &[HIDDEN])?,
        })
    }

    fn forward(&self, mut hidden: Array2<f32>) -> Result<Array2<f32>> {
        let a = self.attention.forward(&self.norm1.forward(&hidden)?)?;
        add_scaled(&mut hidden, &a, &self.gamma1)?;
        let mut mlp = self.fc1.forward(&self.norm2.forward(&hidden)?)?;
        // Exact GELU as used in the reference model; no tanh approximation.
        mlp.mapv_inplace(|x| 0.5 * x * (1.0 + libm::erff(x / 2.0_f32.sqrt())));
        let mlp = self.fc2.forward(&mlp)?;
        add_scaled(&mut hidden, &mlp, &self.gamma2)?;
        Ok(hidden)
    }
}

pub struct DinoV2Base {
    patch_projection: Linear, // flattened conv [768, 3*14*14]
    cls_token: Vec<f32>,
    position_embeddings: Array2<f32>,
    blocks: Vec<EncoderBlock>,
    norm: LayerNorm,
}

impl DinoV2Base {
    pub fn load(weights: &Path) -> Result<Self> {
        ensure!(
            weights.is_file(),
            "weights not found: {}",
            weights.display()
        );
        let data = fs::read(weights).with_context(|| format!("reading {}", weights.display()))?;
        let st = SafeTensors::deserialize(&data).context("invalid safetensors checkpoint")?;
        let projection_name = "embeddings.patch_embeddings.projection";
        let patch_weights = tensor(
            &st,
            &format!("{projection_name}.weight"),
            &[HIDDEN, 3, PATCH, PATCH],
        )?;
        let patch_bias = tensor(&st, &format!("{projection_name}.bias"), &[HIDDEN])?;
        let cls_token = tensor(&st, "embeddings.cls_token", &[1, 1, HIDDEN])?;
        let position_embeddings = Array2::from_shape_vec(
            (TOKENS, HIDDEN),
            tensor(&st, "embeddings.position_embeddings", &[1, TOKENS, HIDDEN])?,
        )?;
        let mut blocks = Vec::with_capacity(LAYERS);
        for index in 0..LAYERS {
            blocks.push(EncoderBlock::load(&st, index)?);
        }
        Ok(Self {
            patch_projection: Linear {
                weight: Array2::from_shape_vec((HIDDEN, 3 * PATCH * PATCH), patch_weights)?,
                bias: patch_bias,
            },
            cls_token,
            position_embeddings,
            blocks,
            norm: LayerNorm::load(&st, "layernorm", HIDDEN)?,
        })
    }

    /// Expand image CHW into rows of channel-major [3,14,14] patches.
    fn patchify(input: &ImageTensor) -> Result<Array2<f32>> {
        ensure!(
            input.shape == [1, 3, IMAGE, IMAGE],
            "DINOv2 Base expects [1,3,518,518]"
        );
        ensure!(
            input.values.len() == 3 * IMAGE * IMAGE,
            "malformed image tensor"
        );
        let mut patches = Array2::<f32>::zeros((PATCHES, 3 * PATCH * PATCH));
        for row in 0..GRID {
            for col in 0..GRID {
                let p = row * GRID + col;
                for channel in 0..3 {
                    for dy in 0..PATCH {
                        for dx in 0..PATCH {
                            let src = channel * IMAGE * IMAGE
                                + (row * PATCH + dy) * IMAGE
                                + col * PATCH
                                + dx;
                            let dst = channel * PATCH * PATCH + dy * PATCH + dx;
                            patches[(p, dst)] = input.values[src];
                        }
                    }
                }
            }
        }
        Ok(patches)
    }

    pub fn forward_dense(&self, input: &ImageTensor) -> Result<DenseFeatures> {
        let patches = Self::patchify(input)?;
        let projected = self.patch_projection.forward(&patches)?;
        let mut hidden = Array2::<f32>::zeros((TOKENS, HIDDEN));
        for c in 0..HIDDEN {
            hidden[(0, c)] = self.cls_token[c];
        }
        hidden.slice_mut(s![1.., ..]).assign(&projected);
        hidden += &self.position_embeddings;
        for block in &self.blocks {
            hidden = block.forward(hidden)?;
        }
        let hidden = self.norm.forward(&hidden)?;
        if !hidden.iter().all(|v| v.is_finite()) {
            bail!("DINOv2 forward emitted non-finite features");
        }
        DenseFeatures::from_hidden(&hidden, GRID, GRID)
    }
}

impl ModelAdapter for DinoV2Base {
    fn model_id(&self) -> &'static str {
        "facebook/dinov2-base"
    }
    fn forward_features(&self, input: &ImageTensor) -> Result<DenseFeatures> {
        self.forward_dense(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn architecture_native_grid_has_1369_spatial_cells() {
        assert_eq!(GRID, 37);
        assert_eq!(PATCHES, 1369);
        assert_eq!(TOKENS, 1370);
        assert_eq!(HEAD_DIM, 64);
    }

    #[test]
    fn softmax_is_stable_for_large_logits() {
        let mut x = array![[10000., 10001.], [-10000., -9999.]];
        softmax_in_place(&mut x);
        for row in x.axis_iter(Axis(0)) {
            assert!((row.sum() - 1.0).abs() < 1e-5);
            assert!(row.iter().all(|v| v.is_finite()));
        }
        assert!((x[(0, 0)] - x[(1, 0)]).abs() < 1e-6);
    }

    #[test]
    fn failed_checkpoint_load_reports_missing_file() {
        assert!(DinoV2Base::load(Path::new("no-such-dinov2-weights.safetensors")).is_err());
    }

    #[test]
    fn residual_scale_matches_reference_formula() -> Result<()> {
        let mut base = array![[1., 2.], [3., 4.]];
        add_scaled(&mut base, &array![[10., 20.], [30., 40.]], &[0.1, 0.5])?;
        assert_eq!(base, array![[2., 12.], [6., 24.]]);
        Ok(())
    }
}
