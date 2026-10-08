//! DINOv2 Base feature extraction with native 518 x 518 positional embeddings.
//!
//! Loads the original `facebook/dinov2-base` Hugging Face `model.safetensors`
//! (separate query/key/value projections and no classification head).
//! Do not confuse this with Candle's pretrained `vit_small` ImageNet example.
//!
//! This is the first model-specific baseline. Later adapters must keep the
//! test runner independent of this module's architecture and checkpoint.

use anyhow::{Context, Result, ensure};
use candle_core::{D, DType, Device, IndexOp, Tensor};
use candle_nn::{
    Conv2d, Conv2dConfig, LayerNorm, Linear, Module, VarBuilder, conv2d, layer_norm, linear,
};
use std::path::Path;

const HIDDEN: usize = 768;
const HEADS: usize = 12;
const HEAD_DIM: usize = HIDDEN / HEADS;
const PATCH: usize = 14;
const IMAGE: usize = 518;
const PATCHES: usize = (IMAGE / PATCH) * (IMAGE / PATCH);
const LAYERS: usize = 12;

struct Attention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
}

impl Attention {
    fn load(vb: VarBuilder) -> candle_core::Result<Self> {
        let attn = vb.pp("attention");
        Ok(Self {
            query: linear(HIDDEN, HIDDEN, attn.pp("query"))?,
            key: linear(HIDDEN, HIDDEN, attn.pp("key"))?,
            value: linear(HIDDEN, HIDDEN, attn.pp("value"))?,
            output: linear(HIDDEN, HIDDEN, vb.pp("output.dense"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let (batch, tokens, _) = xs.dims3()?;

        let q = self
            .query
            .forward(xs)?
            .reshape((batch, tokens, HEADS, HEAD_DIM))?
            .transpose(1, 2)?
            .contiguous()?;

        let k = self
            .key
            .forward(xs)?
            .reshape((batch, tokens, HEADS, HEAD_DIM))?
            .transpose(1, 2)?;

        let v = self
            .value
            .forward(xs)?
            .reshape((batch, tokens, HEADS, HEAD_DIM))?
            .transpose(1, 2)?
            .contiguous()?;

        let k_t = k.transpose(2, 3)?.contiguous()?;

        let scale = 1.0 / (HEAD_DIM as f64).sqrt();
        let scores = (q.matmul(&k_t)? * scale)?;

        let probs = candle_nn::ops::softmax(&scores, D::Minus1)?;

        let context = probs
            .contiguous()?
            .matmul(&v)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, tokens, HIDDEN))?;

        self.output.forward(&context)
    }
}

struct EncoderBlock {
    norm1: LayerNorm,
    attention: Attention,
    gamma1: Tensor,
    norm2: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    gamma2: Tensor,
}

impl EncoderBlock {
    fn load(vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            norm1: layer_norm(HIDDEN, 1e-6, vb.pp("norm1"))?,
            attention: Attention::load(vb.pp("attention"))?,
            gamma1: vb.get(HIDDEN, "layer_scale1.lambda1")?,
            norm2: layer_norm(HIDDEN, 1e-6, vb.pp("norm2"))?,
            fc1: linear(HIDDEN, HIDDEN * 4, vb.pp("mlp.fc1"))?,
            fc2: linear(HIDDEN * 4, HIDDEN, vb.pp("mlp.fc2"))?,
            gamma2: vb.get(HIDDEN, "layer_scale2.lambda1")?,
        })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let attn = self
            .attention
            .forward(&self.norm1.forward(xs)?)?
            .broadcast_mul(&self.gamma1)?;
        let xs = (xs + &attn)?;
        let hidden = self.fc1.forward(&self.norm2.forward(&xs)?)?.gelu()?;
        let mlp = self.fc2.forward(&hidden)?.broadcast_mul(&self.gamma2)?;
        xs + mlp
    }
}

pub struct DinoV2Base {
    patch_embed: Conv2d,
    cls_token: Tensor,
    pos_embed: Tensor,
    blocks: Vec<EncoderBlock>,
    norm: LayerNorm,
}

impl DinoV2Base {
    /// The safetensors file must be immutable while mmaped.
    pub fn load(weights: &Path, device: &Device) -> Result<Self> {
        ensure!(
            weights.is_file(),
            "weights not found: {}",
            weights.display()
        );
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, device) }
            .with_context(|| format!("opening original HF checkpoint {}", weights.display()))?;
        Self::from_var_builder(vb).context("loading DINOv2 Base: check checkpoint keys and shapes")
    }

    fn from_var_builder(vb: VarBuilder) -> Result<Self> {
        let embed = vb.pp("embeddings");
        let patch_embed = conv2d(
            3,
            HIDDEN,
            PATCH,
            Conv2dConfig {
                stride: PATCH,
                ..Default::default()
            },
            embed.pp("patch_embeddings.projection"),
        )?;
        let cls_token = embed.get((1, 1, HIDDEN), "cls_token")?;
        let pos_embed = embed.get((1, PATCHES + 1, HIDDEN), "position_embeddings")?;
        let mut blocks = Vec::with_capacity(LAYERS);
        for index in 0..LAYERS {
            blocks.push(EncoderBlock::load(vb.pp(format!("encoder.layer.{index}")))?);
        }
        let norm = layer_norm(HIDDEN, 1e-6, vb.pp("layernorm"))?;
        Ok(Self {
            patch_embed,
            cls_token,
            pos_embed,
            blocks,
            norm,
        })
    }

    /// Returns CLS embeddings with shape [batch, 768] (not tumor logits).
    /// The 518px baseline deliberately avoids positional interpolation.
    pub fn forward_features(&self, image: &Tensor) -> Result<Tensor> {
        let (batch, channels, height, width) = image.dims4()?;
        ensure!(
            channels == 3 && height == IMAGE && width == IMAGE,
            "DINOv2 Base native profile expects [N, 3, 518, 518]"
        );
        ensure!(
            image.dtype() == DType::F32,
            "DINOv2 Base expects FP32 input"
        );
        let patches = self
            .patch_embed
            .forward(image)?
            .reshape((batch, HIDDEN, PATCHES))?
            .transpose(1, 2)?;
        let class_tokens = self.cls_token.broadcast_as((batch, 1, HIDDEN))?;
        let mut hidden =
            Tensor::cat(&[&class_tokens, &patches], 1)?.broadcast_add(&self.pos_embed)?;
        for block in &self.blocks {
            hidden = block.forward(&hidden)?;
        }
        Ok(self.norm.forward(&hidden)?.i((.., 0))?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_match_dinov2_base_config() {
        assert_eq!(IMAGE / PATCH, 37);
        assert_eq!(PATCHES + 1, 1370);
        assert_eq!(HIDDEN / HEADS, 64);
        assert_eq!(LAYERS, 12);
    }
}
