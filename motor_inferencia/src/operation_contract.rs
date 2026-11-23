//! Model-agnostic, runtime-independent operation contracts.
//!
//! The same semantic checks are used by CPU reference tests and NVIDIA Tile
//! kernel probes. They explicitly reject host/device mixes and shape errors;
//! a GPU operation cannot silently execute on CPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Cuda,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    RowMajor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kernel {
    ResidualAdd,
    Gemm,
    SoftmaxLast,
    LayerNorm,
    Gelu,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSpec {
    pub dims: Vec<usize>,
    pub dtype: DType,
    pub layout: Layout,
    pub device: Device,
}

impl TensorSpec {
    pub fn f32(dims: &[usize], device: Device) -> Self {
        Self {
            dims: dims.to_vec(),
            dtype: DType::F32,
            layout: Layout::RowMajor,
            device,
        }
    }

    pub fn elements(&self) -> Result<usize, String> {
        if self.dims.is_empty() || self.dims.contains(&0) {
            return Err("empty or zero-dimension tensor".to_string());
        }
        self.dims.iter().try_fold(1usize, |count, value| {
            count.checked_mul(*value).ok_or("tensor size overflow".to_string())
        })
    }
}

#[derive(Debug, Clone)]
pub struct KernelSpec {
    pub kernel: Kernel,
    pub inputs: Vec<TensorSpec>,
    pub output: TensorSpec,
}

impl KernelSpec {
    /// Enforce shape/dtype/device semantics *before* allocating GPU memory.
    /// A CUDA kernel is never allowed to convert to CPU silently.
    pub fn validate(&self) -> Result<(), String> {
        self.output.elements()?;
        for input in &self.inputs {
            input.elements()?;
            if input.device != self.output.device {
                return Err("device mismatch: implicit CPU/GPU fallback forbidden".into());
            }
            if input.dtype != self.output.dtype {
                return Err("dtype mismatch: implicit conversion forbidden".into());
            }
            if input.layout != self.output.layout {
                return Err("layout mismatch: explicit transform required".into());
            }
        }
        let input = |idx: usize| {
            self.inputs.get(idx).map(|value| value.dims.as_slice())
                .ok_or("missing kernel input".to_string())
        };
        match self.kernel {
            Kernel::Gelu | Kernel::SoftmaxLast => {
                if self.inputs.len() != 1 || input(0)? != self.output.dims {
                    return Err("unary operator expects identical input/output shapes".into());
                }
                if self.kernel == Kernel::SoftmaxLast && self.output.dims.len() != 2 {
                    return Err("softmax requires [rows, entire_row_width]".into());
                }
            }
            Kernel::ResidualAdd => {
                if self.inputs.len() != 2
                    || input(0)? != self.output.dims
                    || input(1)? != self.output.dims
                {
                    return Err("residual add requires identically shaped tensors".into());
                }
            }
            Kernel::Gemm => {
                if self.inputs.len() != 2 {
                    return Err("GEMM expects 2 input tensors".into());
                }
                let (a, b) = (input(0)?, input(1)?);
                if a.len() != 2 || b.len() != 2
                    || self.output.dims.len() != 2
                    || a[1] != b[0]
                    || self.output.dims != [a[0], b[1]]
                {
                    return Err("GEMM expects A[M,K] × B[K,N] → C[M,N]".into());
                }
            }
            Kernel::LayerNorm => {
                if self.inputs.len() != 3 {
                    return Err("LayerNorm expects input, gamma and beta".into());
                }
                let a = input(0)?;
                if a.len() != 2
                    || a != self.output.dims
                    || input(1)? != [a[1]]
                    || input(2)? != [a[1]]
                {
                    return Err("LayerNorm expects X[M,N], gamma[N], beta[N] → Y[M,N]".into());
                }
            }
        }
        Ok(())
    }

    /// Validate the requested device explicitly; never imply a fallback.
    pub fn require_device(&self, expected: Device) -> Result<(), String> {
        self.validate()?;
        if self.output.device != expected {
            return Err(format!(
                "requested {expected:?} backend but tensor is on {:?}",
                self.output.device
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(kind: Kernel, input_shapes: &[&[usize]], output: &[usize]) -> KernelSpec {
        KernelSpec {
            kernel: kind,
            inputs: input_shapes.iter().map(|s| TensorSpec::f32(s, Device::Cpu)).collect(),
            output: TensorSpec::f32(output, Device::Cpu),
        }
    }

    #[test]
    fn gemm_contract_accepts_transformer_shapes_and_rejects_mismatched_k() {
        assert!(spec(Kernel::Gemm, &[&[1370, 768], &[768, 3072]], &[1370, 3072]).validate().is_ok());
        assert!(spec(Kernel::Gemm, &[&[1370, 768], &[767, 3072]], &[1370, 3072]).validate().is_err());
    }

    #[test]
    fn residual_softmax_layernorm_contracts_cover_spatial_token_dimensions() {
        assert!(spec(Kernel::ResidualAdd, &[&[1370, 768], &[1370, 768]], &[1370, 768]).validate().is_ok());
        assert!(spec(Kernel::SoftmaxLast, &[&[1370, 1370]], &[1370, 1370]).validate().is_ok());
        assert!(spec(Kernel::LayerNorm, &[&[1370, 768], &[768], &[768]], &[1370, 768]).validate().is_ok());
        assert!(spec(Kernel::SoftmaxLast, &[&[1370, 1370]], &[1370, 769]).validate().is_err());
    }

    #[test]
    fn wrong_device_dtype_and_size_are_explicit_failures() {
        let mut request = spec(Kernel::ResidualAdd, &[&[8], &[8]], &[8]);
        request.inputs[0].device = Device::Cuda;
        assert!(request.validate().unwrap_err().contains("device mismatch"));
        request.inputs[0].device = Device::Cpu;
        request.inputs[0].dtype = DType::F16;
        assert!(request.validate().unwrap_err().contains("dtype mismatch"));
        request.inputs[0].dtype = DType::F32;
        assert!(request.require_device(Device::Cuda).is_err());
        assert!(TensorSpec::f32(&[usize::MAX, 2], Device::Cpu).elements().is_err());
        assert!(TensorSpec::f32(&[0, 768], Device::Cpu).elements().is_err());
    }
}
