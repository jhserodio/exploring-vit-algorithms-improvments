//! Logical spatial representation, independently of tensor runtime and model.
//! A map is a feature representation, never a tumor mask or diagnosis.
use anyhow::{Result, ensure};
use ndarray::Array2;

#[derive(Debug, Clone)]
pub struct DenseFeatures {
    /// Shape [hidden_dim]; global CLS descriptor.
    pub cls: Vec<f32>,
    /// Row-major [grid_height * grid_width, hidden_dim].
    pub patch_tokens: Vec<f32>,
    /// CHW feature layout [hidden_dim, grid_height, grid_width].
    pub spatial_map: Vec<f32>,
    pub grid_height: usize,
    pub grid_width: usize,
    pub hidden_dim: usize,
}

impl DenseFeatures {
    /// Input tokens must be ordered [CLS, row-major patches], with no registers.
    pub fn from_hidden(
        hidden: &Array2<f32>,
        grid_height: usize,
        grid_width: usize,
    ) -> Result<Self> {
        ensure!(grid_height > 0 && grid_width > 0, "empty spatial grid");
        let patches = grid_height
            .checked_mul(grid_width)
            .ok_or_else(|| anyhow::anyhow!("spatial grid overflow"))?;
        ensure!(
            hidden.nrows() == patches + 1,
            "expected CLS plus {patches} patches, got {}",
            hidden.nrows()
        );
        ensure!(hidden.ncols() > 0, "empty feature dimension");
        let dim = hidden.ncols();
        let cls = hidden.row(0).to_vec();
        let patch_tokens: Vec<f32> = hidden
            .rows()
            .into_iter()
            .skip(1)
            .flat_map(|row| row.to_vec())
            .collect();
        let mut spatial_map = vec![0.0; patches * dim];
        for patch in 0..patches {
            for channel in 0..dim {
                spatial_map[channel * patches + patch] = patch_tokens[patch * dim + channel];
            }
        }
        Ok(Self {
            cls,
            patch_tokens,
            spatial_map,
            grid_height,
            grid_width,
            hidden_dim: dim,
        })
    }

    pub fn hidden_state_row_major(&self) -> Vec<f32> {
        let mut hidden = Vec::with_capacity(self.cls.len() + self.patch_tokens.len());
        hidden.extend_from_slice(&self.cls);
        hidden.extend_from_slice(&self.patch_tokens);
        hidden
    }

    pub fn cell(&self, row: usize, col: usize, channel: usize) -> Result<f32> {
        ensure!(
            row < self.grid_height && col < self.grid_width && channel < self.hidden_dim,
            "feature index out of bounds"
        );
        Ok(self.spatial_map
            [channel * self.grid_height * self.grid_width + row * self.grid_width + col])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn sp_02_spatial_reorders_channels_without_losing_cls() -> Result<()> {
        // CLS followed by 2x2 patch tokens, 3 channels each.
        let x = array![
            [100., 101., 102.],
            [1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 9.],
            [10., 11., 12.]
        ];
        let dense = DenseFeatures::from_hidden(&x, 2, 2)?;
        assert_eq!(dense.cls, vec![100., 101., 102.]);
        assert_eq!(
            dense.patch_tokens,
            vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]
        );
        assert_eq!(
            dense.spatial_map,
            vec![1., 4., 7., 10., 2., 5., 8., 11., 3., 6., 9., 12.]
        );
        assert_eq!(dense.cell(1, 0, 2)?, 9.0);
        assert_eq!(
            dense.hidden_state_row_major(),
            x.iter().copied().collect::<Vec<_>>()
        );
        Ok(())
    }

    #[test]
    fn sp_02_rejects_auxiliary_tokens_and_bad_shapes() {
        assert!(DenseFeatures::from_hidden(&Array2::zeros((6, 3)), 2, 2).is_err());
        assert!(DenseFeatures::from_hidden(&Array2::zeros((5, 3)), 0, 2).is_err());
    }
}
