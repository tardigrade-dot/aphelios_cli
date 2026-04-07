use anyhow::{Context, Result};
use ndarray::{Array2, ArrayView2, Axis};
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// Voice styles for Kokoro TTS.
/// Each voice file (.bin) contains multiple style vectors of size 256.
pub struct KokoroVoices {
    /// Style vectors with shape (N, 256)
    data: Array2<f32>,
}

impl KokoroVoices {
    /// Load voices from a .bin file.
    /// Expects a file containing contiguous f32 (little-endian) values.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut file = File::open(path.as_ref())
            .with_context(|| format!("Failed to open voice file: {:?}", path.as_ref()))?;
        
        let metadata = file.metadata()?;
        let size = metadata.len() as usize;
        
        if size % 4 != 0 {
            anyhow::bail!("Voice file size is not a multiple of 4: {}", size);
        }
        
        let num_floats = size / 4;
        if num_floats % 256 != 0 {
            anyhow::bail!("Voice file float count is not a multiple of 256: {}", num_floats);
        }
        
        let mut buffer = vec![0u8; size];
        file.read_exact(&mut buffer)?;
        
        let data: Vec<f32> = buffer
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
            
        let num_vectors = num_floats / 256;
        let array = Array2::from_shape_vec((num_vectors, 256), data)
            .context("Failed to reshape voice data to (N, 256)")?;
            
        Ok(Self { data: array })
    }

    /// Get a style vector by index.
    /// Returns a view of shape (1, 256).
    pub fn get_style(&self, index: usize) -> Option<ArrayView2<'_, f32>> {
        if index < self.data.nrows() {
            Some(self.data.slice(ndarray::s![index..index+1, ..]))
        } else {
            None
        }
    }

    /// Get the number of style vectors available.
    pub fn num_styles(&self) -> usize {
        self.data.nrows()
    }
}
