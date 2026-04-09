use anyhow::Result;
use candle_core::{DType, Device, Tensor};

#[derive(Debug, Clone)]
pub struct LongCatRng {
    state: u64,
    cached_normal: Option<f32>,
}

impl LongCatRng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed,
            cached_normal: None,
        }
    }

    pub fn fork(&self, stream: u64) -> Self {
        Self::new(mix_seed(self.state ^ stream))
    }

    pub fn standard_normal_tensor(
        &mut self,
        shape: &[usize],
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        let elem_count = shape.iter().product::<usize>();
        let mut values = Vec::with_capacity(elem_count);
        for _ in 0..elem_count {
            values.push(self.next_standard_normal());
        }
        Ok(Tensor::from_vec(values, shape.to_vec(), device)?.to_dtype(dtype)?)
    }

    fn next_standard_normal(&mut self) -> f32 {
        if let Some(value) = self.cached_normal.take() {
            return value;
        }

        let u1 = self.next_unit_open_open().max(f32::MIN_POSITIVE);
        let u2 = self.next_unit_open_open();
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = std::f32::consts::TAU * u2;
        let z0 = radius * theta.cos();
        let z1 = radius * theta.sin();
        self.cached_normal = Some(z1);
        z0
    }

    fn next_unit_open_open(&mut self) -> f32 {
        let bits = self.next_u64() >> 40;
        ((bits as f32) + 0.5) / ((1u32 << 24) as f32)
    }

    fn next_u64(&mut self) -> u64 {
        self.state = mix_seed(self.state);
        self.state
    }
}

fn mix_seed(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeded_normal_stream_is_deterministic() {
        let mut lhs = LongCatRng::new(1024);
        let mut rhs = LongCatRng::new(1024);
        let lhs = lhs
            .standard_normal_tensor(&[2, 3], DType::F32, &Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let rhs = rhs
            .standard_normal_tensor(&[2, 3], DType::F32, &Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn forked_streams_are_distinct() {
        let mut lhs = LongCatRng::new(1024).fork(1);
        let mut rhs = LongCatRng::new(1024).fork(2);
        let lhs = lhs
            .standard_normal_tensor(&[4], DType::F32, &Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let rhs = rhs
            .standard_normal_tensor(&[4], DType::F32, &Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_ne!(lhs, rhs);
    }
}
