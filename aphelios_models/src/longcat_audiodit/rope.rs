use candle_core::{D, Result, Tensor};

pub fn apply_qwen_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    // x: [1, heads, seq, head_dim]
    // cos, sin: [seq, head_dim/2]
    let (b, h, s, d) = x.dims4()?;
    let cos = cos.reshape((1, 1, s, d / 2))?.broadcast_as((b, h, s, d / 2))?;
    let sin = sin.reshape((1, 1, s, d / 2))?.broadcast_as((b, h, s, d / 2))?;
    
    let x1 = x.narrow(D::Minus1, 0, d / 2)?;
    let x2 = x.narrow(D::Minus1, d / 2, d / 2)?;
    
    let x1_rotated = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
    let x2_rotated = (x1.broadcast_mul(&sin)? + x2.broadcast_mul(&cos)?)?;
    
    Ok(Tensor::cat(&[&x1_rotated, &x2_rotated], D::Minus1)?)
}
