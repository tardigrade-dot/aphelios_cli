use candle_core::{safetensors, Device, Result, Tensor};

#[test]
fn main_test() -> Result<()> {
    // let device = Device::Cpu;
    let device = Device::new_metal(0)?;

    // 1. 使用 safetensors 库加载整个文件
    // 注意：需要添加依赖 safetensors = "0.4" 在 Cargo.toml
    let tensors = candle_core::safetensors::load(
        "/Users/larry/coderesp/aphelios_cli/test_models/linear_model.safetensors",
        &device,
    )?;

    // 2. 从 Map 中获取对应的权重
    // 注意：Key 必须与 Python 中 weights 字典的 Key 完全一致
    let w = tensors
        .get("linear.weight")
        .ok_or_else(|| candle_core::Error::Msg("Weight not found".to_string()))?;
    let b = tensors
        .get("linear.bias")
        .ok_or_else(|| candle_core::Error::Msg("Bias not found".to_string()))?;

    // 3. 准备输入 (1, 3)
    let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (1, 3), &device)?;

    // 4. 前向计算: y = x * W^T + b
    // PyTorch 的 Linear 层权重存储为 (out_features, in_features)，所以需要转置
    let y = x.matmul(&w.t()?)?.broadcast_add(b)?;

    println!("Output: {:?}", y);
    Ok(())
}
