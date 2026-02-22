use candle_core::{backend::BackendDevice, Device, Tensor, WgpuDevice};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn test_add() -> String {
    // Web 环境下默认使用 CPU
    let wgpu = WgpuDevice::new(0).unwrap();
    let device = &Device::Wgpu(wgpu);
    let a = Tensor::new(&[1f32, 2.0, 3.0], device).unwrap();
    let b = Tensor::new(&[4f32, 5.0, 6.0], device).unwrap();
    let c = (a + b).unwrap();

    format!("Candle 运行成功! 结果: {}", c.to_string())
}
