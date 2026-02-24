use candle_core::WgpuDeviceConfig;
use candle_core::{backend::BackendDevice, Device, Tensor};
use gloo::console::log;
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

// #[cfg(target_arch = "wasm32")]
// #[wasm_bindgen]
// pub async fn test_add(x: u16, y: u16) -> Result<u16, JsValue> {
//     let mut use_wgpu = false;

//     let device = {
//         #[cfg(feature = "wgpu")]
//         {
//             // use wgpu_compute_layer::{WgpuDevice, WgpuDeviceConfig};
//             use_wgpu = true;
//             Device::new_wgpu_async(0).await?
//         }

//         #[cfg(not(feature = "wgpu"))]
//         {
//             Device::Cpu
//         }
//     };

//     let a = Tensor::randn(0f32, 1., (2, 3), &device)?;
//     let b = Tensor::randn(0f32, 1., (3, 4), &device)?;

//     let c = a.matmul(&b)?;
//     let c = c.to_device_async(&Device::Cpu).await?;
//     //or c.to_vec2_async().await?

//     format!("use wgpu [{}]", use_wgpu);
//     Ok(x + y)
// }

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn add_method(x: u16, y: u16) -> u16 {
    use gloo::console::log;
    log!("hello ", x, y);
    x + y
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn run_vector_add() -> Result<String, JsError> {
    log!("start run run_vector_add");
    // 1. 初始化 wgpu 设备配置
    // 参考你提供的代码中的 DeviceConfig 逻辑
    let config = WgpuDeviceConfig {
        ..Default::default()
    };

    // 2. 创建 wgpu 设备 (异步)
    // 0 代表第一个 GPU 适配器
    let device = Device::new_wgpu_config_async(0, config).await?;

    log!("prepare data ...");
    // 3. 在 GPU 上创建两个向量 (Tensor)
    let data1 = [1.0f32, 2.0, 3.0, 4.0];
    let data2 = [10.0f32, 20.0, 30.0, 40.0];

    let v1 = Tensor::new(&data1, &device)?;
    let v2 = Tensor::new(&data2, &device)?;

    log!("prepare data finished");
    // 4. 执行向量操作 (在 GPU 上计算)
    let result = (v1 + v2)?;

    log!("正在计算结束...");
    // 5. 同步并获取结果回到 CPU
    // 在 wgpu 中，计算通常是延迟执行的，to_vec1 会触发同步
    let cpu_result = result.to_device_async(&Device::Cpu).await?;

    // 5. 现在已经在 CPU 上了，可以同步提取数据
    let values = cpu_result.to_vec1::<f32>()?;

    log!("结果: ", values); // 输出: [11.0, 22.0, 33.0, 44.0]

    Ok("finish".to_string())
}
