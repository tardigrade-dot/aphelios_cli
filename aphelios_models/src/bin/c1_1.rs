use aphelios_models::metal_device;
use candle_core::{backend::BackendDevice, DType, Device, MetalDevice, Result, Tensor};
use candle_nn::{Linear, Module, Optimizer, VarBuilder, VarMap, SGD};

const MODEL_FILE: &str =
    "/Users/larry/coderesp/aphelios_cli/aphelios_models/output/rust_model.safetensors";
// 定义与 Python 端对应的结构体
struct LinearModel {
    linear: Linear,
}

impl LinearModel {
    fn new(vs: VarBuilder) -> Result<Self> {
        // vs.pp("linear") 相当于进入 "linear" 命名空间
        // 这样它会自动寻找 "linear.weight" 和 "linear.bias"
        let linear = candle_nn::linear(10, 5, vs.pp("linear"))?;
        Ok(Self { linear })
    }
}

impl Module for LinearModel {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.linear.forward(xs)
    }
}

fn train() -> Result<()> {
    let device = metal_device()?;

    // 2. 初始化变量管理器 (VarMap 用于管理可训练参数)
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    // 3. 创建模型
    let model = LinearModel::new(vs)?;

    // 4. 准备模拟数据 (x: 100x3, y: 100x2)
    let x = Tensor::randn(0f32, 1.0, (100, 10), &device)?;
    // 假设真实规律是 y = x * w + b
    let target = x.narrow(1, 0, 5)?; // 简单取前两列作为目标

    // 5. 设置优化器 (SGD)
    let mut opt = SGD::new(varmap.all_vars(), 0.1)?;

    // 6. 训练循环
    for epoch in 1..=100 {
        let prediction = model.forward(&x)?;

        // 计算均方误差 (MSE Loss)
        let loss = prediction.sub(&target)?.sqr()?.mean_all()?;

        // 反向传播与更新
        opt.backward_step(&loss)?;

        if epoch % 20 == 0 {
            println!("Epoch: {}, Loss: {:?}", epoch, loss.to_vec0::<f32>()?);
        }
    }

    // 7. 保存训练好的权重
    varmap.save(MODEL_FILE)?;
    println!("模型已保存至 rust_model.safetensors");

    Ok(())
}

fn main() -> Result<()> {
    train();
    infer();
    Ok(())
}
fn infer() -> Result<()> {
    let device = Device::Metal(MetalDevice::new(0)?);

    // 1. 加载权重 (VarBuilder 会自动处理 safetensors 格式)
    let weights_path = MODEL_FILE;
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights_path], candle_core::DType::F32, &device)?
    };

    // 2. 实例化模型
    let model = LinearModel::new(vb)?;

    // 3. 推理
    let input = Tensor::new(&[[1.0f32, 2.0, 3.0]], &device)?;
    let output = model.forward(&input)?;

    println!("Input:  {:?}", input.to_vec2::<f32>()?);
    println!("Output: {:?}", output.to_vec2::<f32>()?);

    Ok(())
}
