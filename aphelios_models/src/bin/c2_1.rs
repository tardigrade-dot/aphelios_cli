use candle_core::{DType, Device};
use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use candle_nn::{Optimizer, VarMap, SGD};

const MODEL_FILE: &str =
    "/Users/larry/coderesp/aphelios_cli/aphelios_models/output/rust_model2.safetensors";
// 定义一个简单的 MLP
// 结构：输入(3) -> 隐藏层(4) -> ReLU -> 输出层(2)
struct Mlp {
    ln1: Linear,
    ln2: Linear,
}

impl Mlp {
    fn new(vs: VarBuilder) -> Result<Self> {
        // 使用 .pp() 创建不同的命名空间，防止权重名冲突
        let ln1 = candle_nn::linear(3, 4, vs.pp("ln1"))?;
        let ln2 = candle_nn::linear(4, 2, vs.pp("ln2"))?;
        Ok(Self { ln1, ln2 })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // 第一层线性变换
        let xs = self.ln1.forward(xs)?;
        // 激活函数：ReLU (将所有负数变为 0)
        let xs = xs.relu()?;
        // 第二层线性变换
        self.ln2.forward(&xs)
    }
}

fn main() -> Result<()> {
    train();
    Ok(())
}
fn train() -> Result<()> {
    let device = Device::Cpu;
    let vm = VarMap::new();
    let vs = VarBuilder::from_varmap(&vm, DType::F32, &device);

    // 1. 初始化模型
    let model = Mlp::new(vs)?;

    // 2. 准备数据 (假设我们想拟合一个稍微复杂的逻辑)
    let x = Tensor::randn(0f32, 1.0, (16, 3), &device)?; // Batch 为 16
    let y = Tensor::randn(0f32, 1.0, (16, 2), &device)?;

    // 3. 优化器
    let mut opt = SGD::new(vm.all_vars(), 1e-2)?;

    // 4. 训练一步看看
    let logits = model.forward(&x)?;
    let loss = logits.sub(&y)?.sqr()?.mean_all()?;
    opt.backward_step(&loss)?;

    println!("Initial Loss: {:?}", loss.to_vec0::<f32>()?);
    vm.save(MODEL_FILE)?;
    println!("model saved {}", MODEL_FILE);

    println!("{}", render_dynamic_dot(&model));
    Ok(())
}

fn render_dynamic_dot(model: &Mlp) -> String {
    // 假设我们在 Linear 模块中能访问到 weight
    // 这里仅演示逻辑：通过获取 weights 的 shape 来自动标注连线
    let w1_shape = model.ln1.weight().dims(); // [out, in]
    let w2_shape = model.ln2.weight().dims();

    format!(
        "digraph G {{
            rankdir=LR;
            Input [label=\"Input (dim:{in_dim})\"];
            Hidden [label=\"Hidden (dim:{h_dim})\"];
            Output [label=\"Output (dim:{out_dim})\"];

            Input -> Hidden [label=\"Weight: {w1:?}\"];
            Hidden -> Output [label=\"Weight: {w2:?}\"];
        }}",
        in_dim = w1_shape[1],
        h_dim = w1_shape[0],
        out_dim = w2_shape[0],
        w1 = w1_shape,
        w2 = w2_shape
    )
}
