use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{conv2d, linear, ops, Conv2d, Conv2dConfig, Linear, Module, VarBuilder};
use image::GenericImageView;

struct MnistCNN {
    conv1: Conv2d,
    conv2: Conv2d,
    fc: Linear,
}

impl MnistCNN {
    fn new(vs: VarBuilder) -> Result<Self> {
        // Conv2d 参数: 输入通道, 输出通道, 卷积核大小, 配置(padding等)
        let config = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv1 = conv2d(1, 16, 3, config, vs.pp("conv1"))?;
        let conv2 = conv2d(16, 32, 3, config, vs.pp("conv2"))?;

        // 注意：这里的输入维度 1568 是 32 * 7 * 7 算出来的
        let fc = linear(1568, 10, vs.pp("fc"))?;

        Ok(Self { conv1, conv2, fc })
    }
}

impl Module for MnistCNN {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // 第一层：Conv -> ReLU -> MaxPool
        let xs = self.conv1.forward(xs)?;
        let xs = xs.relu()?;
        let xs = xs.max_pool2d(2)?; // 28x28 -> 14x14

        // 第二层：Conv -> ReLU -> MaxPool
        let xs = self.conv2.forward(&xs)?;
        let xs = xs.relu()?;
        let xs = xs.max_pool2d(2)?; // 14x14 -> 7x7

        // 展平：将 [Batch, 32, 7, 7] 变为 [Batch, 1568]
        let xs = xs.flatten_from(1)?;

        // 全连接层
        self.fc.forward(&xs)
    }
}

fn load_image_as_tensor(path: &str, device: &Device) -> candle_core::Result<Tensor> {
    let img = image::open(path).expect("Failed to open image").grayscale();
    let img = img.resize_exact(28, 28, image::imageops::FilterType::Lanczos3);

    let mut pixels = Vec::new();
    for y in 0..28 {
        for x in 0..28 {
            let p = img.get_pixel(x, y)[0] as f32 / 255.0; // 缩放到 0-1
            pixels.push((p - 0.1307) / 0.3081);
        }
    }

    // 形状必须是 [1, 1, 28, 28] -> [Batch, Channel, Height, Width]
    Tensor::from_vec(pixels, (1, 1, 28, 28), device)
}

fn main() -> Result<()> {
    let device = Device::Cpu;

    // 1. 加载权重
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &["/Users/larry/coderesp/aphelios_cli/aphelios_models/output/mnist_cnn.safetensors"],
            DType::F32,
            &device,
        )?
    };
    let model = MnistCNN::new(vb)?;

    // 2. 准备输入数据 (模拟一张 28x28 的全黑图片，中间有一个白点)
    // 形状必须是 [Batch, Channel, Height, Width] -> [1, 1, 28, 28]
    // let input = Tensor::zeros((1, 1, 28, 28), DType::F32, &device)?;
    let input = load_image_as_tensor(
        "/Users/larry/coderesp/aphelios_cli/test_data/mnist_digit_7.png",
        &device,
    )?;

    // 3. 执行推理
    let logits = model.forward(&input)?;

    // 4. 获取结果 (找概率最大的索引)
    let probabilities = ops::softmax(&logits, 1)?;
    let predicted_label = probabilities.argmax(1)?.to_vec1::<u32>()?[0];

    println!("Predicted digit: {}", predicted_label);

    Ok(())
}
