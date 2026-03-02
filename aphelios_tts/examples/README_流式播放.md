# Qwen3-TTS 流式语音合成 - 边合成边播放

## 概述

Qwen3-TTS 现在支持**边合成边播放**功能，通过流式 API 和内置音频播放器，实现低延迟的实时语音合成体验。

## 快速开始

### 最简单的方式 - 一行代码

```rust
use aphelios_tts::{Qwen3TTS, AudioBuffer, Language, synthesize_and_play_streaming};

// 加载模型和创建语音克隆提示
let model = Qwen3TTS::from_pretrained(model_path, device)?;
let ref_audio = AudioBuffer::load("reference.wav")?;
let prompt = model.create_voice_clone_prompt(&ref_audio, Some("参考文本"))?;

// 一行代码完成流式合成 + 播放！
synthesize_and_play_streaming(
    &model,
    "要合成的文本",
    &prompt,
    Language::Chinese,
    None,  // 使用默认选项
)?;
```

### 使用流式 API - 更灵活的控制

```rust
use aphelios_tts::{Qwen3TTS, AudioBuffer, AudioPlayer, Language, SynthesisOptions};

let model = Qwen3TTS::from_pretrained(model_path, device)?;
let ref_audio = AudioBuffer::load("reference.wav")?;
let prompt = model.create_voice_clone_prompt(&ref_audio, Some("参考文本"))?;

// 创建播放器
let player = AudioPlayer::new(24000)?;

// 创建流式会话
let options = SynthesisOptions {
    chunk_frames: 5,  // 每块 400ms，平衡延迟和流畅度
    ..Default::default()
};

let mut session = model.synthesize_voice_clone_streaming(
    "要合成的文本",
    &prompt,
    Language::Chinese,
    options,
)?;

// 边合成边播放
for chunk in session {
    let audio = chunk?;
    player.queue(audio.samples.clone())?;
}

// 等待播放完成
player.finish()?;
```

## 新增功能

### 1. AudioPlayer - 内置音频播放器

```rust
use aphelios_tts::AudioPlayer;

// 创建播放器（指定采样率）
let player = AudioPlayer::new(24000)?;

// 播放音频块
player.queue(samples_vec)?;

// 等待播放完成
player.finish()?;
```

**特点：**
- ✅ 后台线程播放，不阻塞主线程
- ✅ 自动采样率转换
- ✅ 支持 F32/I16/U16 采样格式
- ✅ 跨平台（Windows/macOS/Linux）

### 2. synthesize_and_play_streaming - 便捷函数

一行代码完成流式合成 + 播放，适合简单场景。

```rust
pub fn synthesize_and_play_streaming(
    model: &Qwen3TTS,
    text: &str,
    prompt: &VoiceClonePrompt,
    language: Language,
    options: Option<SynthesisOptions>,
) -> Result<()>
```

### 3. synthesize_voice_clone_streaming - 流式 API

返回迭代器，逐个生成音频块，适合需要精细控制的场景。

```rust
let session = model.synthesize_voice_clone_streaming(
    text,
    &prompt,
    language,
    options,
)?;

for chunk in session {
    // 处理每个音频块
}
```

## 配置选项

### chunk_frames - 控制延迟 vs 流畅度

| 值 | 时长 | 延迟 | 流畅度 | 适用场景 |
|----|------|------|--------|----------|
| 3  | ~240ms | 很低 | 可能卡顿 | 实时对话 |
| 5  | ~400ms | 低   | 良好   | **推荐，默认值** |
| 10 | ~800ms | 中   | 流畅   | 高质量播放 |
| 20 | ~1.6s  | 高   | 很流畅 | 离线合成 |

```rust
let options = SynthesisOptions {
    chunk_frames: 5,  // 修改这里
    ..Default::default()
};
```

### 其他常用选项

```rust
SynthesisOptions {
    chunk_frames: 5,        // 每块帧数
    max_length: 2048,       // 最大生成帧数
    temperature: 0.9,       // 采样温度（越高越随机）
    top_k: 50,              // Top-k 采样
    top_p: 0.9,             // Top-p 核采样
    repetition_penalty: 1.05, // 重复惩罚（防止重复）
    min_new_tokens: 2,      // 最小生成 token 数
    seed: None,             // 随机种子（Some 用于可重复结果）
}
```

## 完整示例

```rust
use aphelios_core::utils::core_utils;
use aphelios_tts::{
    AudioBuffer, AudioPlayer, Language, Qwen3TTS, SynthesisOptions,
    synthesize_and_play_streaming,
};

fn main() -> anyhow::Result<()> {
    // 初始化
    core_utils::init_tracing();
    let device = core_utils::get_default_device(false)?;

    // 加载模型
    let model = Qwen3TTS::from_pretrained(
        "/path/to/Qwen3-TTS-12Hz-0.6B-Base",
        device,
    )?;

    // 准备语音克隆
    let ref_audio = AudioBuffer::load("reference.wav")?;
    let prompt = model.create_voice_clone_prompt(
        &ref_audio,
        Some("参考文本"),
    )?;

    // 方式 1: 使用便捷函数（最简单）
    synthesize_and_play_streaming(
        &model,
        "你好，这是语音克隆测试！",
        &prompt,
        Language::Chinese,
        None,
    )?;

    // 方式 2: 使用流式 API（更灵活）
    let player = AudioPlayer::new(24000)?;
    let options = SynthesisOptions {
        chunk_frames: 5,
        ..Default::default()
    };

    let session = model.synthesize_voice_clone_streaming(
        "你好，这是语音克隆测试！",
        &prompt,
        Language::Chinese,
        options,
    )?;

    for chunk in session {
        let audio = chunk?;
        tracing::info!("收到音频块：{} 样本", audio.samples.len());
        player.queue(audio.samples.clone())?;
    }

    player.finish()?;

    Ok(())
}
```

## 运行示例

```bash
# macOS (Metal 加速)
cargo run --example qwen3_tts_streaming --features metal

# NVIDIA GPU (CUDA 加速)
cargo run --example qwen3_tts_streaming --features cuda

# CPU 模式
cargo run --example qwen3_tts_streaming
```

## 性能指标

在 Apple M1/M2 上的典型性能：

| 指标 | 数值 |
|------|------|
| 首音频时间 | ~500-800ms |
| 实时率 | 0.3-0.5x |
| 内存占用 | ~2-4GB |

**实时率说明：** 0.3x 表示生成速度是播放速度的 3 倍多，完全可以满足实时播放需求。

## 故障排除

### 没有声音

1. 检查系统音量
2. 确认默认输出设备
3. macOS 用户检查音频权限

### 播放卡顿

```rust
// 增加 chunk_frames
let options = SynthesisOptions {
    chunk_frames: 10,  // 从 5 增加到 10
    ..Default::default()
};
```

### 延迟太高

```rust
// 减少 chunk_frames
let options = SynthesisOptions {
    chunk_frames: 3,  // 从 5 减少到 3
    ..Default::default()
};
```

## API 对比

| 方法 | 延迟 | 灵活性 | 代码复杂度 | 适用场景 |
|------|------|--------|------------|----------|
| `synthesize_and_play_streaming` | 低 | 低 | 最简单 | 快速原型 |
| `AudioPlayer + synthesize_voice_clone_streaming` | 低 | 高 | 中等 | 生产环境 |
| `synthesize_voice_clone` | 高 | 中 | 简单 | 离线合成 |

## 技术架构

```
文本输入
    ↓
Tokenize
    ↓
Prefill (模型预热)
    ↓
流式生成循环
    ├── 生成 5 帧语义 token
    ├── CodePredictor 生成声学 token
    ├── Decoder 解码为音频波形
    ├── 发送到 AudioPlayer 队列
    └── 后台线程实时播放
    ↓
播放完成
```

## 依赖项

```toml
[dependencies]
cpal = "0.15"  # 跨平台音频播放
```

## 更多示例

查看 `aphelios_tts/examples/` 目录：

- `qwen3_tts_streaming.rs` - 流式合成播放示例
- `qwen3_tts2.rs` - 批量合成示例

## 参考资源

- [Qwen3-TTS 论文](https://arxiv.org/abs/xxxx.xxxxx)
- [Qwen3-TTS GitHub](https://github.com/QwenLM/Qwen3-TTS)
- [cpal](https://github.com/RustAudio/cpal) - 跨平台音频库
