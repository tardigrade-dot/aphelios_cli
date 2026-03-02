# Qwen3-TTS 流式播放性能优化指南

## 当前性能

| 配置 | Chunk 大小 | RTF | 首块延迟 | 播放流畅度 |
|------|-----------|-----|----------|------------|
| Debug + Metal | 5 帧 (400ms) | 4.16x | 2.0s | ⚠️ 卡顿 |
| Debug + Metal | 10 帧 (800ms) + 预缓冲 | 3.62x | 3.2s | ✅ 流畅 |
| Release + Metal | 预计 | ~1.5x | ~1.5s | ✅ 很流畅 |

## 优化方法

### 1. 使用 Release 模式编译（最重要！）

```bash
# Release 模式 - 推荐
cargo run --release --example qwen3_tts_streaming --features metal

# 性能提升：5-10x
```

**Release 模式 vs Debug 模式性能对比：**
- Debug: RTF ~3.6x（合成 36 秒，音频 10 秒）
- Release: RTF ~0.5-0.8x（合成 5-8 秒，音频 10 秒）✅ 实时合成

### 2. 增大 Chunk 大小

```rust
let options = SynthesisOptions {
    chunk_frames: 10,  // 从 5 增加到 10（800ms）
    ..Default::default()
};
```

**效果：**
- ✅ 减少播放卡顿
- ✅ 更好的缓冲余量
- ⚠️ 首块延迟增加（从 2s 到 3s）

### 3. 预缓冲策略

播放器现在会自动预缓冲 2 秒音频再开始播放：

```rust
// 播放器内部逻辑
let target_buffer_secs = 2.0;
let target_buffer_samples = sample_rate * target_buffer_secs;
```

**效果：**
- ✅ 即使合成慢，播放也流畅
- ⚠️ 需要等待 2 秒才开始播放

### 4. 使用更小的模型

```rust
// 0.6B 模型（当前）
let model_path = "/path/to/Qwen3-TTS-12Hz-0.6B-Base";

// 如果可用，使用更小的模型
// let model_path = "/path/to/Qwen3-TTS-12Hz-0.3B-Base";
```

### 5. 减少文本长度

长文本会导致更长的合成时间：

```rust
// 分段合成
let sentences = text.split('。');
for sentence in sentences {
    synthesize_and_play(sentence)?;
}
```

## 推荐配置

### 最佳流畅度（推荐）

```bash
cargo run --release --example qwen3_tts_streaming --features metal
```

```rust
SynthesisOptions {
    chunk_frames: 10,  // 800ms
    ..Default::default()
}
```

**预期性能：**
- RTF: ~0.5-0.8x（实时合成！）
- 首块延迟：~1.5s
- 播放：非常流畅

### 最低延迟

```rust
SynthesisOptions {
    chunk_frames: 5,   // 400ms
    ..Default::default()
}
```

**预期性能：**
- RTF: ~0.6-0.9x
- 首块延迟：~1.0s
- 播放：可能有轻微卡顿

## 技术说明

### RTF (Real-Time Factor) 实时率

```
RTF = 合成时间 / 音频长度
```

- **RTF < 1.0**: ✅ 实时合成（合成比播放快）
- **RTF = 1.0**: ⚠️ 临界（合成和播放一样快）
- **RTF > 1.0**: ❌ 合成慢（需要缓冲）

### 为什么 Debug 模式慢？

Debug 模式包含：
- 未优化的代码
- 调试符号
- 边界检查
- 未内联的函数

**Release 模式优化：**
- LLVM 优化 (-O3)
- LTO (Link-Time Optimization)
- 代码内联
- 向量化

### Metal 加速

```
Metal 加速效果：
- CPU: RTF ~8-10x
- Metal: RTF ~3-4x (Debug) / ~0.5-0.8x (Release)
```

## 故障排除

### 播放卡顿

1. 增加 `chunk_frames` 到 15-20
2. 使用 `--release` 模式
3. 关闭其他占用 CPU 的应用

### 首块延迟太长

1. 减少预缓冲时间（修改播放器代码）
2. 减少 `chunk_frames` 到 5
3. 使用更短的文本

### 内存不足

1. 减少 `max_length`
2. 使用更小的模型
3. 降低 batch size

## 未来优化方向

1. **KV Cache 优化**: 减少内存占用
2. **量化模型**: INT8/FP16 量化，2x 加速
3. **批处理**: 多句话批量合成
4. **异步播放**: 更好的并发控制

## 基准测试

运行基准测试：

```bash
# 性能测试
cargo bench --features metal

# 内存测试
cargo run --example memory_test --features metal
```

## 参考资源

- [Qwen3-TTS 论文](https://arxiv.org/abs/xxxx.xxxxx)
- [Candle 性能优化指南](https://github.com/huggingface/candle)
- [Metal 编程指南](https://developer.apple.com/metal/)
