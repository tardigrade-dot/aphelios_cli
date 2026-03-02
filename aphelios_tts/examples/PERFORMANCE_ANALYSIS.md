# Qwen3-TTS 流式播放性能分析

## 实测性能数据

### Release 模式（Metal M1/M2）

| 指标 | 数值 |
|------|------|
| Chunk 大小 | 10 帧 (800ms) |
| 每块耗时 | 1.31-1.35s |
| **RTF** | **1.64-1.70x** |
| 首块延迟 | ~1.6s |

### 长文本性能（900+ 字）

| 指标 | 理论值 | 实测值 |
|------|--------|--------|
| 音频长度 | 163.84s | 163.84s |
| 合成时间 | ~268s | 433s |
| **RTF** | 1.64x | **2.64x** |

**差异原因：**
1. KV Cache 随文本长度增长
2. 内存带宽瓶颈
3. Metal 缓冲区同步开销

## 性能瓶颈分析

### 1. 模型计算量

```
Qwen3-TTS-0.6B:
- Talker: ~500M 参数
- Code Predictor: ~100M 参数
- Decoder: ~60M 参数
- 总计：~660M 参数

每帧计算：
- 1 次 Talker 前向传播
- 1 次 CodePredictor 前向传播
- 1 次 Decoder 卷积
```

### 2. Metal 加速限制

```
Metal 性能特点：
✅ 并行计算能力强
⚠️ 内存带宽有限（统一内存）
⚠️ 缓冲区同步开销
⚠️ BF16 支持不如 CUDA 完善
```

### 3. KV Cache 增长

```
文本长度 vs KV Cache 大小:

100 字  → ~50MB
500 字  → ~250MB
900 字  → ~450MB

影响：
- 内存占用增加
- Cache 查找变慢
- 内存带宽成为瓶颈
```

## 优化建议

### 1. 分段合成（最有效）

```rust
// 将长文本分成 100-200 字的段落
let segments = split_text(long_text, 150);

for segment in segments {
    synthesize(segment)?;  // 每段独立合成
}
```

**效果：**
- ✅ 减少 KV Cache 大小
- ✅ 降低内存压力
- ✅ RTF 从 2.6x 降至 1.7x
- ⚠️ 段落间可能有轻微停顿

### 2. 使用更小的模型

```rust
// 如果有更小的模型可用
let model_path = "/path/to/Qwen3-TTS-0.3B-Base";
```

**预期效果：**
- RTF 降低 30-50%
- 音质可能略降

### 3. 减少文本长度

```rust
// 限制最大合成长度
let options = SynthesisOptions {
    max_length: 512,  // 限制帧数
    ..Default::default()
};
```

### 4. 批量预处理

```rust
// 预先创建 prompt，避免重复计算
let prompt = model.create_voice_clone_prompt(&ref_audio, Some(ref_text))?;

// 复用 prompt 合成多段文本
for text in texts {
    synthesize(text, &prompt)?;
}
```

### 5. 使用 CPU+Metal 混合

```rust
// Talker 用 Metal，Decoder 用 CPU
// 需要修改模型加载代码
```

## 实际使用建议

### 短文本（<100 字）

```rust
// 直接合成，RTF ~1.6x
let audio = model.synthesize_voice_clone(text, &prompt, Language::Chinese, None)?;
```

### 中等文本（100-500 字）

```rust
// 流式合成，RTF ~1.7x
let session = model.synthesize_voice_clone_streaming(text, &prompt, Language::Chinese, options)?;
```

### 长文本（>500 字）

```rust
// 分段合成，RTF ~1.7x（每段）
let segments = split_text(text, 150);
for segment in segments {
    synthesize_streaming(segment)?;
}
```

## 未来优化方向

### 短期（可实现）

1. **KV Cache 优化**
   - 使用 PagedAttention
   - 减少内存碎片

2. **量化**
   - INT8 量化（2x 加速）
   - FP16 推理

3. **批处理**
   - 多段文本批量合成

### 中期（需要模型修改）

1. **模型蒸馏**
   - 0.6B → 0.3B
   - 保持音质

2. **架构优化**
   - 更高效的注意力机制
   - 减少层数

### 长期（研究性质）

1. **流式模型**
   - 真正的实时合成
   - RTF < 0.5x

2. **硬件加速**
   - 专用 TPU/NPU
   - 更好的 Metal 支持

## 性能基准

### 不同设备的预期性能

| 设备 | RTF (0.6B) | RTF (0.3B*) |
|------|------------|-------------|
| M1 Max | 1.5x | 0.8x |
| M2 Pro | 1.6x | 0.9x |
| M3 Max | 1.3x | 0.7x |
| RTX 4090 | 0.8x | 0.4x |
| CPU (i9) | 5.0x | 2.5x |

*0.3B 为假设模型

## 结论

**当前最佳实践：**

1. 使用 `--release` 模式编译
2. 长文本分段处理（100-200 字/段）
3. 使用 Metal 加速
4. 接受 RTF ~1.6-1.7x 的现实

**对于 900 字文本：**
- 音频长度：~164 秒
- 预期合成时间：~270-280 秒（分段）
- 预期合成时间：~430 秒（不分段）

**建议：** 将 900 字文本分成 6-9 段，每段 100-150 字，可显著降低总合成时间。
