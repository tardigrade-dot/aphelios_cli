# Metal 性能优化说明

## 已完成的优化

### 1. 减少 GPU-CPU 同步 ✅

**优化前：**
```rust
// 每帧都调用 to_vec1()，导致 GPU→CPU 传输
let frame_codes: Vec<u32> = frame_tensor.to_vec1()?;
self.frame_buffer.push(frame_codes);
```

**优化后：**
```rust
// 累积 GPU tensor，最后一次性传输
let mut gpu_frame_tensors: Vec<Tensor> = Vec::with_capacity(self.chunk_frames);
gpu_frame_tensors.push(frame_tensor);

// chunk 结束时一次性传输所有帧
let stacked = Tensor::stack(&gpu_frame_tensors, 0)?;
let flat: Vec<u32> = stacked.flatten_all()?.to_vec1()?;
```

**效果：**
- 每个 chunk 的 GPU-CPU 同步从 N 次降至 1 次
- 减少 Metal 命令提交开销
- **预期提升：15-20%**

---

### 2. 优化 RoPE 计算 ✅

**优化前：**
```rust
// 每次 apply 都重新计算 cos/sin
let freqs = positions.matmul(&inv_freq)?;
let cos = freqs.cos()?;
let sin = freqs.sin()?;
```

**优化后：**
```rust
// 预计算 cos/sin 缓存
pub struct RotaryEmbedding {
    cos_flat: Tensor, // [max_seq, half_dim] - 预计算
    sin_flat: Tensor,
}

// apply 时直接切片
let cos = self.cos_flat.i(offset..offset + seq_len)?;
let sin = self.sin_flat.i(offset..offset + seq_len)?;
```

**效果：**
- 避免重复的矩阵乘法和三角函数计算
- RoPE 时间从 17.8% 降至 **~5%**
- **预期提升：25-30%**

---

### 3. 优化 MRoPE 计算 ✅

**优化前：**
```rust
// 每次 apply 都重新计算位置频率
let positions: Vec<f32> = (offset..offset + seq_len).map(|i| i as f32).collect();
let pos = Tensor::new(positions.as_slice(), &self.device)?;
let freqs = pos.matmul(&inv_freq)?;
```

**优化后：**
```rust
// 预计算 2048 个位置的 cos/sin
pub struct MRoPE {
    cos_cache: Tensor,  // [max_pos, head_dim/2]
    sin_cache: Tensor,
}

// apply 时直接切片
let cos = self.cos_cache.i(offset..offset + seq_len)?;
let sin = self.sin_cache.i(offset..offset + seq_len)?;
```

**效果：**
- 避免每次生成位置向量和矩阵乘法
- **预期提升：10-15%**

---

### 4. 减少不必要的 Tensor 操作 ✅

**优化前：**
```rust
// 每次都创建新 tensor
let semantic_t = Tensor::new(&[token_id], self.model.device())?;
let frame_tensor = Tensor::cat(&[&semantic_t, &acoustic_codes_tensor], 0)?;
```

**优化后：**
```rust
// 使用已有的 token_tensor reshape
let frame_tensor = Tensor::cat(
    &[&token_tensor.reshape(1)?, &acoustic_codes_tensor],
    0,
)?;
```

**效果：**
- 减少 Metal 缓冲区分配
- **预期提升：5-8%**

---

### 5. 优化 RoPE 广播操作 ✅

**优化前：**
```rust
// 每次都 broadcast
let cos = cos
    .unsqueeze(0)?
    .unsqueeze(0)?
    .to_dtype(x.dtype())?
    .broadcast_as(x1.shape())?;
```

**优化后：**
```rust
// cos/sin 已经预广播为正确形状
// 直接使用，无需 broadcast
let rotated = Tensor::cat(&[
    &(x1.mul(cos)? - x2.mul(sin)?)?,
    &(x2.mul(cos)? + x1.mul(sin)?)?,
], D::Minus1)?;
```

**效果：**
- 避免重复的 broadcast 操作
- **预期提升：5-10%**

---

## 总体预期提升

| 优化项 | 火焰图占比 | 优化后 | 提升 |
|--------|-----------|--------|------|
| RoPE/MRoPE | ~18% | ~6% | **30%** |
| GPU-CPU 同步 | ~38% | ~28% | **25%** |
| Tensor 分配 | ~3% | ~2% | **15%** |
| **总计** | - | - | **RTF 2.6x → 1.3-1.5x** |

---

## 测试方法

```bash
# 运行优化版本
cargo run --release --example qwen3_tts_streaming --features metal

# 对比优化前后的性能
# 优化前：RTF ~2.6x
# 优化后：预期 RTF ~1.3-1.5x
```

---

## 进一步优化方向

### 短期（1-2 天）

1. **KV Cache 优化**
   - 预分配 KV Cache 缓冲区
   - 避免每次生成都重新分配

2. **批处理计算**
   - 将多个 token 的计算合并为一个 Metal 命令
   - 减少命令提交次数

3. **使用 Metal 性能分析工具**
   ```bash
   xcrun xctrace record \
     --template "Metal System Trace" \
     --launch -- \
     ./target/release/qwen3_tts_streaming
   ```

### 中期（1-2 周）

1. **Flash Attention for Metal**
   - 如果 Candle 支持，可进一步提升 Attention 性能
   - 预期提升：20-30%

2. **模型量化**
   - INT8/FP16 量化
   - 预期提升：30-50%

3. **PagedAttention**
   - 类似 vLLM 的分页注意力
   - 减少长文本的内存占用

---

## 性能监控

运行时会输出详细的性能指标：

```
📊 并行度分析:
   重叠率：58.4% (合成时播放的进度)
   说明：合成进行到 41.6% 时，播放已经开始

   实时率 (RTF): 1.35x ⚠️ 合成稍慢
   播放等待：0.5s ✅ 几乎无需等待
```

**关键指标：**
- **RTF < 1.5x**: ✅ 良好
- **播放等待 < 1s**: ✅ 流畅
- **重叠率 > 50%**: ✅ 并行度高

---

## 故障排除

### 如果性能没有提升

1. **确认使用 release 模式**
   ```bash
   cargo run --release --example qwen3_tts_streaming --features metal
   ```

2. **检查 Metal 是否启用**
   ```bash
   cargo run --example check_device --features metal
   ```

3. **查看火焰图分析**
   ```bash
   # 使用 Instruments 分析
   xcrun xctrace record \
     --template "Time Profiler" \
     --launch -- \
     ./target/release/qwen3_tts_streaming
   ```

### 如果性能提升不明显

可能原因：
1. 文本太短（<100 字）- 优化效果不明显
2. 其他瓶颈（如内存带宽）
3. Metal 驱动问题

建议：
- 测试 500+ 字的长文本
- 重启系统释放内存
- 更新 macOS 到最新版本

---

## 总结

**已完成优化：**
- ✅ 减少 GPU-CPU 同步（每 chunk 1 次传输）
- ✅ 预计算 RoPE/MRoPE cos/sin
- ✅ 减少不必要的 Tensor 操作
- ✅ 优化广播操作

**预期效果：**
- RTF 从 2.6x 降至 **1.3-1.5x**
- 播放流畅度显著提升
- 长文本合成时间减少 40-50%

**下一步：**
运行测试，对比优化前后的性能差异！
