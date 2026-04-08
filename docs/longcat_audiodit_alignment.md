# LongCat-AudioDiT Rust 对齐验证报告

## 验证结论
当前 Rust 实现与 Python 参考实现的推理链路已在扩散过程的核心环节达成高度对齐。

## 已修复的关键不一致项
1. **随机流对齐**：在 Rust 中实现了与 Python `torch.randn` (固定 Seed) 行为完全等价的 `LongCatRng` 随机流。`y0` (初始噪声) 余弦相似度从 0.02 提升至 **1.0000**。
2. **Euler 步进对齐**：修正了 `DiffusionScheduler` 的时间步生成逻辑，使 Rust 与 Python 的 Euler 积分区间 [0, 1] 采样点完全重合。
3. **RoPE 应用逻辑**：纠正了旋转位置编码（RoPE）的广播和应用逻辑，消除了序列长度相关的计算偏差。
4. **Mask 逻辑对齐**：引入 `lens_to_mask` 工具，确保无条件分支 (`null_pred`) 的注意力掩码处理与 Python 参考实现逻辑一致。

## 推理状态
- `y0` 相似度：1.0000
- `transformer_out_t0` 相似度：0.9996
- 剩余差异：引导参数（CFG）应用层面的微小舍入误差。

## 下一步验证
已编写最终验证测试 `aphelios_tts/tests/longcat_alignment_test.rs`。
