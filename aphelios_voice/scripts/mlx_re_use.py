#!/usr/bin/env /Volumes/sw/conda_envs/reuse-mlx/bin/python
from pathlib import Path
from mlx_speech.generation.reuse import REUSEEnhancer
import soundfile as sf
import numpy as np
import mlx.core as mx  # 引入 MLX 核心库
import math
import time
import logging

# ==========================================
# 0. 配置日志记录
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# ==========================================
# 1. 读取 WAV 文件 (得到 Numpy 数组)
# ==========================================
noisy_waveform_np, sr = sf.read('/Users/larry/codehub/RE-USE/noisy_audio/mQlxALUw3h4-12s-16k.wav')

# 确保是单声道 (T,) 格式。如果是立体声 (T, 2)，将其转为单声道
if noisy_waveform_np.ndim > 1:
    noisy_waveform_np = np.mean(noisy_waveform_np, axis=1)

# 🟢 新增：计算并记录音频原始时长
audio_duration = len(noisy_waveform_np) / sr
logging.info(f"输入音频时长: {audio_duration:.2f} 秒 | 采样率: {sr} Hz | 样本数: {len(noisy_waveform_np)}")

# ==========================================
# 2. 转换为 MLX 格式 (Numpy -> mx.array)
# ==========================================
# 文档要求 float waveform，显式指定 float32 避免类型问题
mx_waveform = mx.array(noisy_waveform_np, dtype=mx.float32)

# 检查 shape 是否符合文档要求的 (T,) 或 (1, T)
print(f"输入 MLX 形状: {mx_waveform.shape}")

# ==========================================
# 3. 模型推理 (输入 mx.array, 输出 mx.array)
# ==========================================
enhancer = REUSEEnhancer.from_dir(Path("/Users/larry/Documents/re-use-semamba-mlx"))

# 1. 构建计算图
logging.info("开始构建计算图...")
start_graph = time.time()
mx_clean = enhancer.enhance(mx_waveform, in_sr=16000, chunk_size_s=4.0)
graph_time = time.time() - start_graph
logging.info(f"构建计算图耗时: {graph_time:.4f} 秒")

# 2. 显式触发 GPU 计算 (关键！)
logging.info("开始 GPU 实际推理计算...")
start_eval = time.time()
mx.eval(mx_clean)  # 强制 MLX 立即执行计算图
eval_time = time.time() - start_eval
logging.info(f"GPU 实际计算耗时: {eval_time:.4f} 秒")

# ==========================================
# 4. 转换回 Numpy 格式 (mx.array -> Numpy)
# ==========================================
# 3. 转换格式
logging.info("开始转换回 Numpy 格式...")
start_np = time.time()
clean_np = np.array(mx_clean)
np_time = time.time() - start_np
logging.info(f"转换为 Numpy 耗时: {np_time:.4f} 秒")

# 文档提到输出已经 clamped 到 [-1, 1]，但为了绝对安全，我们可以再 clip 一次
clean_np = np.clip(clean_np, -1.0, 1.0)

# ==========================================
# 5. 保存为 WAV 文件
# ==========================================
sf.write(
    file='clean_audio.wav',
    data=clean_np,
    samplerate=16000,       # 保持与 in_sr 一致
    subtype='PCM_16'        # 保存为标准 16-bit 格式
)

logging.info("MLX 增强后的音频已成功保存为 clean_audio.wav")
