# LongCat-AudioDiT Rust/Candle 接入说明

## 目标

在 `aphelios_tts` 中新增一条基于 Rust + Candle 的 `LongCat-AudioDiT-1B` 推理链路，先以本地模型目录加载为主，优先支持 Apple Silicon 上的 `metal` feature。

python版本代码在:/Users/larry/coderesp/LongCat-AudioDiT
google/umt5-base模型地址:/Volumes/sw/pretrained_models/umt5-base
LongCat-AudioDiT-1B模型地址:/Volumes/sw/pretrained_models/LongCat-AudioDiT-1B
rust版本代码路径: aphelios_tts/src/longcat_audiodit

参考的 Python 调用形式如下：

```bash
/Volumes/sw/conda_envs/lcataudio/bin/python /Users/larry/coderesp/LongCat-AudioDiT/inference.py \
  --text "在20世纪上半叶的中国乡村，有两个巨大的历史进程值得注意，它们使此一时期的中国有别于前一时代。" \
  --prompt_text "贝克莱的努力并未产生有形的结果" \
  --prompt_audio "/Volumes/sw/video/youyi-5s.wav" \
  --output_audio output3.wav \
  --model_dir /Volumes/sw/pretrained_models/LongCat-AudioDiT-1B
```

## 当前可确认的模型结构

从 `/Volumes/sw/pretrained_models/LongCat-AudioDiT-1B/config.json` 和 `model.safetensors` 可以确认 LongCat-AudioDiT-1B 至少由三块组成：

1. `text_encoder.*`
   使用 `UMT5` 编码文本，隐藏维度 `768`，12 层。
2. `transformer.*`
   主干是 `AudioDiT`，维度 `1536`，24 层，多头数 `24`，使用 `cross attention`、`AdaLN`、`text conv`、`latent condition`。
3. `vae.*`
   波形 latent VAE，latent 维度 `64`，`latent_hop = 2048`，采样率 `24000`。

这和仓库里现有的 `Qwen3-TTS` 自回归 codec 路径完全不同，不能复用原有 `talker/code_predictor/codec` 主流程，只能复用基础设施：

1. Candle 设备与 `metal` feature。
2. 音频读写、重采样、公用工具。
3. safetensors / tokenizer / CLI 接线方式。

## 可行性判断

结论：可行，但要按阶段推进，不能假设一两处改动就能直接跑通。

原因：

1. 权重是标准 `safetensors`，配置完整，适合 Candle 侧解析。
2. 模型目录里已经包含 `config.json` 和 `model.safetensors`，本地加载不依赖在线下载。
3. `aphelios_tts` 已经启用了 Candle，并且已有 `metal` feature 组织方式，可以直接沿用。

## 主要难点

### 1. 不是自回归，而是 diffusion 推理

LongCat-AudioDiT 的主干不是逐 token 生成，而是：

1. 文本编码得到条件特征。
2. 将 prompt audio 编到 latent 空间，作为 voice cloning 条件。
3. 在 latent 空间做多步 diffusion / denoising。
4. 用 VAE decoder 把 latent 还原回波形。

这要求 Rust 端实现：

1. timestep embedding。
2. diffusion scheduler。
3. CFG / APG guidance。
4. prompt latent 拼接与目标时长控制。

### 2. UMT5 encoder 复刻成本

文本编码不是仓库现成模型，而是 `UMT5`。需要两件事：

1. tokenizer 对齐 `google/umt5-base`。
2. encoder 前向与相对位置 bias 对齐。

如果 Candle 现成 `T5` 实现和 UMT5 权重命名不完全一致，需要单独适配。

### 3. VAE 结构不是仓库现有 codec

LongCat 的 `vae.*` 权重命名显示其实现包含：

1. encoder / decoder。
2. `Snake` 类激活参数。
3. 多层下采样和上采样卷积。
4. `quant/post_quant` 风格 latent 映射。

这和 `Qwen3-TTS` 的 codec decoder 不是同一个结构，不能直接套。

### 4. prompt voice cloning 路径

Python 用法里 `prompt_text + prompt_audio` 联合决定声音风格。Rust 侧不仅要支持：

1. 纯文本 TTS。
2. prompt audio voice cloning。

还要明确：

1. prompt audio 是否需要固定采样率重采样到 `24kHz`。
2. duration 是总 latent 帧数还是生成段帧数。
3. prompt latent 是直接前缀拼接，还是作为独立 latent condition。

### 5. guidance_method 细节

README 明确支持：

1. `cfg`
2. `apg`

其中 `apg` 是 LongCat 特别强调的推理技巧。首版若没有完全复刻其数学细节，只能先提供接口和默认调度，不能宣称与 Python 完全一致。

## 推荐推进方式

### 阶段 1

先完成 Rust 侧基础骨架：

1. `config.json` 解析。
2. 本地模型目录发现。
3. `metal` 优先设备选择。
4. 权重索引与组件存在性校验。
5. 推理请求结构体。
6. duration / latent 长度 / prompt 时长的规划逻辑。

目标是先把“模型可识别、请求可规划、代码结构稳定”做出来。

### 阶段 2

补齐组件前向：

1. UMT5 encoder。
2. AudioDiT blocks。
3. waveform VAE encoder / decoder。
4. diffusion scheduler。

这一阶段要以“先能跑通 zero-shot TTS”为准。

### 阶段 3

再补 voice cloning 和 APG 对齐：

1. prompt audio 编码。
2. prompt_text + text 的拼接策略。
3. APG / CFG guidance 完整实现。
4. 与 Python 输出做误差对比。

## 阶段

目前可以合成语音, 但是质量不如python版本. 是否可以结合不同阶段, 对比和python在相同输入下的输出向量数据, 以此判断是哪一阶段导致的质量区别