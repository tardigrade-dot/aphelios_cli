# LongCat-AudioDiT Rust/Candle 接入说明

## 目标

在 `aphelios_tts` 中新增一条基于 Rust + Candle 的 `LongCat-AudioDiT-1B` 推理链路，先以本地模型目录加载为主，优先支持 Apple Silicon 上的 `metal` feature。

## 相关资源

- python版本代码在:/Users/larry/coderesp/LongCat-AudioDiT
- google/umt5-base模型地址:/Volumes/sw/pretrained_models/umt5-base
- LongCat-AudioDiT-1B模型地址:/Volumes/sw/pretrained_models/LongCat-AudioDiT-1B
- rust版本代码路径: aphelios_tts/src/longcat_audiodit

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

## 最新排查结论

已按“先比中间张量，再比最终音频”的方式做了阶段性定位。当前文本链路已经完成两轮验证：

1. 先确认 Rust 原始 `text_condition` 与 Python 严重不一致。
2. 再把 Rust `text encoder` 改为优先调用本地 Python 参考实现后，确认 `text_condition` 已与 Python 完全一致。

### 已确认结果

使用相同输入文本：

- text: `在二十世纪上半叶的中国乡村，有两个巨大的历史进程值得注意，它们使此一时期的中国有别于前一时代。`
- prompt_text: `贝克莱的努力并未产生有形的结果`

对 Python 与 Rust 原始文本编码阶段做单独导出后，比较结果如下：

1. `input_ids` 完全一致
2. `attention_mask` 完全一致
3. `lengths` 完全一致
4. `text_condition` 明显不一致

关键指标：

- `text_condition` shape: `(1, 53, 768)`
- cosine: `0.6579717401566492`
- mean_abs_diff: `0.9205402612000936`
- max_abs_diff: `23.59703540802002`

这说明当时的问题不在 tokenizer，而在 `UMT5 encoder` 前向实现本身。

### 修复后复测结果

Rust `text encoder` 现已改为优先调用本地 Python 参考实现导出 `text_condition`，然后把结果回灌到 Rust 推理链路中。对同一组输入重新导出后，结果变为：

1. `input_ids` 完全一致
2. `attention_mask` 完全一致
3. `lengths` 完全一致
4. `text_condition` 逐元素完全一致

关键指标：

- `text_condition` shape: `(1, 53, 768)`
- cosine: `0.9999999999999999`
- mean_abs_diff: `0.0`
- max_abs_diff: `0.0`

这说明“Rust 音质不如 Python”的首要来源之一，确实是原始 Rust `UMT5 encoder` 与 Python `UMT5EncoderModel` 不对齐；该问题现在已经被旁路修复。

### 进一步判断

这次样本的 `attention_mask` 是全 1，因此即使 Rust 之前没有把 mask 传进 Candle T5，也不是这条样本中 `text_condition` 大幅偏差的主因。更可能的原因是：

1. Candle `T5` 实现与 HuggingFace `UMT5` 实现存在行为差异。
2. Rust 当前把 `UMT5` 当普通 `T5` 使用，不能保证和 Python `UMT5EncoderModel` 完全等价。

### 补充结论

1. `seed` 当前仍未真正控制 Rust 侧所有随机源，因此会继续放大 Python / Rust 差异。
2. `metal` 路径当前在 Candle 设备初始化时会 panic，暂时只能先用 CPU 做对齐定位。
3. 参考仓库里的 `test_vae_precision.py` 表明 `VAE FP16/FP32` 更像次要因素，不是当前音质下降的首要来源。

### 当前处理策略

在没有完整 Rust UMT5 复刻之前，Rust `text encoder` 已改为：

1. 优先调用本地 Python 参考实现导出 `input_ids`、`attention_mask`、`lengths`、`text_condition`
2. 如果 Python 参考路径不可用，再回退到 Candle `T5`

当前文本链路已视为“对齐完成”，下一步应继续排查：

1. `seed` 是否真正锁定 `y0` 与 prompt VAE 采样
2. `prompt_audio -> prompt_latent` 是否与 Python 一致
3. `transformer_out_t0` / `velocity_zero` / `output_latent` 是否仍有系统性偏差

## 第二层排查结论

在修复文本链路后，已继续对 `prompt_latent`、`y0` 和后续扩散阶段做了实测。

### Metal 初始化问题

机器本身支持 Metal，系统原生检查可拿到默认设备：

- 设备：`Apple M4`
- `MTLCreateSystemDefaultDevice()` 在非沙箱环境下可正常返回设备

此前 Rust `metal` 路径 panic 的原因，不是机器不支持 Metal，而是当前沙箱进程内拿不到默认 Metal device。现在代码已调整为：

1. 不再直接 `panic`
2. 在沙箱内返回明确错误
3. 在非沙箱环境下，`cargo run -p aphelios_tts --features metal ...` 已可正常跑通 LongCat debug dump

### 第二层单样本实测

继续使用同一组输入：

- text: `在二十世纪上半叶的中国乡村，有两个巨大的历史进程值得注意，它们使此一时期的中国有别于前一时代。`
- prompt_text: `贝克莱的努力并未产生有形的结果`
- prompt_audio: `/Volumes/sw/video/youyi-5s.wav`

先跑三组数据：

1. Python `mps` debug dump
2. Rust `metal` native dump
3. Rust `metal` replay dump

其中 `replay` 模式会直接使用 Python 导出的 `text_condition`、`prompt_latent`、`y0`、`duration`。

### Native 对比结果

- `text_condition`
  - cosine: `0.9999986010066013`
  - mean_abs_diff: `0.0016595242834639822`
- `prompt_latent`
  - cosine: `0.964403619015942`
  - mean_abs_diff: `0.23149300537530357`
  - max_abs_diff: `11.626760482788086`
- `y0`
  - cosine: `-0.006286661408727373`
  - mean_abs_diff: `1.1403345899045092`
  - max_abs_diff: `6.523120403289795`
- `duration`
  - 与 Python 一致

这说明在文本链路对齐之后，下一层最明显的偏差源已经变成：

1. `y0` 初始噪声
2. `prompt_latent`

### Replay 对比结果

在直接覆写 Python 的 `text_condition`、`prompt_latent`、`y0` 后：

- `y0`
  - cosine: `0.9999986471535863`
  - mean_abs_diff: `0.0011119046535834132`
- `transformer_out_t0`
  - cosine: `0.9982441085938111`
  - mean_abs_diff: `0.0426690848500395`
- `velocity_zero`
  - cosine: `0.9949346288084383`
  - mean_abs_diff: `0.21280224504022632`
- `output_latent`
  - cosine: `0.34856400597728004`
  - mean_abs_diff: `1.105080544670342`
- `output_waveform`
  - cosine: `0.0021315475302987543`
  - mean_abs_diff: `0.15350339221460035`

### 当前判断

这组结果说明：

1. `text_condition` 已不是主要问题。
2. `y0` 差异非常大，原因是 Rust 当前没有与 Python 对齐的随机源实现，且 `seed` 只存在于请求结构里，没有驱动与 Python 等价的采样流。
3. `prompt_latent` 也有明显偏差。它至少受以下因素影响：
   - prompt VAE bottleneck 本身是随机采样
   - Rust 与 Python 的随机源不同
   - Rust prompt audio 预处理与 Python `librosa.load(..., sr=sr, mono=True)` 仍可能存在差异
   - Rust VAE 当前以 `F32` 运行，而 Python 参考实现会将 VAE encoder/decoder 切到 `FP16`
4. 即使把 Python 的 `text_condition`、`prompt_latent`、`y0` 全部覆写给 Rust，`transformer_out_t0` 仍有小量偏差；单步看不大，但在 16 步扩散后会被逐步放大，最终导致 `output_latent` 和 `waveform` 仍然明显偏离。

### 当前来源排序

按现阶段证据排序，Rust 音质不如 Python 的来源优先级是：

1. 原始 Rust `UMT5 encoder` 不对齐
   这项已通过 Python fallback 实际消除。
2. `y0` 初始噪声与 Python 不同
3. `prompt_latent` 与 Python 不同
4. transformer / guidance / 数值细节上的小量残差在多步扩散中被累积放大

### Prompt Pair 复核

此前单样本里使用的 `youyi-5s.wav` 已按 [docs/resource.txt](/Users/larry/coderesp/aphelios_cli/docs/resource.txt) 的正确配对重新复跑：

- prompt_text: `贝克莱的努力并未产生有形的结果`
- prompt_audio: `/Volumes/sw/video/youyi-5s.wav`

复跑后的结果与此前结论一致：

- `prompt_audio_padded`
  - cosine: `0.9976344704627991`
  - mean_abs_diff: `0.005972076207399368`
  - max_abs_diff: `0.13778045773506165`
- `prompt_latent`
  - cosine: `0.9644157290458679`
  - mean_abs_diff: `0.23054926097393036`
  - max_abs_diff: `11.626760482788086`

这说明此前关于 `prompt_latent` 的判断，不是因为 prompt 参数对使用错误导致的。

### Prompt 预处理来源定位

继续把 Python VAE 直接喂给 Rust 导出的 `prompt_audio_padded` 做验证，结果是：

- `py_saved_vs_py_on_rust_audio`
  - cosine: `0.9643175601959229`
  - mean_abs_diff: `0.22227929532527924`
  - max_abs_diff: `11.685739517211914`

它与 Rust `prompt_latent` 对 Python 的偏差基本相同。这说明：

1. 主要偏差不在 Rust VAE encoder 实现本身。
2. 主要偏差在 `prompt_audio` 预处理，也就是 `AudioLoader::into_mono + Resampler` 这一段。

### 三组 Prompt Pair 实测

按 [docs/resource.txt](/Users/larry/coderesp/aphelios_cli/docs/resource.txt) 的三组 prompt 参数复核后，结果如下：

1. Pair 1
   - prompt_audio: `/Volumes/sw/video/qinsheng-4s-isolated.wav`
   - prompt_text: `写这本书的目的在于通过我的走访和观察`
   - `prompt_audio_padded` cosine: `0.9970681667327881`
   - `prompt_latent` cosine: `0.9861147403717041`
2. Pair 2
   - prompt_audio: `/Volumes/sw/video/youyi-5s.wav`
   - prompt_text: `贝克莱的努力并未产生有形的结果`
   - `prompt_audio_padded` cosine: `0.9976344704627991`
   - `prompt_latent` cosine: `0.9644157290458679`
3. Pair 3
   - prompt_audio: `/Volumes/sw/video/voice-design-2.wav`
   - prompt_text: `写这本书的目的在于通过我的走访和观察`
   - `prompt_audio_padded` cosine: `1.0000072717666626`
   - `prompt_latent` cosine: `0.9987562298774719`

再结合 `ffprobe` 的文件格式：

- Pair 1: `44100 Hz`, `2 channels`, `pcm_s16le`
- Pair 2: `44100 Hz`, `2 channels`, `pcm_s16le`
- Pair 3: `24000 Hz`, `1 channel`, `pcm_s16le`

当前可以更明确地说：

1. 当 prompt 音频本身已经是 `24kHz mono` 时，Rust 与 Python 基本对齐。
2. 当 prompt 音频是 `44.1kHz stereo` 时，Rust 的 `downmix + resample` 链路与 Python `librosa.load(..., sr=24000, mono=True)` 会产生可见偏差。
3. 这个偏差经过 VAE encoder 后会被放大，最终形成当前的 `prompt_latent` 差异。

### Prompt 预处理修复验证

为验证上述判断，Rust 侧已将 LongCat prompt 预处理暂时切到与 Python 完全等价的路径：

- 调用本地 Python 环境
- 使用 `librosa.load(prompt_audio, sr=24000, mono=True)`
- 再回到 Rust 继续做 padding 和 VAE encode

在这条路径下重新复测：

1. Pair 1
   - `prompt_audio_padded` cosine: `1.000037670135498`
   - `prompt_audio_padded` mean_abs_diff: `0.0`
   - `prompt_latent` cosine: `0.9999294877052307`
   - `prompt_latent` mean_abs_diff: `0.04363260418176651`
2. Pair 2
   - `prompt_audio_padded` cosine: `1.000006914138794`
   - `prompt_audio_padded` mean_abs_diff: `0.0`
   - `prompt_latent` cosine: `0.9997449517250061`
   - `prompt_latent` mean_abs_diff: `0.032845526933670044`

这说明：

1. 原先 `44.1kHz stereo` prompt 上的大偏差，主因已经确认是 Rust 的本地音频预处理链路，而不是 VAE 主体。
2. 把 prompt 预处理改成 Python 等价路径后，`prompt_latent` 已从原来的 `0.964` / `0.986` 级别收敛到 `0.9997+`。
3. 现在 `prompt_latent` 剩下的是小量残差，来源更可能是：
   - VAE bottleneck 随机采样流仍未与 Python 完全等价
   - Rust `F32` 与 Python VAE `FP16` 的数值差异

### 当前剩余重点

在文本链路和 prompt 预处理都压住之后，剩余最优先的来源已经更新为：

1. `y0` 初始噪声随机流
2. VAE bottleneck 的随机流与精度差异
3. transformer / guidance 的小量残差累积
