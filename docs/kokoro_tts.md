# Kokoro TTS: Current Status & Future Recommendations

This document summarizes the progress, technical architecture, and future roadmap for the Kokoro TTS implementation within the `aphelios_tts` crate.

## Current Completion Status

| Feature | Status | Implementation Details |
| :--- | :--- | :--- |
| **ONNX Inference** | ✅ **Done** | High-performance inference using `ort` (ONNX Runtime). Supports `model.onnx`, `logits` fallback, and dynamic output extraction. |
| **Voice Style Loading** | ✅ **Done** | Loading of `f32` style vectors from `.bin` files via `KokoroVoices` and `ndarray`. |
| **English Phonemization** | ✅ **Done** | Currently using `misaki-rs` for US and UK English (Text -> IPA). |
| **Multi-dialect API** | ✅ **Done** | `KokoroModel::generate` supports `KokoroLanguage` enum for EnUs and EnGb. |
| **Integration Testing** | ✅ **Done** | Comprehensive tests in `tests/koko_tts_test.rs` covering multi-voice and speed variation. |

## Future Expectations: Kokoro-82M-v1.1-zh

To support the Chinese model (`v1.1-zh`), the following steps are required:

1.  **Model Acquisition**: Download `Kokoro-82M-v1.1-zh-ONNX` and its corresponding `tokenizer.json`.
2.  **Phonemization (G2P)**:
    *   Implement the `Zh` branch in `KokoroPhonemizer`.
    *   This requires Chinese word segmentation (using the already integrated `jieba-rs`) and a mapping from characters/pinyin to the specific IPA symbols used by the `v1.1-zh` model.
3.  **Tokenization Integration**: Ensure the new `tokenizer.json` structure is compatible with our `KokoroTokenizer` fallback logic.

## Technical Recommendations: G2P Maturity

While `misaki-rs` provides a pure-Rust solution for English, we've identified potential maturity concerns. For a more robust, industry-standard solution that inherently supports Chinese, we recommend exploring the **`espeak-rs`** path used by projects like `Kokoros`.

### Option A: `misaki-rs` (Current Path)
*   **Pros**: Pure Rust (no system dependencies), self-contained, POS-aware.
*   **Cons**: Less mature (v0.3.0), English-only (currently), may require custom logic for Chinese.

### Option B: `espeak-rs` (Recommended for Stability)
*   **Pros**: Highly mature, supports 100+ languages (including ZH), standard in Kokoro community.
*   **Cons**: Requires system dependency (`libespeak-ng` via `brew install espeak-ng`), uses a global mutex in Rust.

> [!TIP]
> If stability and multi-lingual support are more important than binary portability, switching to **Path B** (`espeak-rs`) is the most future-proof decision for the `aphelios_tts` crate.

## Project Structure

- `aphelios_tts/src/kokoro/mod.rs`: Main ONNX session and API.
- `aphelios_tts/src/kokoro/phonemize.rs`: G2P layer (English currently implemented).
- `aphelios_tts/src/kokoro/tokenizer.rs`: Robust IPA-to-Token conversion.
- `aphelios_tts/src/kokoro/voice.rs`: Voice style binary loading.
- `aphelios_tts/tests/koko_tts_test.rs`: Integration and performance tests.
