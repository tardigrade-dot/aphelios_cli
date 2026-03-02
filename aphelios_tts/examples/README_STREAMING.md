# Qwen3-TTS Streaming Speech Synthesis Example

This example demonstrates **real-time streaming TTS** with the Qwen3-TTS model, synthesizing and playing audio chunks as they are generated.

## Features

- **Streaming synthesis**: Generates audio in small chunks (~400ms each)
- **Real-time playback**: Plays each chunk immediately using `cpal`
- **Voice cloning**: Supports ICL (In-Context Learning) voice cloning from reference audio
- **Low latency**: Configurable chunk size for trade-off between latency and smoothness

## Architecture

The streaming implementation leverages Qwen3-TTS's autoregressive architecture:

1. **Talker Model**: Generates semantic tokens autoregressively from text input
2. **Code Predictor**: For each semantic token, generates 15 acoustic tokens
3. **Decoder12Hz**: Converts 16-codebook codec tokens to 24kHz audio waveform

The streaming session:
- Prefills the model with the input text and voice prompt
- Generates audio chunks incrementally (default 5 frames ≈ 400ms)
- Yields each chunk as an `AudioBuffer` for immediate playback
- Maintains GPU-resident state for efficient generation

## Usage

### Basic Example

```bash
cargo run --example qwen3_tts_streaming --features metal
```

### Code Example

```rust
use aphelios_tts::{Qwen3TTS, AudioBuffer, Language, SynthesisOptions};

// Load model
let model = Qwen3TTS::from_pretrained(model_path, device)?;

// Create voice clone prompt
let ref_audio = AudioBuffer::load("reference.wav")?;
let prompt = model.create_voice_clone_prompt(&ref_audio, Some("reference text"))?;

// Configure streaming (smaller chunks = lower latency)
let options = SynthesisOptions {
    chunk_frames: 5,  // ~400ms per chunk
    ..Default::default()
};

// Stream with real-time playback
let mut session = model.synthesize_voice_clone_streaming(
    "Hello, this is my cloned voice!",
    &prompt,
    Language::English,
    options,
)?;

for chunk in session {
    let audio = chunk?;
    // Play audio.samples immediately (f32 samples in [-1.0, 1.0])
}
```

## Configuration

### Chunk Size

The `chunk_frames` parameter controls latency vs. smoothness:

| `chunk_frames` | Duration | Latency | Smoothness |
|----------------|----------|---------|------------|
| 3              | ~240ms   | Very Low | May stutter |
| 5              | ~400ms   | Low      | Good       |
| 10             | ~800ms   | Medium   | Very Smooth |
| 20             | ~1.6s    | High     | Excellent  |

### Voice Cloning Modes

1. **X-Vector Only** (faster, no reference text needed):
   ```rust
   let prompt = model.create_voice_clone_prompt(&ref_audio, None)?;
   ```

2. **ICL (In-Context Learning)** (better quality, requires reference text):
   ```rust
   let prompt = model.create_voice_clone_prompt(&ref_audio, Some("reference text"))?;
   ```

## Performance

Typical performance on Apple M1/M2:
- **Time to first audio**: ~500-800ms (prefill + first chunk generation)
- **Real-time factor**: 0.3-0.5x (generation is 2-3x faster than playback)
- **Memory usage**: ~2-4GB (depending on model size)

## Dependencies

- `cpal`: Cross-platform audio playback
- `rubato`: High-quality audio resampling (if needed)

## Troubleshooting

### No audio output

1. Check that your system has a default output device
2. Verify audio permissions on macOS
3. Try increasing `chunk_frames` for smoother playback

### Stuttering playback

1. Increase `chunk_frames` to 10 or higher
2. Close other audio applications
3. Use release build: `cargo run --release`

### High latency

1. Decrease `chunk_frames` to 3-5
2. Use Metal/CUDA acceleration
3. Reduce `max_length` if synthesizing short text

## Advanced Usage

### Custom Voice Design (VoiceDesign models only)

```rust
let session = model.synthesize_voice_design_streaming(
    "Text to synthesize",
    "A cheerful young female voice",  // Voice description
    Language::English,
    options,
)?;
```

### Preset Speakers (CustomVoice models only)

```rust
use aphelios_tts::Speaker;

let session = model.synthesize_streaming(
    "Text to synthesize",
    Speaker::Ryan,  // Preset speaker
    Language::English,
    options,
)?;
```

## References

- [Qwen3-TTS Paper](https://arxiv.org/abs/xxxx.xxxxx)
- [Qwen3-TTS GitHub](https://github.com/QwenLM/Qwen3-TTS)
- [qwen3-tts-rs](https://github.com/tardigrade-dot/qwen3-tts-rs)
