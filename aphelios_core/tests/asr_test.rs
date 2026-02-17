// https://github.com/openai/whisper/blob/main/whisper/model.py/rgs
// TODO:
// - Batch size greater than 1.
// - More token filters (SuppressBlanks, ApplyTimestampRules).

use std::path::Path;

use anyhow::Result;
use aphelios_core::{
    asr::{generate_srt, run_whisper},
    common::core_utils,
};
use candle_core::{Device, MetalDevice, backend::BackendDevice};

use tracing::{error, info};

#[test]
fn asr_test() -> Result<()> {
    core_utils::init_tracing();
    let metal = MetalDevice::new(0)?;
    let device = Device::Metal(metal);

    let model_dir = "/Volumes/sw/pretrained_models/whisper-large-v3-turbo";
    let input = "/Users/larry/coderesp/aphelios_cli/test_data/mQlxALUw3h4_16k.wav";

    let output_path = Path::new(input)
        .with_extension("srt")
        .to_str()
        .unwrap()
        .to_string();

    println!("正在处理音频：{}", input);
    let result = run_whisper(model_dir, input, &device);
    info!("print asr result");
    match result {
        Ok(segments) => {
            for segment in &segments {
                // 1. 打印大段落的边界（可选，用颜色或分隔线增加可读性）
                println!(
                    "\n[Segment: {:.2}s - {:.2}s]",
                    segment.start,
                    segment.start + segment.duration
                );

                if !segment.sub_segments.is_empty() {
                    // 2. 如果有详细时间戳，逐行对齐打印
                    for sub in &segment.sub_segments {
                        println!(
                            "  => [{:>6.2}s -> {:>6.2}s]  {}",
                            sub.start,
                            sub.end,
                            sub.text.trim()
                        );
                    }
                } else {
                    // 3. 兜底逻辑：如果没有子片段，打印整段文本（并清理掉可能残留的特殊字符）
                    let clean_text = segment.dr.text.replace(
                        |c: char| c == '<' || c == '|' || c == '>' || c.is_numeric() || c == '.',
                        "",
                    );
                    if !clean_text.trim().is_empty() {
                        println!("  (Raw): {}", clean_text.trim());
                    }
                }
            }
            let _ = generate_srt(&segments, output_path.as_str());
        }
        Err(e) => {
            error!("ASR task failed: {:?}", e);
        }
    }

    Ok(())
}
