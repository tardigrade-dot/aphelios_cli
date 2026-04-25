use std::path::Path;

use anyhow::Result;
use aphelios_asr::qwen3llm::llm::qwen3_llm;

#[test]
fn test_qwen3llm() -> Result<()> {
    let qwen3llm_model_dir = Path::new("/Volumes/sw/pretrained_models/Qwen3-0.6B");

    qwen3_llm("教我学习微积分", qwen3llm_model_dir.to_str().unwrap())?;

    Ok(())
}
