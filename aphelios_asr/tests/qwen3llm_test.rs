use aphelios_asr::base::qwen3llm::qwen3_llm;

#[test]
fn test_qwen3llm() -> anyhow::Result<()> {
    let qwen3llm_model_dir = std::path::Path::new("/Volumes/sw/pretrained_models/Qwen3-0.6B");

    qwen3_llm("教我学习微积分", qwen3llm_model_dir.to_str().unwrap())?;

    Ok(())
}
