use anyhow::Result;
use aphelios_asr::granite::GraniteModel;
use aphelios_core::utils::logger;

#[test]
fn granite_test() -> Result<()> {
    logger::init_logging();

    let model_id = "/Volumes/sw/pretrained_models/granite-4.0-1b";
    let mut model = GraniteModel::new(model_id)?;
    let _res = model.generate("introduce yourself?", 512, 0.1)?;
    Ok(())
}
