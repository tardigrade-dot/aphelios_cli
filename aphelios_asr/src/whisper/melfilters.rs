use anyhow::Result;
use candle_transformers::models::whisper::Config;

pub fn get_mel_filters(config: &Config) -> Result<Vec<u8>> {
    let mel_bytes = match config.num_mel_bins {
        80 => include_bytes!("melfilters.bytes").as_slice(),
        128 => include_bytes!("melfilters128.bytes").as_slice(),
        nmel => anyhow::bail!("unexpected num_mel_bins {nmel}"),
    };
    Ok(mel_bytes.to_vec())
}
