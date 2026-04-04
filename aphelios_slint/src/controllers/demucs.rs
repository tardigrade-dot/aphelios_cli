use anyhow::{anyhow, Result};
use std::sync::Arc;
use tracing::info;

use crate::controllers::AppContext;

pub struct DemucsLogic {
    ctx: Arc<AppContext>,
}

impl DemucsLogic {
    pub fn new(ctx: Arc<AppContext>) -> Self {
        Self { ctx }
    }

    pub fn start_separation(
        &self,
        audio_file: String,
        model_path: String,
        output_dir: String,
        separation_mode: String,
        progress_callback: impl Fn(f32) + Send + Sync + 'static,
        on_complete: impl Fn(Result<()>) + Send + 'static,
    ) {
        // 更新设置 - 只保存模型路径
        let mut settings = self.ctx.get_settings();
        settings.demucs_model_path = Some(model_path.clone());
        self.ctx.save_settings(settings);

        // 如果输出目录为空，使用输入音频的同路径
        let output = if output_dir.is_empty() {
            std::path::Path::new(&audio_file)
                .parent()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|| ".".to_string())
        } else {
            output_dir.clone()
        };

        std::thread::spawn(move || {
            info!(
                "Starting Demucs separation: audio={}, mode={}, output={}",
                audio_file, separation_mode, output
            );

            let progress_cb = Arc::new(progress_callback);

            // 根据分离模式调用不同的函数
            let result = if separation_mode == "vocals_instrumental" {
                // 人声/伴奏分离模式
                aphelios_core::demucs::run_vocal_separation(
                    &model_path,
                    &audio_file,
                    Some(progress_cb.clone()),
                )
            } else {
                // 四轨分离模式
                aphelios_core::demucs::run_demucs(
                    &model_path,
                    &audio_file,
                    Some(progress_cb.clone()),
                )
            };

            // 完成时调用回调
            progress_cb(1.0);

            // Convert Result<(), String> to Result<(), anyhow::Error>
            on_complete(result.map_err(|e| anyhow!(e)));
        });
    }
}
