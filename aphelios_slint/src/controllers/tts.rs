use anyhow::Result;
use std::sync::Arc;
use tracing::info;

use crate::controllers::AppContext;
use aphelios_core::utils::progress::AppProgressBar;

pub struct TtsLogic {
    ctx: Arc<AppContext>,
}

impl TtsLogic {
    pub fn new(ctx: Arc<AppContext>) -> Self {
        Self { ctx }
    }

    pub fn ctx(&self) -> &Arc<AppContext> {
        &self.ctx
    }

    pub fn start_tts(
        &self,
        text: String,
        model_path: String,
        output_path: String,
        ref_audio_path: String,
        ref_text: String,
        progress_callback: impl Fn(f32) + Send + Sync + 'static,
        on_complete: impl Fn(Result<()>) + Send + 'static,
    ) {
        // 更新设置
        let mut settings = self.ctx.get_settings();
        settings.tts_model_path = Some(model_path.clone());
        settings.tts_output_path = Some(output_path.clone());
        settings.tts_ref_audio_path = Some(ref_audio_path.clone());
        settings.tts_ref_text = Some(ref_text.clone());
        self.ctx.save_settings(settings);

        let engine = self.ctx.tts_engine.clone();

        std::thread::spawn(move || {
            info!("Starting TTS: text={}, output={}", text, output_path);

            let progress_bar =
                AppProgressBar::with_ui(indicatif::ProgressBar::hidden(), move |p| {
                    progress_callback(p)
                });

            let result = engine.generate(
                &model_path,
                &ref_audio_path,
                &ref_text,
                &text,
                &output_path,
                Some(progress_bar),
            );

            on_complete(result);
        });
    }

    pub fn start_batch_tts(
        &self,
        txt_file_path: String,
        model_path: String,
        ref_audio_path: String,
        ref_text: String,
        progress_callback: impl Fn(f32) + Send + Sync + 'static,
        on_complete: impl Fn(Result<Vec<String>>) + Send + 'static,
    ) {
        // 更新设置
        let mut settings = self.ctx.get_settings();
        settings.tts_model_path = Some(model_path.clone());
        settings.tts_ref_audio_path = Some(ref_audio_path.clone());
        settings.tts_ref_text = Some(ref_text.clone());
        self.ctx.save_settings(settings);

        let engine = self.ctx.tts_engine.clone();

        std::thread::spawn(move || {
            info!("Starting batch TTS: txt_file={}", txt_file_path);

            let progress_bar =
                AppProgressBar::with_ui(indicatif::ProgressBar::hidden(), move |p| {
                    progress_callback(p)
                });

            let result = engine.generate_batch(
                &model_path,
                &ref_audio_path,
                &ref_text,
                &txt_file_path,
                "",
                3,
                Some(progress_bar),
            );

            on_complete(result);
        });
    }
}
