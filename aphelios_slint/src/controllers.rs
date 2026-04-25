pub mod demucs;
pub mod ocr;
pub mod search;
pub mod tts;

use crate::config::AppSettings;
use aphelios_core::traits::{OcrEngine, SearchEngine, TtsEngine};
use std::sync::{Arc, Mutex};

/// 应用全局上下文，持有所有服务引擎和配置
pub struct AppContext {
    pub ocr_engine: Arc<Mutex<dyn OcrEngine>>,
    pub tts_engine: Arc<dyn TtsEngine>,
    pub search_engine: Arc<dyn SearchEngine>,
    pub settings: Arc<Mutex<AppSettings>>,
    pub audio_output_path: Arc<Mutex<Option<String>>>,
}

impl AppContext {
    pub fn new(
        ocr: Arc<Mutex<dyn OcrEngine>>,
        tts: Arc<dyn TtsEngine>,
        search: Arc<dyn SearchEngine>,
        settings: AppSettings,
    ) -> Self {
        Self {
            ocr_engine: ocr,
            tts_engine: tts,
            search_engine: search,
            settings: Arc::new(Mutex::new(settings)),
            audio_output_path: Arc::new(Mutex::new(None)),
        }
    }

    pub fn get_settings(&self) -> AppSettings {
        self.settings.lock().unwrap().clone()
    }

    pub fn save_settings(&self, settings: AppSettings) {
        let mut lock = self.settings.lock().unwrap();
        *lock = settings.clone();
        let _ = settings.save();
    }
}
