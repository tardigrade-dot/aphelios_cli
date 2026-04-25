use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use tracing::info;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AppSettings {
    // OCR 设置
    pub ocr_model_path: Option<String>,
    pub ocr_output_dir: Option<String>,

    // ASR 设置 (语音识别)
    pub asr_model_path: Option<String>,
    pub asr_aligner_model_path: Option<String>,
    pub asr_vad_model_path: Option<String>,

    // SRT 设置 (文本对齐)
    pub srt_model_path: Option<String>,
    pub srt_min_segment_length: Option<i32>,
    pub srt_max_segment_length: Option<i32>,

    // TTS 设置
    pub tts_model_path: Option<String>,
    pub tts_output_path: Option<String>,
    pub tts_ref_audio_path: Option<String>,
    pub tts_ref_text: Option<String>,

    // Demucs 人声分离设置
    pub demucs_model_path: Option<String>,
    pub demucs_output_dir: Option<String>,
    pub demucs_separation_mode: Option<String>,

    // 搜索设置
    pub books_dir: Option<String>,
    pub search_mode: Option<String>, // "keyword"

    // 通用设置
    pub window_width: Option<i32>,
    pub window_height: Option<i32>,
}

#[allow(unused_variables, unused_imports)]
impl AppSettings {
    /// 获取配置文件路径
    fn config_path() -> Option<PathBuf> {
        dirs::config_dir().map(|mut p| {
            p.push("aphelios_cli");
            p.push("settings.json");
            p
        })
    }

    /// 从文件加载配置
    pub fn load() -> Result<Self> {
        if let Some(path) = Self::config_path() {
            if path.exists() {
                let content = fs::read_to_string(&path)?;
                let settings: AppSettings = serde_json::from_str(&content)?;
                return Ok(settings);
            }
        }
        Ok(AppSettings::default())
    }

    /// 保存配置到文件
    pub fn save(&self) -> Result<()> {
        if let Some(path) = Self::config_path() {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            let content = serde_json::to_string_pretty(self)?;
            info!("setting file :{}", path.to_str().unwrap());
            fs::write(path, content)?;
        }
        Ok(())
    }

    /// 更新并保存 OCR 设置
    pub fn update_ocr_settings(&mut self, model_path: Option<&str>, output_dir: Option<&str>) {
        if let Some(path) = model_path {
            self.ocr_model_path = Some(path.to_string());
        }
        if let Some(dir) = output_dir {
            self.ocr_output_dir = Some(dir.to_string());
        }
        let _ = self.save();
    }

    /// 更新并保存 TTS 设置
    pub fn update_tts_settings(
        &mut self,
        model_path: Option<&str>,
        output_path: Option<&str>,
        ref_audio: Option<&str>,
        ref_text: Option<&str>,
    ) {
        if let Some(path) = model_path {
            self.tts_model_path = Some(path.to_string());
        }
        if let Some(path) = output_path {
            self.tts_output_path = Some(path.to_string());
        }
        if let Some(path) = ref_audio {
            self.tts_ref_audio_path = Some(path.to_string());
        }
        if let Some(text) = ref_text {
            self.tts_ref_text = Some(text.to_string());
        }
        let _ = self.save();
    }

    /// 更新并保存 SRT 设置
    pub fn update_srt_settings(
        &mut self,
        model_path: Option<&str>,
        min_segment_length: Option<i32>,
        max_segment_length: Option<i32>,
    ) {
        if let Some(path) = model_path {
            self.srt_model_path = Some(path.to_string());
        }
        self.srt_min_segment_length = min_segment_length;
        self.srt_max_segment_length = max_segment_length;
        let _ = self.save();
    }

    /// 更新并保存 Demucs 设置
    pub fn update_demucs_settings(
        &mut self,
        model_path: Option<&str>,
        output_dir: Option<&str>,
        separation_mode: Option<&str>,
    ) {
        if let Some(path) = model_path {
            self.demucs_model_path = Some(path.to_string());
        }
        if let Some(dir) = output_dir {
            self.demucs_output_dir = Some(dir.to_string());
        }
        if let Some(mode) = separation_mode {
            self.demucs_separation_mode = Some(mode.to_string());
        }
        let _ = self.save();
    }

    /// 更新并保存搜索设置
    pub fn update_search_settings(
        &mut self,
        books_dir: Option<&str>,
        search_mode: Option<&str>,
    ) {
        if let Some(dir) = books_dir {
            self.books_dir = Some(dir.to_string());
        }
        if let Some(mode) = search_mode {
            self.search_mode = Some(mode.to_string());
        }
        let _ = self.save();
    }

    /// 更新所有设置
    pub fn update_all_settings(
        &mut self,
        ocr_model_path: Option<&str>,
        ocr_output_dir: Option<&str>,
        srt_model_path: Option<&str>,
        srt_min_segment_length: Option<i32>,
        srt_max_segment_length: Option<i32>,
        tts_model_path: Option<&str>,
        tts_output_path: Option<&str>,
        tts_ref_audio_path: Option<&str>,
        tts_ref_text: Option<&str>,
        demucs_model_path: Option<&str>,
        demucs_output_dir: Option<&str>,
        demucs_separation_mode: Option<&str>,
        window_width: Option<i32>,
        window_height: Option<i32>,
    ) {
        if let Some(path) = ocr_model_path {
            self.ocr_model_path = Some(path.to_string());
        }
        if let Some(dir) = ocr_output_dir {
            self.ocr_output_dir = Some(dir.to_string());
        }
        if let Some(path) = srt_model_path {
            self.srt_model_path = Some(path.to_string());
        }
        self.srt_min_segment_length = srt_min_segment_length;
        self.srt_max_segment_length = srt_max_segment_length;
        if let Some(path) = tts_model_path {
            self.tts_model_path = Some(path.to_string());
        }
        if let Some(path) = tts_output_path {
            self.tts_output_path = Some(path.to_string());
        }
        if let Some(path) = tts_ref_audio_path {
            self.tts_ref_audio_path = Some(path.to_string());
        }
        if let Some(text) = tts_ref_text {
            self.tts_ref_text = Some(text.to_string());
        }
        if let Some(path) = demucs_model_path {
            self.demucs_model_path = Some(path.to_string());
        }
        if let Some(dir) = demucs_output_dir {
            self.demucs_output_dir = Some(dir.to_string());
        }
        if let Some(mode) = demucs_separation_mode {
            self.demucs_separation_mode = Some(mode.to_string());
        }
        self.window_width = window_width;
        self.window_height = window_height;
        let _ = self.save();
    }
}
