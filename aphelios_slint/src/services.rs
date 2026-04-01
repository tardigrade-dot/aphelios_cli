use anyhow::Result;
use aphelios_core::traits::{BookInfo, OcrEngine, SearchEngine, SearchResult, TtsEngine};
use aphelios_core::utils::progress::AppProgressBar;
use aphelios_ocr::dolphin::model::DolphinModel;
use aphelios_search as search;
use aphelios_tts::qwen_tts::qwen_tts::generate_voice;

/// OCR 服务的 Dolphin 实现
pub struct DolphinOcrClient;

impl OcrEngine for DolphinOcrClient {
    fn dolphin_ocr(
        &mut self,
        model_path: &str,
        input_path: &str,
        output_dir: &str,
    ) -> Result<Vec<String>> {
        let mut dolphin_model = DolphinModel::load_model(model_path)?;
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(dolphin_model.dolphin_ocr(input_path, output_dir))
    }
}

/// TTS 服务的 Qwen 实现
pub struct QwenTtsClient;

impl TtsEngine for QwenTtsClient {
    fn generate(
        &self,
        model_path: &str,
        ref_audio_path: &str,
        ref_text: &str,
        input_text: &str,
        output_path: &str,
        progress: Option<AppProgressBar>,
    ) -> Result<()> {
        generate_voice(
            model_path,
            ref_audio_path,
            ref_text,
            input_text,
            output_path,
            progress,
        )
    }
}

/// 搜索服务的 Sqlite 实现
pub struct SqliteSearchClient;

impl SearchEngine for SqliteSearchClient {
    fn get_book_count(&self) -> Result<usize> {
        search::get_book_count()
    }

    fn build_index(&self, progress_callback: Option<Box<dyn Fn(usize) + Send>>) -> Result<usize> {
        match progress_callback {
            Some(cb) => search::build_index(Some(&|p| cb(p))),
            None => search::build_index(None),
        }
    }

    fn search_books(&self, query: &str, limit: usize) -> Result<SearchResult> {
        let result = search::search_books(query, limit)?;

        Ok(SearchResult {
            books: result
                .books
                .iter()
                .map(|b| BookInfo {
                    id: b.id,
                    title: b.title.clone(),
                    author: b.author.clone(),
                    file_path: b.file_path.clone(),
                    file_type: b.file_type.clone(),
                    file_size: b.file_size,
                })
                .collect(),
            total: result.total,
        })
    }
}
