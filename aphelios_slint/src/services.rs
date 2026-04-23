use anyhow::Result;
use aphelios_core::traits::{
    BookInfo, OcrEngine, SearchEngine, SearchMode, SearchResult, TtsEngine,
};
use aphelios_core::utils::progress::AppProgressBar;
use aphelios_ocr::dolphin::model::DolphinModel;
use aphelios_search as search;
use aphelios_tts::qwen_tts::{generate_voice, generate_voice_batch_from_txt};

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

    fn generate_batch(
        &self,
        model_path: &str,
        ref_audio_path: &str,
        ref_text: &str,
        txt_file_path: &str,
        output_dir: &str,
        batch_size: usize,
        progress: Option<AppProgressBar>,
    ) -> Result<Vec<String>> {
        generate_voice_batch_from_txt(
            model_path,
            ref_audio_path,
            ref_text,
            txt_file_path,
            output_dir,
            batch_size,
            progress,
        )
    }
}

/// 搜索服务的 Sqlite 实现
pub struct SqliteSearchClient {
    config: search::IndexConfig,
}

impl SqliteSearchClient {
    pub fn new(book_dir: &str) -> Self {
        Self {
            config: search::IndexConfig::from_book_dir(book_dir),
        }
    }

    pub fn with_config(config: search::IndexConfig) -> Self {
        Self { config }
    }
}

impl SearchEngine for SqliteSearchClient {
    fn get_book_count(&self) -> Result<usize> {
        search::get_book_count(&self.config)
    }

    fn get_index_status(&self) -> Result<aphelios_core::traits::IndexStatus> {
        let status = search::get_index_status(&self.config)?;
        Ok(aphelios_core::traits::IndexStatus {
            exists: status.exists,
            book_count: status.book_count,
            created_at: status.created_at,
            updated_at: status.updated_at,
            semantic_exists: status.semantic_exists,
        })
    }

    fn build_index(&self, progress_callback: Option<Box<dyn Fn(usize) + Send>>) -> Result<usize> {
        match progress_callback {
            Some(cb) => search::build_index(&self.config, Some(&|p| cb(p))),
            None => search::build_index(&self.config, None),
        }
    }

    fn search_books(&self, query: &str, limit: usize) -> Result<SearchResult> {
        let result = search::search_books(&self.config, query, limit)?;

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

    fn search_books_with_mode(
        &self,
        query: &str,
        limit: usize,
        mode: SearchMode,
    ) -> Result<SearchResult> {
        let search_mode = match mode {
            SearchMode::Keyword => search::SearchMode::Keyword,
            SearchMode::Semantic => search::SearchMode::Semantic,
            SearchMode::Hybrid => search::SearchMode::Hybrid,
        };

        let options = search::SearchOptions {
            query: query.to_string(),
            limit,
            mode: search_mode,
            normalize_chinese: true,
            config: self.config.clone(),
        };

        let result = search::search_books_with_options(&options)?;

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
