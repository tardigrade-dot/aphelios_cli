use anyhow::Result;
use aphelios_core::traits::{
    BookInfo, IndexStatus, OcrEngine, SearchEngine, SearchMode, SearchResult, TtsEngine,
};
use aphelios_core::utils::progress::AppProgressBar;
use aphelios_ocr::dolphin::model::DolphinModel;
use aphelios_search as search;
use aphelios_tts::qwen_tts::qwen_tts_infer::{generate_voice, generate_voice_batch_from_txt};
use std::sync::{Arc, RwLock};

/// OCR 服务的 Dolphin 实现
pub struct DolphinOcrClient;

impl OcrEngine for DolphinOcrClient {
    fn dolphin_ocr(
        &mut self,
        model_path: &str,
        input_path: &str,
        output_dir: &str,
        progress: Option<AppProgressBar>,
    ) -> Result<Vec<String>> {
        let mut dolphin_model = DolphinModel::load_model(model_path)?;
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(dolphin_model.dolphin_ocr(input_path, output_dir, progress))
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

/// 内存搜索服务的实现。
///
/// 扫描书籍目录后全部加载到内存中，通过子串匹配（含简繁转换）进行搜索。
pub struct InMemorySearchClient {
    books: Arc<RwLock<Vec<search::BookInfo>>>,
    book_dir: Arc<RwLock<String>>,
}

impl InMemorySearchClient {
    pub fn new(book_dir: &str) -> Self {
        let client = Self {
            books: Arc::new(RwLock::new(Vec::new())),
            book_dir: Arc::new(RwLock::new(book_dir.to_string())),
        };
        // Pre-scan on creation
        let _ = client.rescan();
        client
    }

    /// Update the books directory and rescan.
    pub fn set_book_dir(&self, book_dir: &str) {
        if let Ok(mut dir) = self.book_dir.write() {
            *dir = book_dir.to_string();
        }
        let _ = self.rescan();
    }

    /// Scan (or re-scan) the books directory.
    fn rescan(&self) -> Result<usize> {
        let dir = self.book_dir.read().unwrap().clone();
        let books = search::scan_books(&dir)?;
        let count = books.len();
        if let Ok(mut cache) = self.books.write() {
            *cache = books;
        }
        Ok(count)
    }
}

impl SearchEngine for InMemorySearchClient {
    fn get_book_count(&self) -> Result<usize> {
        Ok(self.books.read().unwrap().len())
    }

    fn get_index_status(&self) -> Result<IndexStatus> {
        let count = self.books.read().unwrap().len();
        Ok(IndexStatus {
            exists: count > 0,
            book_count: count,
            created_at: None,
            updated_at: None,
            semantic_exists: false,
        })
    }

    fn build_index(&self, _progress_callback: Option<Box<dyn Fn(usize) + Send>>) -> Result<usize> {
        self.rescan()
    }

    fn search_books(&self, query: &str, limit: usize) -> Result<SearchResult> {
        let books = self.books.read().unwrap();
        let result = search::search_books(&books, query, limit);
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
        _mode: SearchMode,
    ) -> Result<SearchResult> {
        // Only keyword search is supported; mode is ignored.
        self.search_books(query, limit)
    }
}
