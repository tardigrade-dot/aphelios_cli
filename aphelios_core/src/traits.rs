use anyhow::Result;
use crate::utils::progress::AppProgressBar;

/// OCR 引擎接口
pub trait OcrEngine: Send + Sync {
    /// 执行 OCR 识别
    ///
    /// # 参数
    /// * `model_path` - 模型路径
    /// * `input_path` - 输入文件路径（图片或 PDF）
    /// * `output_dir` - 输出目录
    fn dolphin_ocr(&mut self, model_path: &str, input_path: &str, output_dir: &str) -> Result<Vec<String>>;
}

/// TTS 引擎接口
pub trait TtsEngine: Send + Sync {
    /// 执行 TTS 语音合成
    ///
    /// # 参数
    /// * `model_path` - 模型路径
    /// * `ref_audio_path` - 参考音频路径（用于克隆）
    /// * `ref_text` - 参考文本
    /// * `input_text` - 待合成文本
    /// * `output_path` - 输出音频路径
    /// * `progress` - 进度回调
    fn generate(
        &self,
        model_path: &str,
        ref_audio_path: &str,
        ref_text: &str,
        input_text: &str,
        output_path: &str,
        progress: Option<AppProgressBar>,
    ) -> Result<()>;
}

/// 搜索结果中单本书的信息
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BookInfo {
    pub id: i64,
    pub title: String,
    pub author: Option<String>,
    pub file_path: String,
    pub file_type: String,
    pub file_size: u64,
}

/// 搜索结果集
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SearchResult {
    pub books: Vec<BookInfo>,
    pub total: usize,
}

/// 搜索引擎接口
pub trait SearchEngine: Send + Sync {
    /// 获取当前书籍总数
    fn get_book_count(&self) -> Result<usize>;

    /// 构建/更新全文搜索索引
    ///
    /// # 参数
    /// * `progress_callback` - 进度回调（0-100）
    fn build_index(&self, progress_callback: Option<Box<dyn Fn(usize) + Send>>) -> Result<usize>;

    /// 执行书籍搜索
    ///
    /// # 参数
    /// * `query` - 搜索词
    /// * `limit` - 结果数量限制
    fn search_books(&self, query: &str, limit: usize) -> Result<SearchResult>;
}
