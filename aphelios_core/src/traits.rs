use crate::utils::progress::AppProgressBar;
use anyhow::Result;

/// OCR 引擎接口
pub trait OcrEngine: Send + Sync {
    /// 执行 OCR 识别
    ///
    /// # 参数
    /// * `model_path` - 模型路径
    /// * `input_path` - 输入文件路径（图片或 PDF）
    /// * `output_dir` - 输出目录
    fn dolphin_ocr(
        &mut self,
        model_path: &str,
        input_path: &str,
        output_dir: &str,
    ) -> Result<Vec<String>>;
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

/// 索引状态信息
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IndexStatus {
    /// 索引是否存在
    pub exists: bool,
    /// 已索引书籍数量
    pub book_count: usize,
    /// 索引创建时间
    pub created_at: Option<String>,
    /// 索引最后更新时间
    pub updated_at: Option<String>,
    /// 语义索引是否存在
    pub semantic_exists: bool,
}

/// 搜索模式
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SearchMode {
    /// FTS5 关键字前缀搜索（默认）
    #[default]
    Keyword,
    /// 基于 Harrier 嵌入的向量相似度搜索
    Semantic,
    /// 混合搜索：结合关键字和语义搜索
    Hybrid,
}

/// 搜索引擎接口
pub trait SearchEngine: Send + Sync {
    /// 获取当前书籍总数
    fn get_book_count(&self) -> Result<usize>;

    /// 获取索引状态信息
    fn get_index_status(&self) -> Result<IndexStatus>;

    /// 构建/更新完整索引（全文搜索 + 语义向量索引）
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

    /// 执行书籍搜索（指定模式）
    ///
    /// # 参数
    /// * `query` - 搜索词
    /// * `limit` - 结果数量限制
    /// * `mode` - 搜索模式
    fn search_books_with_mode(
        &self,
        query: &str,
        limit: usize,
        mode: SearchMode,
    ) -> Result<SearchResult>;
}
