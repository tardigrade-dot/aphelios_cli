pub mod chinese_norm;
pub mod harrier_embed;
pub mod semantic_search;

use anyhow::Result;
use glob::glob;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tracing::{error, info};

/// Default Harrier model directory
pub const HARRIER_MODEL_DIR: &str = "/Volumes/sw/pretrained_models/harrier-oss-v1-0.6b";

/// Configuration for index building
#[derive(Debug, Clone)]
pub struct IndexConfig {
    /// Books directory (where to scan books from)
    pub book_dir: String,
    /// Database file path (default: {book_dir}/books.db)
    pub db_path: String,
    /// Semantic index file path (default: {book_dir}/embeddings.usearch)
    pub semantic_index_path: String,
    /// Harrier model directory
    pub model_dir: String,
}

impl IndexConfig {
    /// Create config with default paths based on book_dir
    pub fn from_book_dir(book_dir: &str) -> Self {
        Self {
            book_dir: book_dir.to_string(),
            db_path: format!("{}/books.db", book_dir),
            semantic_index_path: format!("{}/embeddings.usearch", book_dir),
            model_dir: HARRIER_MODEL_DIR.to_string(),
        }
    }

    /// Create config with all custom paths
    pub fn new(book_dir: &str, db_path: &str, semantic_index_path: &str, model_dir: &str) -> Self {
        Self {
            book_dir: book_dir.to_string(),
            db_path: db_path.to_string(),
            semantic_index_path: semantic_index_path.to_string(),
            model_dir: model_dir.to_string(),
        }
    }
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self::from_book_dir("/Volumes/sw/books")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BookInfo {
    pub id: i64,
    pub title: String,
    pub author: Option<String>,
    pub file_path: String,
    pub file_type: String,
    pub file_size: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub books: Vec<BookInfo>,
    pub total: usize,
}

/// Index status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStatus {
    /// Whether index exists
    pub exists: bool,
    /// Number of books indexed
    pub book_count: usize,
    /// Index creation timestamp (ISO format)
    pub created_at: Option<String>,
    /// Last index update timestamp (ISO format)
    pub updated_at: Option<String>,
    /// Semantic index exists
    pub semantic_exists: bool,
}

/// Search mode for hybrid search
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SearchMode {
    /// FTS5 keyword prefix search (default)
    #[default]
    Keyword,
    /// Vector similarity search using Harrier embeddings
    Semantic,
    /// Combine keyword and semantic search with RRF fusion
    Hybrid,
}

/// Search options
#[derive(Debug, Clone)]
pub struct SearchOptions {
    pub query: String,
    pub limit: usize,
    pub mode: SearchMode,
    /// Enable Chinese normalization (S2T/T2S). Default: true
    pub normalize_chinese: bool,
    /// Index configuration (database path, semantic index path, etc.)
    pub config: IndexConfig,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            query: String::new(),
            limit: 20,
            mode: SearchMode::Keyword,
            normalize_chinese: true,
            config: IndexConfig::default(),
        }
    }
}

/// 初始化 SQLite 数据库
pub fn init_database(db_path: &str) -> Result<Connection> {
    // Ensure directory exists
    if let Some(parent) = Path::new(db_path).parent() {
        std::fs::create_dir_all(parent).ok();
    }

    let conn = Connection::open(db_path)?;

    conn.execute(
        "CREATE TABLE IF NOT EXISTS books (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            author TEXT,
            file_path TEXT NOT NULL UNIQUE,
            file_type TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )",
        [],
    )?;

    // 添加简繁标准化列（如果不存在）
    conn.execute("ALTER TABLE books ADD COLUMN title_norm TEXT", [])
        .ok(); // Ignore error if column exists
    conn.execute("ALTER TABLE books ADD COLUMN author_norm TEXT", [])
        .ok(); // Ignore error if column exists

    // 创建全文搜索虚拟表（使用原始文本和标准化文本）
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS books_fts USING fts5(
            title,
            author,
            title_norm,
            author_norm,
            content='books',
            content_rowid='id'
        )",
        [],
    )?;

    // 创建触发器保持 FTS 同步
    conn.execute(
        "CREATE TRIGGER IF NOT EXISTS books_ai AFTER INSERT ON books BEGIN
            INSERT INTO books_fts(rowid, title, author, title_norm, author_norm)
            VALUES (new.id, new.title, new.author, new.title_norm, new.author_norm);
        END",
        [],
    )?;

    conn.execute(
        "CREATE TRIGGER IF NOT EXISTS books_ad AFTER DELETE ON books BEGIN
            INSERT INTO books_fts(books_fts, rowid, title, author, title_norm, author_norm)
            VALUES('delete', old.id, old.title, old.author, old.title_norm, old.author_norm);
        END",
        [],
    )?;

    conn.execute(
        "CREATE TRIGGER IF NOT EXISTS books_au AFTER UPDATE ON books BEGIN
            INSERT INTO books_fts(books_fts, rowid, title, author, title_norm, author_norm)
            VALUES('delete', old.id, old.title, old.author, old.title_norm, old.author_norm);
            INSERT INTO books_fts(rowid, title, author, title_norm, author_norm)
            VALUES (new.id, new.title, new.author, new.title_norm, new.author_norm);
        END",
        [],
    )?;

    Ok(conn)
}

/// 从文件路径提取书名和作者（原始文本 + 标准化文本）
/// Returns: (title, author, title_norm, author_norm)
fn extract_metadata(file_path: &Path) -> (String, Option<String>, String, Option<String>) {
    let filename = file_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("未知书名")
        .to_string();

    // 尝试解析常见格式: "书名 (作者)" 或 "书名 - 作者"
    let (title, author) = if let Some(paren_pos) = filename.find('(') {
        let title = filename[..paren_pos].trim().to_string();
        let author = filename[paren_pos + 1..].find(')').map(|end| {
            filename[paren_pos + 1..paren_pos + 1 + end]
                .trim()
                .to_string()
        });
        (title, author)
    } else if let Some(dash_pos) = filename.find(" - ") {
        let title = filename[..dash_pos].trim().to_string();
        let author = Some(filename[dash_pos + 3..].trim().to_string());
        (title, author)
    } else {
        // 尝试解析 [作者] 书名 格式
        if let (Some(start), Some(end)) = (filename.find('['), filename.find(']')) {
            let author = Some(filename[start + 1..end].trim().to_string());
            let title = filename[end + 1..].trim().to_string();
            if !title.is_empty() {
                let title_norm = chinese_norm::normalize(&title);
                let author_norm = author.as_ref().map(|a| chinese_norm::normalize(a));
                return (title, author, title_norm, author_norm);
            }
        }
        (filename.clone(), None)
    };

    // 生成标准化文本
    let title_norm = chinese_norm::normalize(&title);
    let author_norm = author.as_ref().map(|a| chinese_norm::normalize(a));

    (title, author, title_norm, author_norm)
}

/// 扫描书籍目录并建立完整索引（SQLite FTS + 语义向量索引）
///
/// 这是唯一的索引构建入口，会自动完成两个过程：
/// 1. 构建 SQLite 全文搜索索引（BM25/FTS5）
/// 2. 构建语义向量索引（usearch + Harrier embeddings）
///
/// # 参数
/// * `config` - 索引配置（书籍目录、数据库路径、语义索引路径等）
/// * `progress_callback` - 进度回调（0-100）
///   - 0-50%: SQLite 全文索引构建阶段
///   - 50-100%: 语义向量索引构建阶段
pub fn build_index(
    config: &IndexConfig,
    progress_callback: Option<&dyn Fn(usize)>,
) -> Result<usize> {
    rebuild_search_index(config, progress_callback)
}

/// 内部实现：统一构建搜索索引（SQLite FTS + usearch 语义索引）
fn rebuild_search_index(
    config: &IndexConfig,
    progress_callback: Option<&dyn Fn(usize)>,
) -> Result<usize> {
    let conn = init_database(&config.db_path)?;

    // 清空旧数据
    conn.execute("DELETE FROM books", [])?;
    conn.execute("DELETE FROM books_fts", [])?;

    // 删除旧的语义索引及其元数据
    let _ = std::fs::remove_file(&config.semantic_index_path);
    let _ = std::fs::remove_file(format!("{}.metadata", config.semantic_index_path));

    // 扫描书籍目录
    let pattern = format!("{}/**/*", config.book_dir);
    let paths: Vec<PathBuf> = glob(&pattern)?
        .filter_map(|e| e.ok())
        .filter(|p| {
            if let Some(ext) = p.extension() {
                let ext = ext.to_str().unwrap_or("");
                matches!(
                    ext.to_lowercase().as_str(),
                    "pdf" | "epub" | "txt" | "mobi" | "azw3"
                )
            } else {
                false
            }
        })
        .collect();

    let total = paths.len();
    info!("Found {} books to index", total);

    if total == 0 {
        return Ok(0);
    }

    // 第一阶段：构建 SQLite 索引
    let mut books_for_semantic: Vec<(i64, String)> = Vec::new();

    for (idx, path) in paths.iter().enumerate() {
        if let Some(callback) = progress_callback {
            callback(idx * 50 / total); // 0-50% for SQLite
        }

        let file_path = path.to_string_lossy().to_string();
        let file_type = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        let file_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0) as i64;
        let (title, author, title_norm, author_norm) = extract_metadata(path);

        if let Err(e) = conn.execute(
            "INSERT INTO books (title, author, title_norm, author_norm, file_path, file_type, file_size) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![title, author, title_norm, author_norm, file_path, file_type, file_size],
        ) {
            error!("Failed to insert book {}: {}", file_path, e);
            continue;
        }

        // 获取刚插入的 book id
        let book_id: i64 = conn.query_row("SELECT last_insert_rowid()", [], |row| row.get(0))?;
        books_for_semantic.push((book_id, title));
    }

    // 第二阶段：批量构建语义向量索引（使用 SQLite 中的真实 book_id）
    let semantic_total = books_for_semantic.len();
    let mut semantic_index =
        semantic_search::SemanticIndex::create(&config.model_dir, &config.semantic_index_path)?;

    // 批量处理：每 10 条调用一次模型
    const BATCH_SIZE: usize = 10;
    for chunk_start in (0..semantic_total).step_by(BATCH_SIZE) {
        let chunk_end = (chunk_start + BATCH_SIZE).min(semantic_total);
        let chunk = &books_for_semantic[chunk_start..chunk_end];

        let books_refs: Vec<(i64, &str)> = chunk
            .iter()
            .map(|(id, title)| (*id, title.as_str()))
            .collect();

        semantic_index.add_books(&books_refs)?;

        // 更新进度
        if let Some(callback) = progress_callback {
            callback(50 + chunk_end * 50 / semantic_total); // 50-100% for semantic
        }
    }

    // 保存语义索引
    semantic_index.save(&config.semantic_index_path)?;

    if let Some(callback) = progress_callback {
        callback(100);
    }

    info!(
        "Built search index for {} books (SQLite + usearch)",
        books_for_semantic.len()
    );
    Ok(books_for_semantic.len())
}

/// 搜索书籍
pub fn search_books(config: &IndexConfig, query: &str, limit: usize) -> Result<SearchResult> {
    search_books_internal(config, query, limit, SearchMode::Keyword, true)
}

/// 通用搜索接口，支持多种搜索模式
pub fn search_books_with_options(options: &SearchOptions) -> Result<SearchResult> {
    search_books_internal(
        &options.config,
        &options.query,
        options.limit,
        options.mode,
        options.normalize_chinese,
    )
}

fn search_books_internal(
    config: &IndexConfig,
    query: &str,
    limit: usize,
    mode: SearchMode,
    normalize_chinese: bool,
) -> Result<SearchResult> {
    match mode {
        SearchMode::Keyword | SearchMode::Hybrid => {
            // Use FTS5 search
            let fts_result = fts_search(config, query, limit, normalize_chinese)?;

            if mode == SearchMode::Keyword {
                return Ok(fts_result);
            }

            // Hybrid mode: combine FTS with semantic search
            let semantic_result = semantic_search_internal(config, query, limit)?;

            // RRF fusion
            Ok(hybrid_fusion(config, fts_result, semantic_result, limit))
        }
        SearchMode::Semantic => semantic_search_internal(config, query, limit),
    }
}

/// FTS5 keyword search with optional Chinese normalization
fn fts_search(
    config: &IndexConfig,
    query: &str,
    limit: usize,
    normalize_chinese: bool,
) -> Result<SearchResult> {
    let conn = init_database(&config.db_path)?;

    let sql = if query.is_empty() {
        "SELECT id, title, author, file_path, file_type, file_size FROM books ORDER BY title LIMIT ?1"
    } else {
        "SELECT b.id, b.title, b.author, b.file_path, b.file_type, b.file_size
         FROM books b
         INNER JOIN books_fts f ON b.id = f.rowid
         WHERE books_fts MATCH ?1
         ORDER BY rank
         LIMIT ?2"
    };

    let mut stmt = conn.prepare(sql)?;
    let limit_i32 = limit as i32;

    let books: Vec<BookInfo> = if query.is_empty() {
        stmt.query_map(params![limit_i32], |row| {
            Ok(BookInfo {
                id: row.get(0)?,
                title: row.get(1)?,
                author: row.get(2)?,
                file_path: row.get(3)?,
                file_type: row.get(4)?,
                file_size: row.get::<_, i64>(5)? as u64,
            })
        })?
        .filter_map(|r| r.ok())
        .collect()
    } else {
        // 处理搜索查询
        let fts_query = if normalize_chinese {
            // Generate query with both original and normalized forms
            let query_norm = chinese_norm::normalize(query);
            let query_trad = chinese_norm::to_traditional(query);
            format!(
                "({}* OR {}* OR {}*)",
                query.replace('"', "\"\""),
                query_norm.replace('"', "\"\""),
                query_trad.replace('"', "\"\"")
            )
        } else {
            format!("{}*", query.replace('"', "\"\""))
        };

        stmt.query_map(params![fts_query.as_str(), limit_i32], |row| {
            Ok(BookInfo {
                id: row.get(0)?,
                title: row.get(1)?,
                author: row.get(2)?,
                file_path: row.get(3)?,
                file_type: row.get(4)?,
                file_size: row.get::<_, i64>(5)? as u64,
            })
        })?
        .filter_map(|r| r.ok())
        .collect()
    };

    let total = books.len();
    Ok(SearchResult { books, total })
}

/// Semantic search using Harrier embeddings
fn semantic_search_internal(
    config: &IndexConfig,
    query: &str,
    limit: usize,
) -> Result<SearchResult> {
    if query.is_empty() {
        return Ok(SearchResult {
            books: Vec::new(),
            total: 0,
        });
    }

    let conn = init_database(&config.db_path)?;

    // Try to load existing semantic index
    let index_path = &config.semantic_index_path;
    if !std::path::Path::new(index_path).exists() {
        info!(
            "Semantic index not found at {}, returning empty results",
            index_path
        );
        return Ok(SearchResult {
            books: Vec::new(),
            total: 0,
        });
    }

    let mut semantic_index =
        match semantic_search::SemanticIndex::load(&config.model_dir, index_path) {
            Ok(idx) => idx,
            Err(e) => {
                info!(
                    "Failed to load semantic index: {}, returning empty results",
                    e
                );
                return Ok(SearchResult {
                    books: Vec::new(),
                    total: 0,
                });
            }
        };

    // Search
    let results = semantic_index.search(query, limit)?;

    // Fetch book info for results
    let mut books = Vec::new();
    for (book_id, _score) in results {
        if let Ok(Some(book)) = get_book_internal(&conn, book_id) {
            books.push(book);
        }
    }

    let total = books.len();
    Ok(SearchResult { books, total })
}

/// Helper to get book without re-initializing connection
fn get_book_internal(conn: &Connection, id: i64) -> Result<Option<BookInfo>> {
    let result = conn.query_row(
        "SELECT id, title, author, file_path, file_type, file_size FROM books WHERE id = ?1",
        [id],
        |row| {
            Ok(BookInfo {
                id: row.get(0)?,
                title: row.get(1)?,
                author: row.get(2)?,
                file_path: row.get(3)?,
                file_type: row.get(4)?,
                file_size: row.get::<_, i64>(5)? as u64,
            })
        },
    );

    match result {
        Ok(book) => Ok(Some(book)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

/// Reciprocal Rank Fusion for combining search results
fn hybrid_fusion(
    config: &IndexConfig,
    fts_result: SearchResult,
    semantic_result: SearchResult,
    limit: usize,
) -> SearchResult {
    const RRF_K: f32 = 60.0; // RRF constant

    // Build score map from FTS results
    let mut scores: std::collections::HashMap<i64, f32> = std::collections::HashMap::new();

    for (rank, book) in fts_result.books.iter().enumerate() {
        let score = 1.0 / (RRF_K + (rank + 1) as f32);
        scores.insert(book.id, score);
    }

    // Add semantic scores
    for (rank, book) in semantic_result.books.iter().enumerate() {
        let score = 1.0 / (RRF_K + (rank + 1) as f32);
        *scores.entry(book.id).or_insert(0.0) += score;
    }

    // Sort by combined score
    let mut book_ids: Vec<i64> = scores.keys().cloned().collect();
    book_ids.sort_by(|a, b| {
        scores
            .get(b)
            .unwrap_or(&0.0)
            .partial_cmp(scores.get(a).unwrap_or(&0.0))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Fetch full book info
    let conn = match init_database(&config.db_path) {
        Ok(c) => c,
        Err(_) => {
            return SearchResult {
                books: Vec::new(),
                total: 0,
            }
        }
    };

    let books: Vec<BookInfo> = book_ids
        .into_iter()
        .take(limit)
        .filter_map(|id| get_book_internal(&conn, id).ok().flatten())
        .collect();

    SearchResult {
        total: books.len(),
        books,
    }
}

/// 获取书籍数量
pub fn get_book_count(config: &IndexConfig) -> Result<usize> {
    let conn = init_database(&config.db_path)?;
    let count: i64 = conn.query_row("SELECT COUNT(*) FROM books", [], |row| row.get(0))?;
    Ok(count as usize)
}

/// 获取索引状态信息
pub fn get_index_status(config: &IndexConfig) -> Result<IndexStatus> {
    let db_path = &config.db_path;
    let semantic_path = &config.semantic_index_path;

    // Check if database exists and has data
    let db_exists = Path::new(db_path).exists();
    let mut book_count = 0;
    let mut created_at: Option<String> = None;
    let mut updated_at: Option<String> = None;

    if db_exists {
        if let Ok(conn) = Connection::open(db_path) {
            // Get book count
            book_count = conn
                .query_row("SELECT COUNT(*) FROM books", [], |row| row.get::<_, i64>(0))
                .unwrap_or(0) as usize;

            // Get earliest and latest indexed_at timestamps
            if let Ok(mut stmt) = conn.prepare(
                "SELECT MIN(indexed_at), MAX(indexed_at) FROM books WHERE indexed_at IS NOT NULL",
            ) {
                let result: Option<(Option<String>, Option<String>)> = stmt
                    .query_row([], |row| {
                        let min: Option<String> = row.get(0)?;
                        let max: Option<String> = row.get(1)?;
                        Ok((min, max))
                    })
                    .ok();

                if let Some((min_ts, max_ts)) = result {
                    created_at = min_ts;
                    updated_at = max_ts;
                }
            }
        }
    }

    // Check if semantic index exists
    let semantic_exists = Path::new(semantic_path).exists();

    Ok(IndexStatus {
        exists: db_exists && book_count > 0,
        book_count,
        created_at,
        updated_at,
        semantic_exists,
    })
}

/// 获取书籍详情
pub fn get_book(config: &IndexConfig, id: i64) -> Result<Option<BookInfo>> {
    let conn = init_database(&config.db_path)?;
    let result = conn.query_row(
        "SELECT id, title, author, file_path, file_type, file_size FROM books WHERE id = ?1",
        [id],
        |row| {
            Ok(BookInfo {
                id: row.get(0)?,
                title: row.get(1)?,
                author: row.get(2)?,
                file_path: row.get(3)?,
                file_type: row.get(4)?,
                file_size: row.get::<_, i64>(5)? as u64,
            })
        },
    );

    match result {
        Ok(book) => Ok(Some(book)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

/// 删除书籍记录
pub fn delete_book(config: &IndexConfig, id: i64) -> Result<()> {
    let conn = init_database(&config.db_path)?;
    conn.execute("DELETE FROM books WHERE id = ?1", [id])?;
    Ok(())
}

/// 获取所有书籍
pub fn get_all_books(config: &IndexConfig) -> Result<Vec<BookInfo>> {
    let conn = init_database(&config.db_path)?;
    let mut stmt =
        conn.prepare("SELECT id, title, author, file_path, file_type, file_size FROM books")?;

    let books: Vec<BookInfo> = stmt
        .query_map([], |row| {
            Ok(BookInfo {
                id: row.get(0)?,
                title: row.get(1)?,
                author: row.get(2)?,
                file_path: row.get(3)?,
                file_type: row.get(4)?,
                file_size: row.get::<_, i64>(5)? as u64,
            })
        })?
        .filter_map(|r| r.ok())
        .collect();

    Ok(books)
}
