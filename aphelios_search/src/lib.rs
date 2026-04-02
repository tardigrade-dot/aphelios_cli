pub mod harrier_embed;

use anyhow::Result;
use glob::glob;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tracing::{error, info};

const BOOKS_DIR: &str = "/Volumes/sw/books";
const DB_PATH: &str = "/Volumes/sw/books/books.db";

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

/// 初始化 SQLite 数据库
pub fn init_database() -> Result<Connection> {
    let conn = Connection::open(DB_PATH)?;

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

    // 创建全文搜索虚拟表
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS books_fts USING fts5(
            title,
            author,
            content='books',
            content_rowid='id'
        )",
        [],
    )?;

    // 创建触发器保持 FTS 同步
    conn.execute(
        "CREATE TRIGGER IF NOT EXISTS books_ai AFTER INSERT ON books BEGIN
            INSERT INTO books_fts(rowid, title, author) VALUES (new.id, new.title, new.author);
        END",
        [],
    )?;

    conn.execute(
        "CREATE TRIGGER IF NOT EXISTS books_ad AFTER DELETE ON books BEGIN
            INSERT INTO books_fts(books_fts, rowid, title, author) VALUES('delete', old.id, old.title, old.author);
        END",
        [],
    )?;

    conn.execute(
        "CREATE TRIGGER IF NOT EXISTS books_au AFTER UPDATE ON books BEGIN
            INSERT INTO books_fts(books_fts, rowid, title, author) VALUES('delete', old.id, old.title, old.author);
            INSERT INTO books_fts(rowid, title, author) VALUES (new.id, new.title, new.author);
        END",
        [],
    )?;

    Ok(conn)
}

/// 从文件路径提取书名和作者
fn extract_metadata(file_path: &Path) -> (String, Option<String>) {
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
                return (title, author);
            }
        }
        (filename, None)
    };

    (title, author)
}

/// 扫描书籍目录并建立索引
pub fn build_index(progress_callback: Option<&dyn Fn(usize)>) -> Result<usize> {
    let conn = init_database()?;

    // 清空旧数据
    conn.execute("DELETE FROM books", [])?;
    conn.execute("DELETE FROM books_fts", [])?;

    let pattern = format!("{}/**/*", BOOKS_DIR);
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

    for (idx, path) in paths.iter().enumerate() {
        if let Some(callback) = progress_callback {
            callback(idx * 100 / total);
        }

        let file_path = path.to_string_lossy().to_string();
        let file_type = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        let file_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0) as i64;
        let (title, author) = extract_metadata(path);

        if let Err(e) = conn.execute(
            "INSERT INTO books (title, author, file_path, file_type, file_size) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![title, author, file_path, file_type, file_size],
        ) {
            error!("Failed to insert book {}: {}", file_path, e);
        }
    }

    if let Some(callback) = progress_callback {
        callback(100);
    }

    info!("Indexed {} books successfully", total);
    Ok(total)
}

/// 搜索书籍
pub fn search_books(query: &str, limit: usize) -> Result<SearchResult> {
    let conn = init_database()?;

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
        let fts_query = format!("{}*", query.replace('"', "\"\""));
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

/// 获取书籍数量
pub fn get_book_count() -> Result<usize> {
    let conn = init_database()?;
    let count: i64 = conn.query_row("SELECT COUNT(*) FROM books", [], |row| row.get(0))?;
    Ok(count as usize)
}

/// 获取书籍详情
pub fn get_book(id: i64) -> Result<Option<BookInfo>> {
    let conn = init_database()?;
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
pub fn delete_book(id: i64) -> Result<()> {
    let conn = init_database()?;
    conn.execute("DELETE FROM books WHERE id = ?1", [id])?;
    Ok(())
}
