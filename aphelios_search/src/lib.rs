//! 书籍搜索库
//!
//! 核心能力:
//! - 递归扫描目录, 按文件扩展名过滤出书籍文件
//! - 默认支持「简体 ↔ 繁体」互相检索: 查询词与文件名都会先归一化
//!   (转小写 → 转为简体 → 去掉空白与常见分隔/标点) 再做子串匹配
//! - 多关键字以空白分隔, 采用 AND 语义
//! - 支持按文件类型过滤 (`SearchOptions::extensions`)
//!
//! 使用示例:
//! ```no_run
//! use aphelios_search::{BookSearcher, SearchOptions};
//!
//! let searcher = BookSearcher::new();
//! let options = SearchOptions {
//!     dir: std::path::PathBuf::from("/path/to/books"),
//!     query: "深度学习".to_string(), // 可以匹配「深度學習」开头的书名
//!     ..Default::default()
//! };
//! if let Ok(hits) = searcher.search(&options) {
//!     for hit in hits {
//!         println!("{}", hit.file_name);
//!     }
//! }
//! ```

use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use anyhow::Result;

/// 默认支持的书籍文件扩展名(小写、不含点号)
pub const BOOK_EXTENSIONS: &[&str] = &["pdf", "epub", "txt", "mobi", "azw3", "docx", "djvu", "chm", "md", "rtf", "fb2"];

/// 递归扫描的最大深度, 防止符号链接/循环目录导致无限递归
const MAX_DEPTH: usize = 24;

/// 默认的搜索结果上限
pub const DEFAULT_LIMIT: usize = 1000;

/// 搜索参数
#[derive(Debug, Clone)]
pub struct SearchOptions {
    /// 要扫描的根目录
    pub dir: PathBuf,
    /// 关键字(简体/繁体均可), 多个关键字用空白分隔, AND 匹配
    pub query: String,
    /// 文件类型过滤: `Some([...])` 时只匹配这些扩展名(不区分大小写、可带点);
    /// `None` 或空列表时使用 [`BOOK_EXTENSIONS`]
    pub extensions: Option<Vec<String>>,
    /// 是否递归子目录
    pub recursive: bool,
    /// 返回结果上限
    pub limit: usize,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            dir: PathBuf::from("."),
            query: String::new(),
            extensions: None,
            recursive: true,
            limit: DEFAULT_LIMIT,
        }
    }
}

/// 一条书籍搜索结果
#[derive(Debug, Clone)]
pub struct BookHit {
    /// 完整路径
    pub path: PathBuf,
    /// 相对根目录的路径
    pub relative_path: String,
    /// 文件名(含扩展名)
    pub file_name: String,
    /// 文件名(不含扩展名)
    pub stem: String,
    /// 扩展名(小写, 不含点)
    pub extension: String,
    /// 文件大小(字节)
    pub size: u64,
    /// 最后修改时间
    pub modified: Option<SystemTime>,
}

impl BookHit {
    /// 人类可读的文件大小, 例如 "1.2 MB"
    pub fn size_human(&self) -> String {
        human_size(self.size)
    }
}

/// 书籍搜索器。
///
/// 无内部状态, 可随意克隆; 搜索为阻塞调用, 建议放到后台线程执行。
#[derive(Clone, Default)]
pub struct BookSearcher {}

impl BookSearcher {
    pub fn new() -> Self {
        Self::default()
    }

    /// 归一化文本: 转小写 → 转为简体 → 去掉空白与常见分隔/标点。
    ///
    /// 这样「繁體文件名」和「简体搜索词」(或反过来) 也能互相匹配,
    /// 例如: `三體` → `三体`, `机器-学习导论` → `机器学习导论`。
    pub fn normalize(&self, text: &str) -> String {
        let lower = text.to_lowercase();
        let simplified = zhconv::zhconv(&lower, zhconv::Variant::ZhCN);
        let mut out = String::with_capacity(simplified.len());
        for ch in simplified.chars() {
            if !ch.is_whitespace() && !is_separator(ch) {
                out.push(ch);
            }
        }
        out
    }

    /// 执行搜索(阻塞)。返回按匹配质量排序后的结果。
    pub fn search(&self, options: &SearchOptions) -> Result<Vec<BookHit>> {
        let dir = &options.dir;
        if !dir.is_dir() {
            anyhow::bail!("目录不存在: {}", dir.display());
        }

        let extensions: Vec<String> = match &options.extensions {
            Some(list) if !list.is_empty() => list
                .iter()
                .map(|s| s.trim_start_matches('.').to_lowercase())
                .collect(),
            _ => BOOK_EXTENSIONS
                .iter()
                .map(|s| s.to_string())
                .collect(),
        };

        let query = options.query.trim();
        let terms: Vec<String> = if query.is_empty() {
            Vec::new()
        } else {
            query
                .split_whitespace()
                .map(|t| self.normalize(t))
                .filter(|t| !t.is_empty())
                .collect()
        };

        // 用 (匹配质量, 命中) 暂存, 便于排序后丢弃质量字段
        let mut scored: Vec<(u8, BookHit)> = Vec::new();
        self.walk(dir, dir, options.recursive, 0, &extensions, &terms, &mut scored)?;

        scored.sort_by(|(qa, a), (qb, b)| {
            qa.cmp(qb).then_with(|| {
                a.file_name
                    .to_lowercase()
                    .cmp(&b.file_name.to_lowercase())
            })
        });

        let mut hits: Vec<BookHit> = scored
            .into_iter()
            .map(|(_, h)| h)
            .collect();
        hits.truncate(options.limit);
        Ok(hits)
    }

    /// 递归遍历目录, 收集匹配的书籍文件
    #[allow(clippy::too_many_arguments)]
    fn walk(&self, root: &Path, dir: &Path, recursive: bool, depth: usize, extensions: &[String], terms: &[String], out: &mut Vec<(u8, BookHit)>) -> Result<()> {
        if depth > MAX_DEPTH {
            return Ok(());
        }

        let entries = match fs::read_dir(dir) {
            Ok(entries) => entries,
            Err(err) => {
                tracing::debug!(path = %dir.display(), "跳过不可读目录: {err}");
                return Ok(());
            }
        };

        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry
                .file_name()
                .to_string_lossy()
                .to_string();
            if name.starts_with('.') {
                continue; // 跳过隐藏文件/目录
            }

            let file_type = match entry.file_type() {
                Ok(t) => t,
                Err(_) => continue,
            };

            if file_type.is_dir() {
                if recursive {
                    self.walk(root, &path, recursive, depth + 1, extensions, terms, out)?;
                }
                continue;
            }
            if !file_type.is_file() {
                continue;
            }

            let Some(ext) = path
                .extension()
                .map(|e| e.to_string_lossy().to_lowercase())
            else {
                continue;
            };
            if !extensions.iter().any(|e| *e == ext) {
                continue;
            }

            let stem = path
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default();
            let relative_path = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .to_string_lossy()
                .to_string();

            let stem_norm = self.normalize(&stem);
            let path_norm = self.normalize(&relative_path);
            if !matches_terms(terms, &stem_norm, &path_norm) {
                continue;
            }

            let meta = match fs::metadata(&path) {
                Ok(meta) => meta,
                Err(_) => continue,
            };

            let quality = match_quality(&terms.join(""), &stem_norm, &path_norm);
            out.push((
                quality,
                BookHit {
                    path,
                    relative_path,
                    file_name: name,
                    stem,
                    extension: ext,
                    size: meta.len(),
                    modified: meta.modified().ok(),
                },
            ));
        }
        Ok(())
    }
}

/// 关键字是否全部命中(AND 语义)。
/// 空关键字表示不限制(匹配所有)。
fn matches_terms(terms: &[String], stem: &str, path: &str) -> bool {
    if terms.is_empty() {
        return true;
    }
    terms
        .iter()
        .all(|t| stem.contains(t.as_str()) || path.contains(t.as_str()))
}

/// 匹配质量(数值越小越靠前):
/// 0 = 文件名(不含扩展名)与关键字完全相等
/// 1 = 文件名以关键字开头
/// 2 = 文件名包含关键字
/// 3 = 仅相对路径包含关键字
fn match_quality(query: &str, stem: &str, _path: &str) -> u8 {
    if query.is_empty() {
        return 0;
    }
    if stem == query {
        0
    } else if stem.starts_with(query) {
        1
    } else if stem.contains(query) {
        2
    } else {
        3 // 只有相对路径命中
    }
}

/// 常见的分隔/标点, 归一化时会被去掉(对查询词与文件名一视同仁)
fn is_separator(c: char) -> bool {
    matches!(
        c,
        '，' | '。'
            | '、'
            | '；'
            | '：'
            | '？'
            | '！'
            | '「'
            | '」'
            | '『'
            | '』'
            | '《'
            | '》'
            | '〈'
            | '〉'
            | '（'
            | '）'
            | '【'
            | '】'
            | '〔'
            | '〕'
            | '…'
            | '—'
            | '·'
            | '"'
            | '\''
            | '('
            | ')'
            | '['
            | ']'
            | '{'
            | '}'
            | ','
            | '.'
            | ';'
            | ':'
            | '?'
            | '!'
            | '-'
            | '_'
            | '`'
            | '~'
            | '@'
            | '#'
            | '$'
            | '%'
            | '^'
            | '&'
            | '*'
            | '+'
            | '='
            | '|'
            | '\\'
            | '/'
            | '<'
            | '>'
    )
}

/// 人类可读的文件大小
pub fn human_size(bytes: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    const GB: f64 = MB * 1024.0;
    if bytes as f64 >= GB {
        format!("{:.1} GB", bytes as f64 / GB)
    } else if bytes as f64 >= MB {
        format!("{:.1} MB", bytes as f64 / MB)
    } else if bytes as f64 >= KB {
        format!("{:.1} KB", bytes as f64 / KB)
    } else {
        format!("{bytes} B")
    }
}
pub mod epub;
