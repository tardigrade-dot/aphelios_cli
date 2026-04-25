//! In-memory book search with Chinese Simplified/Traditional support.
//!
//! Scans a books directory, extracts metadata from filenames, and
//! provides substring search with zhconv-based S/T conversion
//! so searching in simplified Chinese also matches traditional titles.

use anyhow::Result;
use std::path::Path;
use tracing::info;

/// Search mode — currently only keyword search is supported.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SearchMode {
    #[default]
    Keyword,
}

/// A single book entry.
#[derive(Debug, Clone)]
pub struct BookInfo {
    pub id: i64,
    pub title: String,
    pub author: Option<String>,
    pub file_path: String,
    pub file_type: String,
    pub file_size: u64,
}

/// A set of search results.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub books: Vec<BookInfo>,
    pub total: usize,
}

/// Convert simplified Chinese text to traditional Chinese.
pub fn to_traditional(text: &str) -> String {
    zhconv::zhconv(text, zhconv::Variant::ZhHant)
}

/// Convert traditional Chinese text to simplified Chinese.
pub fn to_simplified(text: &str) -> String {
    zhconv::zhconv(text, zhconv::Variant::ZhHans)
}

/// Supported book file extensions.
const BOOK_EXTENSIONS: &[&str] = &["pdf", "epub", "txt", "mobi", "azw3"];

/// Check whether a file path has a book extension.
fn is_book_file(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| BOOK_EXTENSIONS.contains(&e.to_lowercase().as_str()))
        .unwrap_or(false)
}

/// Parse a filename into (title, optional author).
///
/// Supports these formats:
/// - `书名 (作者)` / `书名（作者）`
/// - `书名 - 作者`
/// - `[作者] 书名`
/// - Everything else → treated as the title alone.
pub fn extract_metadata(file_path: &Path) -> (String, Option<String>) {
    let filename = file_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("未知书名")
        .to_string();

    // Try "书名 (作者)"  with ASCII parens
    if let Some(paren_pos) = filename.find('(') {
        let title = filename[..paren_pos].trim().to_string();
        let author = filename[paren_pos + 1..]
            .find(')')
            .map(|end| filename[paren_pos + 1..paren_pos + 1 + end].trim().to_string());
        // Strip English subtitle after " = "
        let title = if let Some(eq_pos) = title.find(" = ") {
            title[..eq_pos].trim().to_string()
        } else {
            title
        };
        return (title, author);
    }

    // Try "书名（作者）" with fullwidth parens
    if let Some(paren_pos) = filename.find('（') {
        let content_start = paren_pos + '（'.len_utf8();
        let title = filename[..paren_pos].trim().to_string();
        let author = filename[content_start..]
            .find('）')
            .map(|end| filename[content_start..content_start + end].trim().to_string());
        let title = if let Some(eq_pos) = title.find(" = ") {
            title[..eq_pos].trim().to_string()
        } else {
            title
        };
        return (title, author);
    }

    // Try "书名 - 作者"
    if let Some(dash_pos) = filename.find(" - ") {
        let title = filename[..dash_pos].trim().to_string();
        let author = Some(filename[dash_pos + 3..].trim().to_string());
        return (title, author);
    }

    // Try "[作者] 书名"
    if let (Some(start), Some(end)) = (filename.find('['), filename.find(']')) {
        let author = Some(filename[start + 1..end].trim().to_string());
        let title = filename[end + 1..].trim().to_string();
        if !title.is_empty() {
            return (title, author);
        }
    }

    // Fallback: whole filename is the title
    (filename, None)
}

/// Recursively scan a directory and return all books found.
pub fn scan_books(book_dir: &str) -> Result<Vec<BookInfo>> {
    let dir = Path::new(book_dir);
    if !dir.exists() {
        info!("Books directory does not exist: {}", book_dir);
        return Ok(Vec::new());
    }

    let mut books = Vec::new();
    scan_dir_recursive(dir, &mut books, &mut 0i64)?;

    info!("Scanned {} books from {}", books.len(), book_dir);
    Ok(books)
}

fn scan_dir_recursive(dir: &Path, books: &mut Vec<BookInfo>, next_id: &mut i64) -> Result<()> {
    if !dir.is_dir() {
        return Ok(());
    }

    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            scan_dir_recursive(&path, books, next_id)?;
        } else if is_book_file(&path) {
            let file_path = path.to_string_lossy().to_string();
            let file_type = path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_lowercase();
            let file_size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            let (title, author) = extract_metadata(&path);

            // Normalize author empty string to None
            let author = author.filter(|a| !a.is_empty());

            *next_id += 1;
            books.push(BookInfo {
                id: *next_id,
                title,
                author,
                file_path,
                file_type,
                file_size,
            });
        }
    }

    Ok(())
}

/// Search books in-memory by title or author.
///
/// The query is matched as a case-insensitive substring against both the
/// original title and its simplified-Chinese form, so searching in either
/// simplified or traditional Chinese will find matching titles.
pub fn search_books(books: &[BookInfo], query: &str, limit: usize) -> SearchResult {
    let query = query.trim();
    if query.is_empty() {
        let total = books.len();
        let results: Vec<BookInfo> = books.iter().take(limit).cloned().collect();
        return SearchResult {
            total,
            books: results,
        };
    }

    // Pre-compute query variants: original, simplified, traditional
    let query_lower = query.to_lowercase();
    let query_s = to_simplified(&query_lower);
    let query_t = to_traditional(&query_lower);

    let matched: Vec<BookInfo> = books
        .iter()
        .filter(|b| {
            let title_lower = b.title.to_lowercase();
            let title_s = to_simplified(&title_lower);

            // Check original title
            if title_lower.contains(&query_lower) || title_s.contains(&query_s) {
                return true;
            }

            // Check traditional query against simplified title
            if query_t != query_s && title_s.contains(&query_t) {
                return true;
            }

            // Also check author if present, with the same S/T conversion
            if let Some(ref author) = b.author {
                let author_lower = author.to_lowercase();
                let author_s = to_simplified(&author_lower);
                if author_lower.contains(&query_lower)
                    || author_s.contains(&query_s)
                    || (query_t != query_s && author_s.contains(&query_t))
                {
                    return true;
                }
            }

            false
        })
        .take(limit)
        .cloned()
        .collect();

    let total = matched.len();
    SearchResult {
        books: matched,
        total,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_metadata_ascii_parens() {
        let path = Path::new("/books/三体 (刘慈欣).epub");
        let (title, author) = extract_metadata(path);
        assert_eq!(title, "三体");
        assert_eq!(author.as_deref(), Some("刘慈欣"));
    }

    #[test]
    fn test_extract_metadata_fullwidth_parens() {
        let path = Path::new("/books/三体（刘慈欣）.epub");
        let (title, author) = extract_metadata(path);
        assert_eq!(title, "三体");
        assert_eq!(author.as_deref(), Some("刘慈欣"));
    }

    #[test]
    fn test_extract_metadata_dash() {
        let path = Path::new("/books/三体 - 刘慈欣.pdf");
        let (title, author) = extract_metadata(path);
        assert_eq!(title, "三体");
        assert_eq!(author.as_deref(), Some("刘慈欣"));
    }

    #[test]
    fn test_extract_metadata_bracket() {
        let path = Path::new("/books/[刘慈欣] 三体.txt");
        let (title, author) = extract_metadata(path);
        assert_eq!(title, "三体");
        assert_eq!(author.as_deref(), Some("刘慈欣"));
    }

    #[test]
    fn test_extract_metadata_title_only() {
        let path = Path::new("/books/三体.pdf");
        let (title, author) = extract_metadata(path);
        assert_eq!(title, "三体");
        assert_eq!(author, None);
    }

    #[test]
    fn test_search_simplified_finds_traditional() {
        let books = vec![BookInfo {
            id: 1,
            title: "中國歷史".to_string(),
            author: None,
            file_path: "/books/中國歷史.pdf".to_string(),
            file_type: "pdf".to_string(),
            file_size: 100,
        }];

        // Search in simplified should find traditional title
        let result = search_books(&books, "中国", 10);
        assert_eq!(result.total, 1);
    }

    #[test]
    fn test_search_traditional_finds_simplified() {
        let books = vec![BookInfo {
            id: 1,
            title: "中国历史".to_string(),
            author: None,
            file_path: "/books/中国历史.pdf".to_string(),
            file_type: "pdf".to_string(),
            file_size: 100,
        }];

        // Search in traditional should find simplified title
        let result = search_books(&books, "中國", 10);
        assert_eq!(result.total, 1);
    }

    #[test]
    fn test_search_partial_match() {
        let books = vec![BookInfo {
            id: 1,
            title: "中国历史长卷".to_string(),
            author: None,
            file_path: "/books/test.pdf".to_string(),
            file_type: "pdf".to_string(),
            file_size: 100,
        }];

        let result = search_books(&books, "历史", 10);
        assert_eq!(result.total, 1);
    }

    #[test]
    fn test_search_empty_query_returns_all() {
        let books = vec![
            BookInfo {
                id: 1,
                title: "A".to_string(),
                author: None,
                file_path: "/a.pdf".to_string(),
                file_type: "pdf".to_string(),
                file_size: 100,
            },
            BookInfo {
                id: 2,
                title: "B".to_string(),
                author: None,
                file_path: "/b.pdf".to_string(),
                file_type: "pdf".to_string(),
                file_size: 100,
            },
        ];

        let result = search_books(&books, "", 10);
        assert_eq!(result.total, 2);
    }

    #[test]
    fn test_to_traditional_and_simplified() {
        assert_eq!(to_traditional("计算机"), "計算機");
        assert_eq!(to_simplified("計算機"), "计算机");
    }
}
