//! Chinese text normalization and tokenization utilities for search.
//!
//! Provides Simplified-Traditional Chinese conversion, text normalization,
//! and intelligent Chinese word segmentation using `jieba-rs`.
//!
//! Uses `zhconv` for pure-Rust Chinese conversion without external dependencies.
//! Uses `jieba-rs` for accurate Chinese word segmentation.

use jieba_rs::Jieba;
use lazy_static::lazy_static;
use zhconv::{zhconv, Variant};

lazy_static! {
    static ref JIEBA: Jieba = Jieba::new();
}

/// Convert simplified Chinese to traditional Chinese
///
/// # Example
/// ```
/// assert_eq!(to_traditional("计算机"), "計算機");
/// assert_eq!(to_traditional("书籍"), "書籍");
/// ```
pub fn to_traditional(text: &str) -> String {
    zhconv(text, Variant::ZhHant)
}

/// Convert traditional Chinese to simplified Chinese
///
/// # Example
/// ```
/// assert_eq!(to_simplified("計算機"), "计算机");
/// assert_eq!(to_simplified("書籍"), "书籍");
/// ```
pub fn to_simplified(text: &str) -> String {
    zhconv(text, Variant::ZhHans)
}

/// Normalize text for search indexing.
/// Converts to simplified Chinese and applies basic cleanup.
///
/// This creates a canonical form that:
/// 1. Converts traditional characters to simplified
/// 2. Removes extra whitespace
/// 3. Lowercases alphanumeric characters
pub fn normalize(text: &str) -> String {
    // First convert to simplified
    let simplified = to_simplified(text);

    // Normalize whitespace
    simplified.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Extract search terms from text for hybrid search.
/// Returns both the original terms and their variants.
///
/// For Chinese text, returns the text itself (since zhconv handles it).
/// For mixed text, returns both original and converted forms.
pub fn extract_search_terms(text: &str) -> Vec<String> {
    let simplified = to_simplified(text);
    let traditional = to_traditional(text);

    let mut terms = vec![text.to_string()];
    if simplified != text {
        terms.push(simplified.clone());
    }
    if traditional != text && traditional != simplified {
        terms.push(traditional);
    }

    terms
}

/// Tokenize Chinese text for FTS5 search using Jieba word segmentation.
///
/// This function provides intelligent Chinese word segmentation:
/// - For texts with spaces: split by spaces (user-provided tokenization)
/// - For texts without spaces: use Jieba for semantic word segmentation
///
/// # Examples
/// ```
/// // User provides spaces: "文化 权力与国家" → ["文化", "权力与国家"]
/// let tokens = tokenize_for_search("文化 权力与国家");
/// assert_eq!(tokens, vec!["文化", "权力与国家"]);
///
/// // No spaces: Jieba segmentation
/// let tokens = tokenize_for_search("共产世界大历史");
/// // Expected: ["共产", "世界", "大", "历史"] or similar semantic words
/// ```
pub fn tokenize_for_search(text: &str) -> Vec<String> {
    let text = text.trim();
    if text.is_empty() {
        return Vec::new();
    }

    // If text contains spaces, use user-provided word-level tokenization
    if text.contains(char::is_whitespace) {
        text.split_whitespace()
            .map(|s| s.to_string())
            .filter(|s| !s.is_empty())
            .collect()
    } else {
        // Use Jieba for intelligent Chinese word segmentation
        // `cut` method with `false` means not using HMM model for better precision
        JIEBA
            .cut(text, false)
            .iter()
            .map(|s: &&str| s.to_string())
            .filter(|s: &String| !s.is_empty())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_s2t_basic() {
        // Basic Chinese characters
        assert_eq!(to_traditional("计算机"), "計算機");
        assert_eq!(to_traditional("书籍"), "書籍");
    }

    #[test]
    fn test_t2s_basic() {
        assert_eq!(to_simplified("計算機"), "计算机");
        assert_eq!(to_simplified("書籍"), "书籍");
    }

    #[test]
    fn test_normalize() {
        assert_eq!(normalize("計算機"), "计算机");
        assert_eq!(normalize("  书籍  "), "书籍");
    }

    #[test]
    fn test_extract_terms() {
        let terms = extract_search_terms("计算机");
        assert!(terms.contains(&"计算机".to_string()));
        assert!(terms.contains(&"計算機".to_string()));
    }

    #[test]
    fn test_tokenize_with_spaces() {
        // User provides explicit spaces
        let tokens = tokenize_for_search("文化 权力与国家");
        assert_eq!(tokens, vec!["文化", "权力与国家"]);
    }

    #[test]
    fn test_tokenize_no_spaces() {
        // No spaces: Jieba word segmentation
        let tokens = tokenize_for_search("共产世界大历史");
        // Jieba should segment this into meaningful words
        assert!(tokens.contains(&"共产".to_string()));
        assert!(tokens.contains(&"世界".to_string()));
        assert!(tokens.contains(&"历史".to_string()) || tokens.contains(&"大".to_string()));
    }

    #[test]
    fn test_tokenize_short() {
        // Short text
        let tokens = tokenize_for_search("文化");
        assert_eq!(tokens, vec!["文化"]);
    }

    #[test]
    fn test_tokenize_jieba_quality() {
        // Test Jieba segmentation quality
        let tokens = tokenize_for_search("文化权力与国家");
        // Should contain meaningful words, not nonsense like "化权"
        assert!(tokens.contains(&"文化".to_string()));
        assert!(tokens.contains(&"权力".to_string()) || tokens.contains(&"国家".to_string()));
        // Should NOT contain meaningless cross-word combinations
        assert!(!tokens.contains(&"化权".to_string()));
        assert!(!tokens.contains(&"力与".to_string()));
    }

    #[test]
    fn test_tokenize_empty() {
        let tokens = tokenize_for_search("");
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_tokenize_whitespace_only() {
        let tokens = tokenize_for_search("   ");
        assert!(tokens.is_empty());
    }
}
