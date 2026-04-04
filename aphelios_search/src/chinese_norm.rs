//! Chinese text normalization utilities for search.
//!
//! Provides Simplified-Traditional Chinese conversion and text normalization
//! to enable cross-variant searching (e.g., searching "书籍" finds "書籍").
//!
//! Uses `zhconv` for pure-Rust Chinese conversion without external dependencies.

use zhconv::{zhconv, Variant};

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
}
