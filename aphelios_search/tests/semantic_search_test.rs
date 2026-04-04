//! Semantic search using Harrier OSS embeddings and usearch vector index.
//!
//! Provides vector similarity search for book titles using
//! `microsoft/harrier-oss-v1-0.6b` embeddings stored in a usearch index.

#[cfg(test)]
mod tests {
    use aphelios_search::semantic_search::SemanticIndex;

    /// Default path for Harrier model
    pub const DEFAULT_MODEL_DIR: &str = "/Volumes/sw/pretrained_models/harrier-oss-v1-0.6b";

    #[test]
    fn test_semantic_search() {
        let model_dir = DEFAULT_MODEL_DIR;
        let index_path = "/tmp/test_embeddings.usearch";

        // Clean up any existing test index
        let _ = std::fs::remove_file(index_path);

        let mut index = SemanticIndex::create(model_dir, index_path).unwrap();

        // Add some test books
        index.add_book(1, "计算机程序设计").unwrap();
        index.add_book(2, "数据结构与算法").unwrap();
        index.add_book(3, "机器学习入门").unwrap();

        // Search for similar
        let results = index.search("电脑编程", 3).unwrap();
        assert!(!results.is_empty());

        // Save
        index.save(index_path).unwrap();

        // Load
        let loaded = SemanticIndex::load(model_dir, index_path).unwrap();
        assert_eq!(loaded.len(), 3);

        // Clean up
        let _ = std::fs::remove_file(index_path);
    }
}
