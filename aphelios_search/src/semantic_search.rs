//! Semantic search using Harrier OSS embeddings and usearch vector index.
//!
//! Provides vector similarity search for book titles using
//! `microsoft/harrier-oss-v1-0.6b` embeddings stored in a usearch index.

use anyhow::{Context, Result};
use candle_core::DType;
use usearch::{Index, IndexOptions, Key, MetricKind, ScalarKind};

use crate::harrier_embed::HarrierEmbedModel;

/// Embedding dimension for Harrier OSS v1 (1024)
pub const EMBEDDING_DIM: usize = 1024;

/// Default path for the semantic index
pub const DEFAULT_INDEX_PATH: &str = "/Volumes/sw/books/embeddings.usearch";

/// Default path for Harrier model
pub const DEFAULT_MODEL_DIR: &str = "/Volumes/sw/pretrained_models/harrier-oss-v1-0.6b";

/// Semantic search index for book titles
pub struct SemanticIndex {
    index: Index,
    model: HarrierEmbedModel,
    book_ids: Vec<i64>,
}

impl SemanticIndex {
    /// Create a new semantic index (tries to load existing)
    pub fn new(model_dir: &str, index_path: &str) -> Result<Self> {
        let options = IndexOptions {
            dimensions: EMBEDDING_DIM,
            metric: MetricKind::Cos,
            quantization: ScalarKind::F32,
            ..Default::default()
        };

        let index = Index::new(&options).context("Failed to create usearch index")?;

        // Try to load existing index
        if std::path::Path::new(index_path).exists() {
            let index = index;
            index
                .load(index_path)
                .context(format!("Failed to load index from {}", index_path))?;
            return Ok(Self {
                index,
                model: HarrierEmbedModel::new(model_dir)?,
                book_ids: Vec::new(),
            });
        }

        Ok(Self {
            index,
            model: HarrierEmbedModel::new(model_dir)?,
            book_ids: Vec::new(),
        })
    }

    /// Create a new empty index (overwrites existing)
    pub fn create(model_dir: &str, _index_path: &str) -> Result<Self> {
        let options = IndexOptions {
            dimensions: EMBEDDING_DIM,
            metric: MetricKind::Cos,
            quantization: ScalarKind::F32,
            ..Default::default()
        };

        let index = Index::new(&options).context("Failed to create usearch index")?;

        // Reserve initial space for vectors - key to avoid SIGSEGV on Apple Silicon
        index
            .reserve(100)
            .context("Failed to reserve space in index")?;

        Ok(Self {
            index,
            model: HarrierEmbedModel::new(model_dir)?,
            book_ids: Vec::new(),
        })
    }

    /// Add a book with its title embedding
    pub fn add_book(&mut self, book_id: i64, title: &str) -> Result<()> {
        // Clean title: remove trailing " (Z-Library)" if present
        let cleaned_title = title
            .strip_suffix(" (Z-Library)")
            .unwrap_or(title)
            .strip_suffix(" (z-lib.org)")
            .unwrap_or(title)
            .strip_suffix(" [z-lib.org]")
            .unwrap_or(title)
            .trim();

        // Encode the title (returns [1, 1024] for single input, dtype BF16 on metal)
        let embedding = self.model.encode(vec![cleaned_title])?;

        // Convert to F32 then make contiguous before copying to CPU
        // (handle BF16 -> F32 and 2D tensor [1, 1024] -> [1024])
        let embedding_f32 = embedding.to_dtype(DType::F32)?.contiguous()?;
        let embedding_vec = embedding_f32.to_vec2::<f32>()?.remove(0);

        // Add to index (use book_ids.len() as key)
        let vector_id: Key = self.book_ids.len() as Key;

        // Check if we need to reserve more space (reserve in batches of 100)
        // usearch::size() returns current number of vectors
        if vector_id as usize >= self.index.size() {
            let new_capacity = ((vector_id as usize / 100) + 1) * 100;
            let _ = self.index.reserve(new_capacity);
        }

        self.book_ids.push(book_id);

        self.index
            .add(vector_id, embedding_vec.as_slice())
            .context("Failed to add embedding to index")?;

        Ok(())
    }

    /// Add multiple books at once (batch processing)
    pub fn add_books(&mut self, books: &[(i64, &str)]) -> Result<()> {
        if books.is_empty() {
            return Ok(());
        }

        // Reserve space for all books before adding - prevents reallocation issues
        let additional = books.len();
        let current_len = self.book_ids.len();
        self.index
            .reserve(current_len + additional)
            .context("Failed to reserve space")?;

        // Batch encode all titles (dtype BF16 on metal)
        let titles: Vec<&str> = books.iter().map(|(_, t)| *t).collect();
        let embeddings = self.model.encode(titles)?;
        let embeddings_f32 = embeddings.to_dtype(DType::F32)?.contiguous()?;
        let embeddings_2d = embeddings_f32.to_vec2::<f32>()?;

        for (i, (book_id, _title)) in books.iter().enumerate() {
            let vector_id: Key = self.book_ids.len() as Key;
            self.book_ids.push(*book_id);
            self.index
                .add(vector_id, embeddings_2d[i].as_slice())
                .context("Failed to add embedding to index")?;
        }

        Ok(())
    }

    /// Search for similar titles
    /// Returns vector of (book_id, similarity_score) sorted by relevance
    pub fn search(&mut self, query: &str, limit: usize) -> Result<Vec<(i64, f32)>> {
        // Encode query (dtype BF16 on metal)
        let query_emb = self.model.encode(vec![query])?;
        let query_emb_f32 = query_emb.to_dtype(DType::F32)?.contiguous()?;
        let query_vec = query_emb_f32.to_vec2::<f32>()?.remove(0);

        // Search
        let results = self
            .index
            .search(query_vec.as_slice(), limit)
            .context("Search failed")?;

        // Extract keys and distances from Matches struct
        let keys = &results.keys;
        let distances = &results.distances;

        // Map back to book_ids and sort by score
        let mut scored_results: Vec<(i64, f32)> = Vec::new();

        for i in 0..keys.len() {
            let idx = keys[i];
            let distance = distances[i];

            if idx < self.book_ids.len() as Key {
                let book_id = self.book_ids[idx as usize];
                // usearch returns distance, convert to similarity (1 - distance for Cos)
                let similarity = 1.0 - distance;
                scored_results.push((book_id, similarity));
            }
        }

        // Sort by similarity (descending)
        scored_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(scored_results)
    }

    /// Save index to disk (saves both the vector index and book_ids metadata)
    pub fn save(&self, path: &str) -> Result<()> {
        // Save the usearch index
        self.index.save(path).context("Failed to save index")?;

        // Save book_ids alongside the index (as path.metadata)
        let metadata_path = format!("{}.metadata", path);
        let metadata =
            serde_json::to_string(&self.book_ids).context("Failed to serialize book_ids")?;
        std::fs::write(&metadata_path, metadata).context("Failed to write book_ids metadata")?;

        Ok(())
    }

    /// Load index from disk
    pub fn load(model_dir: &str, index_path: &str) -> Result<Self> {
        let options = IndexOptions {
            dimensions: EMBEDDING_DIM,
            metric: MetricKind::Cos,
            quantization: ScalarKind::F32,
            ..Default::default()
        };

        let index = Index::new(&options).context("Failed to create usearch index")?;

        let index = index;
        index
            .load(index_path)
            .with_context(|| format!("Failed to load index from {}", index_path))?;

        // Load book_ids metadata
        let metadata_path = format!("{}.metadata", index_path);
        let book_ids: Vec<i64> = if std::path::Path::new(&metadata_path).exists() {
            let metadata = std::fs::read_to_string(&metadata_path)
                .context("Failed to read book_ids metadata")?;
            serde_json::from_str(&metadata).context("Failed to deserialize book_ids")?
        } else {
            Vec::new()
        };

        Ok(Self {
            index,
            model: HarrierEmbedModel::new(model_dir)?,
            book_ids,
        })
    }

    /// Get the number of indexed books
    pub fn len(&self) -> usize {
        self.book_ids.len()
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.book_ids.is_empty()
    }
}

/// Build semantic index for all books in the database
pub fn build_semantic_index(
    model_dir: &str,
    index_path: &str,
    books: &[(i64, &str)], // (book_id, title)
    progress_callback: Option<&dyn Fn(usize)>,
) -> Result<usize> {
    let mut semantic_index = SemanticIndex::create(model_dir, index_path)?;
    let total = books.len();

    for (idx, (book_id, title)) in books.iter().enumerate() {
        if let Some(callback) = progress_callback {
            callback(idx * 100 / total);
        }
        semantic_index.add_book(*book_id, title)?;
    }

    semantic_index.save(index_path)?;

    if let Some(callback) = progress_callback {
        callback(100);
    }

    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires model files
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
