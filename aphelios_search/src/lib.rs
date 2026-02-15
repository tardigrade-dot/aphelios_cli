use anyhow::Result;
use fastembed::{
    EmbeddingModel, InitOptions, RerankInitOptions, RerankerModel, TextEmbedding, TextRerank,
};
use jieba_rs::Jieba;
use once_cell::sync::Lazy;
use rusqlite::{params, Connection};
use std::collections::{HashMap, HashSet};
use tracing::info;
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

pub struct SearchEngine {
    index: Index,
    db: Connection,
    embedding_model: TextEmbedding,
    rerank_model: TextRerank,
}

// Jieba 只初始化一次
static JIEBA: Lazy<Jieba> = Lazy::new(Jieba::new);

fn tokenize_chinese(text: &str) -> String {
    let words = JIEBA.cut(text, false);
    words.join(" ")
}

impl SearchEngine {
    pub fn new() -> Result<Self> {
        // 初始化 embedding 模型
        let mut model_opts = InitOptions::default();
        model_opts.model_name = EmbeddingModel::BGESmallZHV15;
        let embedding_model = TextEmbedding::try_new(model_opts)?;

        // 初始化 reranker
        let mut rerank_opts = RerankInitOptions::default();
        rerank_opts.model_name = RerankerModel::BGERerankerV2M3;
        let rerank_model = TextRerank::try_new(rerank_opts)?;

        // 初始化 USearch
        let options = IndexOptions {
            dimensions: 512,
            metric: MetricKind::Cos,
            quantization: ScalarKind::F32,
            ..Default::default()
        };
        let index = Index::new(&options)?;
        index.reserve(1000)?;

        // 初始化 SQLite + FTS5
        let db = Connection::open_in_memory()?;

        db.execute(
            "CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY,
                path TEXT
            )",
            [],
        )?;

        db.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS files_fts USING fts5(
                path,
                tokenize='unicode61 remove_diacritics 0'
            )",
            [],
        )?;

        Ok(Self {
            index,
            db,
            embedding_model,
            rerank_model,
        })
    }

    pub fn build_index(&mut self, file_names: Vec<String>) -> Result<()> {
        info!("building index...");

        let embeddings = self.embedding_model.embed(file_names.clone(), None)?;

        for (i, (name, vector)) in file_names.iter().zip(embeddings).enumerate() {
            let id_i64 = i as i64;

            self.index.add(i as u64, &vector)?;

            self.db.execute(
                "INSERT OR REPLACE INTO files (id, path) VALUES (?1, ?2)",
                params![id_i64, name],
            )?;

            let tokenized_name = tokenize_chinese(name);
            self.db.execute(
                "INSERT OR REPLACE INTO files_fts (rowid, path) VALUES (?1, ?2)",
                params![id_i64, tokenized_name],
            )?;
        }

        Ok(())
    }

    pub fn hybrid_search(
        &mut self,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<(String, f32, String)>> {
        let bm25_limit = 20;
        let vector_limit = 20;

        // =========================
        // 1️⃣ BM25 top 20
        // =========================
        let tokenized_query = tokenize_chinese(query);

        let mut stmt = self.db.prepare(
            "SELECT rowid, bm25(files_fts) as score
             FROM files_fts
             WHERE path MATCH ?1
             ORDER BY score
             LIMIT ?2",
        )?;

        let bm25_rows = stmt.query_map(params![tokenized_query, bm25_limit as i64], |row| {
            Ok(row.get::<_, i64>(0)?)
        })?;

        let mut candidate_ids = Vec::new();
        let mut label_map: HashMap<i64, String> = HashMap::new();
        let mut seen = HashSet::new();

        for row in bm25_rows {
            let id = row?;
            if seen.insert(id) {
                candidate_ids.push(id);
                label_map.insert(id, "BM25".to_string());
            }
        }

        info!("bm25 candidates: {}", candidate_ids.len());

        // =========================
        // 2️⃣ Vector top 20
        // =========================
        let query_embedding = self.embedding_model.embed(vec![query], None)?;
        let query_vec = &query_embedding[0];

        let vec_matches = self.index.search(query_vec, vector_limit)?;

        for &key_u64 in vec_matches.keys.iter() {
            let id = key_u64 as i64;

            if seen.insert(id) {
                candidate_ids.push(id);
                label_map.insert(id, "Vector".to_string());
            } else {
                // 已存在，说明来自 BM25
                label_map.insert(id, "BM25+Vector".to_string());
            }
        }

        info!("merged candidates: {}", candidate_ids.len());

        if candidate_ids.is_empty() {
            return Ok(vec![]);
        }

        // =========================
        // 3️⃣ 构造 rerank 文档
        // =========================
        let mut documents = Vec::new();

        for id in &candidate_ids {
            let path: String =
                self.db
                    .query_row("SELECT path FROM files WHERE id = ?1", params![id], |row| {
                        row.get(0)
                    })?;
            documents.push(path);
        }

        // =========================
        // 4️⃣ Rerank
        // =========================
        let rerank_results =
            self.rerank_model
                .rerank(query.to_string(), &documents, false, None)?;

        // =========================
        // 5️⃣ 仅按 rerank_score 排序 + 保留 label
        // =========================
        let mut final_results = Vec::new();

        for r in rerank_results {
            let idx = r.index;
            if idx < documents.len() {
                let id = candidate_ids[idx];
                let path = documents[idx].clone();
                let label = label_map
                    .get(&id)
                    .cloned()
                    .unwrap_or_else(|| "Unknown".to_string());

                final_results.push((path, r.score, label));
            }
        }

        final_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        final_results.truncate(top_k);

        Ok(final_results)
    }
}
