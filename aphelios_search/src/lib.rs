use anyhow::Result;
use fastembed::{
    EmbeddingModel, InitOptions, RerankInitOptions, RerankerModel, TextEmbedding, TextRerank,
};
use jieba_rs::Jieba;
use once_cell::sync::Lazy;
use rusqlite::{params, Connection};
use std::collections::HashMap;
use tracing::info;
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

pub struct SearchEngine {
    index: Index,
    db: Connection,
    embedding_model: TextEmbedding,
    rerank_model: TextRerank,
}

// 使用 Lazy 确保 Jieba 只初始化一次
static JIEBA: Lazy<Jieba> = Lazy::new(Jieba::new);

fn tokenize_chinese(text: &str) -> String {
    let words = JIEBA.cut(text, false);
    words.join(" ")
}

impl SearchEngine {
    pub fn new() -> Result<Self> {
        // 1. 初始化模型 (BGE-Small 维度为 512)
        let mut model_opts = InitOptions::default();
        model_opts.model_name = EmbeddingModel::BGESmallZHV15;
        let embedding_model = TextEmbedding::try_new(model_opts)?;

        let mut rerank_opts = RerankInitOptions::default();
        rerank_opts.model_name = RerankerModel::BGERerankerV2M3;
        let rerank_model = TextRerank::try_new(rerank_opts)?;

        // 2. 初始化 USearch
        let options = IndexOptions {
            dimensions: 512,
            metric: MetricKind::Cos,
            quantization: ScalarKind::F32,
            ..Default::default()
        };
        let index = Index::new(&options)?;
        index.reserve(1000)?;

        // 3. 初始化 SQLite (含 FTS5 虚拟表)
        let db = Connection::open_in_memory()?;
        // let db = Connection::open("data.db")?;
        // 创建元数据表
        db.execute(
            "CREATE TABLE IF NOT EXISTS files (id INTEGER PRIMARY KEY, path TEXT)",
            [],
        )?;

        // 创建 FTS5 虚拟表，使用 unicode61 分词器以支持中文
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
        info!("start build index for search[sqlit, usearch]");
        let embeddings = self.embedding_model.embed(file_names.clone(), None)?;

        for (i, (name, vector)) in file_names.iter().zip(embeddings).enumerate() {
            let id_i64 = i as i64;
            self.index.add(i as u64, &vector)?;

            self.db.execute(
                "INSERT OR REPLACE INTO files (id, path) VALUES (?1, ?2)",
                params![id_i64, name],
            )?;

            // 【关键】：存入 FTS 表的是分词后的结果，例如 "苏联 简史"
            let tokenized_name = tokenize_chinese(name);
            self.db.execute(
                "INSERT OR REPLACE INTO files_fts (rowid, path) VALUES (?1, ?2)",
                params![id_i64, tokenized_name],
            )?;
        }
        Ok(())
    }

    /// 混合搜索：结合BM25和向量搜索
    pub fn hybrid_search(
        &mut self,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<(String, f32, String)>> {
        // =========================
        // 1️⃣ BM25 召回
        // =========================
        let tokenized_query = tokenize_chinese(query);

        let mut stmt = self
            .db
            .prepare("SELECT rowid FROM files_fts WHERE path MATCH ?1 LIMIT ?2")?;

        let bm25_rows = stmt.query_map(params![tokenized_query, top_k as i64], |row| {
            Ok(row.get::<_, i64>(0)?)
        })?;

        let mut candidate_ids = HashMap::new();

        for row in bm25_rows {
            let id = row?;
            candidate_ids.insert(id, "精确".to_string());
        }

        info!("bm25 candidate size: {}", candidate_ids.len());

        // =========================
        // 2️⃣ 向量召回
        // =========================
        let bge_query = query;
        let query_embeddings = self.embedding_model.embed(vec![bge_query], None)?;
        let query_vec = &query_embeddings[0];

        // 增加向量召回的数量，以提高召回率
        let extended_top_k = std::cmp::max(top_k * 2, 10); // 至少召回10个，或top_k的两倍
        let vec_matches = self.index.search(query_vec, extended_top_k)?;

        for (&key_u64, &distance) in vec_matches.keys.iter().zip(vec_matches.distances.iter()) {
            let id_i64 = key_u64 as i64;

            candidate_ids
                .entry(id_i64)
                .and_modify(|label| *label = "精确+语义".to_string())
                .or_insert("语义".to_string());

            let path: String = self.db.query_row(
                "SELECT path FROM files WHERE id = ?1",
                params![id_i64],
                |row| row.get(0),
            )?;

            let similarity = 1.0 - distance;

            // info!(
            //     "vector hit: path={}, distance={:.4}, similarity={:.4}",
            //     path, distance, similarity
            // );
        }

        info!("total merged candidates: {}", candidate_ids.len());

        // =========================
        // 3️⃣ 构造 rerank 文档
        // =========================
        let mut rerank_documents = Vec::new();
        let mut id_list = Vec::new();
        let mut label_map = HashMap::new();
        let mut original_similarities = Vec::new(); // 存储原始相似度分数

        for (id, label) in candidate_ids {
            let path: String =
                self.db
                    .query_row("SELECT path FROM files WHERE id = ?1", params![id], |row| {
                        row.get(0)
                    })?;

            // 获取原始相似度分数
            let query_embeddings = self.embedding_model.embed(vec![query], None)?;
            let doc_embeddings = self.embedding_model.embed(vec![path.clone()], None)?;
            let original_similarity =
                self.calculate_cosine_similarity(&query_embeddings[0], &doc_embeddings[0]);

            rerank_documents.push(path);
            id_list.push(id);
            label_map.insert(id, label);
            original_similarities.push(original_similarity);
        }

        if rerank_documents.is_empty() {
            return Ok(vec![]);
        }

        // =========================
        // 4️⃣ rerank 精排
        // =========================
        let rerank_scores =
            self.rerank_model
                .rerank(query.to_string(), &rerank_documents, false, None)?;

        // =========================
        // 5️⃣ 组织结果 - 结合原始相似度和重排序分数
        // =========================
        let mut final_results = Vec::new();

        for rerank_result in rerank_scores.iter() {
            let doc_index = rerank_result.index; // 这是原始文档列表中的索引
            if doc_index < id_list.len() {
                let id = id_list[doc_index];
                let label = label_map.get(&id).unwrap().clone();
                let path = rerank_documents[doc_index].clone();
                let rerank_score = rerank_result.score;
                let original_similarity = original_similarities[doc_index];

                // 结合原始相似度和重排序分数
                let combined_score = 0.3 * original_similarity + 0.7 * rerank_score;

                final_results.push((path.to_string(), combined_score, label));
            }
        }

        // 按组合分数排序
        final_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // 只返回 top_k
        final_results.truncate(top_k);

        Ok(final_results)
    }

    // 辅助函数：计算余弦相似度
    fn calculate_cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let magnitude_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let magnitude_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if magnitude_a == 0.0 || magnitude_b == 0.0 {
            return 0.0;
        }

        dot_product / (magnitude_a * magnitude_b)
    }
}
