use anyhow::Result;
use aphelios_cli::commands::utils;
use fastembed::{EmbeddingModel, InitOptions, RerankInitOptions, RerankerModel, TextEmbedding, TextRerank};
use jieba_rs::Jieba;
use once_cell::sync::Lazy;
use rusqlite::{params, Connection};
use tracing::info;
use std::collections::HashMap;
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

        utils::init_tracing();
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

    pub fn search(&mut self, query: &str, top_k: usize) -> Result<Vec<(String, f32, String)>> {

        // =========================
        // 1️⃣ BM25 召回
        // =========================
        let tokenized_query = tokenize_chinese(query);

        let mut stmt = self.db.prepare(
            "SELECT rowid FROM files_fts WHERE path MATCH ?1 LIMIT ?2"
        )?;

        let bm25_rows = stmt.query_map(params![tokenized_query, top_k as i64], |row| {
            Ok(row.get::<_, i64>(0)?)
        })?;

        let mut candidate_ids = HashMap::new();

        // for row in bm25_rows {
        //     let id = row?;
        //     candidate_ids.insert(id, "精确".to_string());
        // }

        info!("bm25 candidate size: {}", candidate_ids.len());

        // =========================
        // 2️⃣ 向量召回
        // =========================
        let bge_query = query;//format!("为这个句子生成表示以用于检索相关文章：{}", query);
        let query_embeddings = self.embedding_model.embed(vec![bge_query], None)?;
        let query_vec = &query_embeddings[0];

        let vec_matches = self.index.search(query_vec, top_k)?;

        for (&key_u64, &distance) in vec_matches.keys.iter().zip(vec_matches.distances.iter()){
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

            info!(
                "vector hit: path={}, distance={:.4}, similarity={:.4}",
                path,
                distance,
                similarity
            );
        }

        info!("total merged candidates: {}", candidate_ids.len());

        // =========================
        // 3️⃣ 构造 rerank 文档
        // =========================
        let mut rerank_documents = Vec::new();
        let mut id_list = Vec::new();
        let mut label_map = HashMap::new();

        for (id, label) in candidate_ids {
            let path: String = self.db.query_row(
                "SELECT path FROM files WHERE id = ?1",
                params![id],
                |row| row.get(0),
            )?;

            rerank_documents.push(path);
            id_list.push(id);
            label_map.insert(id, label);
        }

        if rerank_documents.is_empty() {
            return Ok(vec![]);
        }

        // =========================
        // 4️⃣ rerank 精排
        // =========================
        let prompt = "Given a web search query, retrieve relevant passages that answer the query";

        let rerank_query = format!("{}:{}", prompt, query);
        let rerank_scores = self
            .rerank_model
            .rerank(rerank_query, &rerank_documents, false, None)?;

        // =========================
        // 5️⃣ 组织结果
        // =========================
        let mut final_results = Vec::new();

        for (i, rerank_result) in rerank_scores.iter().enumerate() {
            let id = id_list[i];
            let label = label_map.get(&id).unwrap().clone();
            let path = rerank_documents[i].clone();
            let i_score = rerank_result.score;

            // let prob = 1.0 / (1.0 + (-i_score).exp());

            final_results.push((path.to_string(), i_score, label));
        }

        // rerank 分数越大越相关
        final_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // 只返回 top_k
        final_results.truncate(top_k);

        Ok(final_results)
    }

}

#[test]
fn main_test() -> Result<()> {
    let mut se = SearchEngine::new()?;

    let file_list = vec![
        "二手时间".into(),
        "苏联简史".into(),
        "切尔诺贝利的悲鸣".into(),
        "中国文学史".into(),
        "论人生短暂".into(),
        "地图在动".into(),
        "苏联的最后一天".into(),
    ];

    se.build_index(file_list)?;

    println!("\n搜索关键词: '苏联'");
    let results = se.search("苏联历史", 5)?;

    for (name, score, label) in results {
        println!("[{}] {} (得分: {:.4})", label, name, score);
    }

    Ok(())
}