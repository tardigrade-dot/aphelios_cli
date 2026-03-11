use std::{
    fs,
    path::{Path, PathBuf},
    sync::Mutex,
};

use anyhow::{Ok, Result};
use candle_core::{DType, Device};
use fastembed::{
    EmbeddingModel, InitOptions, Qwen3TextEmbedding, RerankInitOptions, RerankerModel,
    TextEmbedding, TextRerank,
};
use once_cell::sync::Lazy;
use rusqlite::{Connection, params};
use tracing::info;
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

pub struct SearchEngine {
    index: Index,
    db: Connection,
    index_file: PathBuf,
    model: Qwen3TextEmbedding,
}

impl SearchEngine {
    pub fn new(model: Qwen3TextEmbedding, index_file: impl AsRef<Path>) -> anyhow::Result<Self> {
        // 1. 初始化 USearch
        let options = IndexOptions {
            dimensions: 1024,
            metric: MetricKind::Cos,
            quantization: ScalarKind::F32,
            ..Default::default()
        };
        let index = Index::new(&options)?;
        index.reserve(1000)?; // 预留 1000 个槽位

        // 2. 初始化 SQLite
        let db = Connection::open("metadata.db")?;
        db.execute(
            "CREATE TABLE IF NOT EXISTS files (id INTEGER PRIMARY KEY, path TEXT)",
            [],
        )?;

        Ok(Self {
            index,
            db,
            index_file: index_file.as_ref().to_path_buf(),
            model,
        })
    }

    pub fn build_index(&mut self, file_names: Vec<String>) -> anyhow::Result<()> {
        if self.index_file.exists() {
            println!("向量文件已存在, 跳过构建向量索引步骤");
            let _ = self.index.load(self.index_file.to_str().unwrap());
            return Ok(());
        }
        print!("向量文件不存在, 执行索引步骤...");
        let embeddings = self.model.embed(&file_names)?;

        for (i, (name, vector)) in file_names.iter().zip(embeddings.iter()).enumerate() {
            let id_u64 = i as u64;
            let id_i64 = i as i64;

            self.index.add(id_u64, vector.as_ref())?;

            self.db.execute(
                "INSERT OR REPLACE INTO files (id, path) VALUES (?1, ?2)",
                params![id_i64, name],
            )?;
        }

        let r = self.index.save(self.index_file.to_str().unwrap());
        if r.is_err() {
            println!();
        }
        println!("完成 {} 个文件的索引构建", file_names.len());
        Ok(())
    }

    pub fn get_vector(&mut self, text: &str) -> Result<Vec<f32>> {
        // let mut model = MODEL.lock().unwrap(); // 获取可变锁

        let embeddings = self.model.embed(&[text]).expect("Embedding 推理失败");

        Ok(embeddings[0].clone())
    }

    pub fn search(&mut self, query: &str, top_k: usize) -> anyhow::Result<Vec<(String, f32)>> {
        let bge_query = format!("给定关键描述, 返回匹配的书籍:{}", query);
        // 1. 生成 query 向量
        let vector = self.get_vector(&bge_query);

        // 2. 向量搜索
        let matches = self.index.search(&vector?, top_k)?;

        let mut results = Vec::with_capacity(matches.keys.len());

        // 3. 通过 id 反查 SQLite
        for (&id, &score) in matches.keys.iter().zip(matches.distances.iter()) {
            let mut stmt = self.db.prepare("SELECT path FROM files WHERE id = ?1")?;
            let path: String = stmt.query_row(params![id as i64], |row| row.get(0))?;

            results.push((path, score));
        }
        Ok(results)
    }
}
