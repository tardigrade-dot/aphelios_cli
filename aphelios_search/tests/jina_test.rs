use std::path::PathBuf;

use anyhow::Result;
use aphelios_core::utils::core_utils;
use candle_core::{DType, Device, Tensor, D};
use fastembed::Qwen3TextEmbedding;
use tracing::info;

/// Jina embedding 模型封装（使用 Qwen3 架构）
struct JinaEmbeddingModel {
    model: Qwen3TextEmbedding,
    device: Device,
}

impl JinaEmbeddingModel {
    /// 加载 Jina 模型（使用 HuggingFace 模型 ID）
    fn load(model_id: &str, cache_dir: Option<PathBuf>) -> Result<Self> {
        let device = Device::new_metal(0).unwrap_or(Device::Cpu);

        info!("load model {} with cache {:?}", model_id, cache_dir);
        
        // 使用 fastembed 的 Qwen3TextEmbedding::from_hf 加载模型
        // 如果是本地路径，需要确保它是有效的 HF 模型 ID 格式
        let model = if model_id.starts_with("/Volumes") {
            // 对于本地路径，我们使用 symlink 或者直接使用 HF 模型 ID
            // 这里我们假设用户想要使用 HF 模型 ID
            info!("detected local path, using HF model ID instead");
            Qwen3TextEmbedding::from_hf(
                "jinaai/jina-embeddings-v5-text-small-retrieval",
                &device,
                DType::F32,
                512,
            )?
        } else {
            Qwen3TextEmbedding::from_hf(model_id, &device, DType::F32, 512)?
        };

        Ok(Self { model, device })
    }

    /// 编码单个文本
    fn encode(&self, text: &str, _prompt_type: &str) -> Result<Tensor> {
        // 使用 fastembed 进行编码
        let embeddings = self.model.embed(&[text])?;
        
        // 获取第一个 embedding 并转换为 Tensor
        let embedding_vec = &embeddings[0];
        let tensor = Tensor::new(embedding_vec.as_slice(), &self.device)?;
        
        Ok(tensor)
    }

    /// 批量编码
    fn encode_batch(&self, texts: Vec<&str>, _prompt_type: &str) -> Result<Tensor> {
        let embeddings = self.model.embed(&texts)?;
        
        // 将 embeddings 转换为 Tensor [batch_size, dim]
        let tensors: Vec<Tensor> = embeddings
            .iter()
            .map(|emb| Tensor::new(emb.as_slice(), &self.device))
            .collect::<Result<Vec<_>, _>>()?;
        
        Ok(Tensor::stack(&tensors, 0)?)
    }

    /// 计算余弦相似度
    fn similarity(query: &Tensor, documents: &Tensor) -> Result<Tensor> {
        // query: [dim], documents: [n, dim]
        // 结果：[n]
        let similarity = documents.matmul(&query.t()?)?;
        Ok(similarity.squeeze(1)?)
    }
}

/// Mean pooling: 对 token embeddings 进行平均池化（未使用，保留用于兼容）
#[allow(dead_code)]
fn mean_pooling(embeddings: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
    let (batch_size, seq_len, hidden_size) = embeddings.dims3()?;

    // 将 attention_mask 扩展为 [batch, seq, hidden]
    let mask_expanded = attention_mask
        .unsqueeze(2)?
        .expand((batch_size, seq_len, hidden_size))?;

    // 应用 mask
    let masked_embeddings = embeddings.mul(&mask_expanded)?;

    // 求和
    let sum_embeddings = masked_embeddings.sum(1)?;

    // 计算每个序列的有效长度
    let sum_mask = attention_mask
        .sum(1)?
        .unsqueeze(1)?
        .expand((1, hidden_size))?;

    // 平均
    let pooled = sum_embeddings.broadcast_div(&sum_mask)?;

    Ok(pooled)
}

/// L2 归一化（未使用，保留用于兼容）
#[allow(dead_code)]
fn normalize_l2(embeddings: &Tensor) -> Result<Tensor> {
    let norm = embeddings.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
    Ok(embeddings.broadcast_div(&norm)?)
}

#[test]
fn jina_test() -> Result<()> {
    let model_id = "jinaai/jina-embeddings-v5-text-small-retrieval";

    // 加载模型
    let model = JinaEmbeddingModel::load(
        model_id,
        Some(PathBuf::from("/Users/larry/coderesp/aphelios_cli/.cache")),
    )?;

    // 测试数据
    let query = "Which planet is known as the Red Planet?";
    let documents = vec![
        "Venus is often called Earth's twin because of its similar size and proximity.",
        "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
        "Jupiter, the largest planet in our solar system, has a prominent red spot.",
        "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
    ];

    // 编码 query
    println!("Encoding query...");
    let query_embedding = model.encode(query, "query")?;
    println!("Query embedding shape: {:?}", query_embedding.dims());

    // 编码 documents
    println!("Encoding documents...");
    let doc_embeddings = model.encode_batch(documents.clone(), "document")?;
    println!("Document embeddings shape: {:?}", doc_embeddings.dims());

    // 计算相似度
    let similarity = JinaEmbeddingModel::similarity(&query_embedding, &doc_embeddings)?;
    let similarity_values: Vec<f32> = similarity.to_vec1()?;

    // 输出结果
    println!("\n=== Similarity Scores ===");
    for (i, (doc, score)) in documents.iter().zip(similarity_values.iter()).enumerate() {
        println!("Doc {}: {:.4} - {}", i, score, doc);
    }

    // 找到最相关的文档
    let max_idx = similarity_values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    println!("\n=== Most Relevant Document ===");
    println!("Index: {}", max_idx);
    println!("Score: {:.4}", similarity_values[max_idx]);
    println!("Content: {}", documents[max_idx]);

    // 验证结果：Mars 应该是得分最高的
    assert_eq!(
        max_idx, 1,
        "Expected Mars document to have highest similarity"
    );

    Ok(())
}

#[test]
fn jina_test_simple() -> Result<()> {
    // 简化版本：只测试模型加载和编码
    core_utils::init_logging();
    
    // 使用 HuggingFace 模型 ID
    let model_id = "jinaai/jina-embeddings-v5-text-small-retrieval";

    let model = JinaEmbeddingModel::load(
        model_id,
        Some(PathBuf::from("/Users/larry/coderesp/aphelios_cli/.cache")),
    )?;

    let query = "What is machine learning?";
    let embedding = model.encode(query, "query")?;

    println!("Query: {}", query);
    println!("Embedding shape: {:?}", embedding.dims());
    println!("Embedding dim: {}", embedding.dims()[1]);

    Ok(())
}
