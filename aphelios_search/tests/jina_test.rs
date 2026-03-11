use std::path::PathBuf;

use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use hf_hub::api::sync::ApiBuilder;
use tokenizers::Tokenizer;

/// Jina embedding 模型封装
struct JinaEmbeddingModel {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl JinaEmbeddingModel {
    /// 加载 Jina 模型
    fn load(model_id: &str, cache_dir: PathBuf) -> Result<Self> {
        let device = Device::new_metal(0).unwrap_or(Device::Cpu);

        // 初始化 HF API
        let hf_api = ApiBuilder::new()
            .with_progress(true)
            .with_cache_dir(cache_dir)
            .build()?;

        let repo = hf_api.repo(hf_hub::Repo::with_revision(
            model_id.to_string(),
            hf_hub::RepoType::Model,
            "main".to_string(),
        ));

        // 下载模型文件
        let model_path = repo.get("model.safetensors")?;
        let config_path = repo.get("config.json")?;
        let tokenizer_path = repo.get("tokenizer.json")?;

        // 加载配置
        let config_str = std::fs::read_to_string(&config_path)?;
        let config: Config = serde_json::from_str(&config_str)?;

        // 加载分词器
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        // 加载 safetensors 权重并直接创建 VarBuilder
        let safetensors = candle_core::safetensors::load(&model_path, &device)?;
        let vb = VarBuilder::from_tensors(safetensors, DType::F32, &device);

        let model = BertModel::load(vb, &config)?;

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    /// 编码单个文本
    fn encode(&self, text: &str, prompt_type: &str) -> Result<Tensor> {
        // 构建带 prompt 的输入
        let input_text = match prompt_type {
            "query" => format!(
                "Represent this query for searching relevant documents: {}",
                text
            ),
            "document" => format!("Represent this document for retrieval: {}", text),
            _ => text.to_string(),
        };

        // 分词
        let encoding = self
            .tokenizer
            .encode(input_text, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        let token_ids = encoding.get_ids();
        let attention_mask = encoding.get_attention_mask();

        // 转换为 Tensor
        let token_ids_tensor = Tensor::new(token_ids, &self.device)?.unsqueeze(0)?;
        let attention_mask_tensor = Tensor::new(attention_mask, &self.device)?.unsqueeze(0)?;

        // 模型推理 - BertModel::forward 需要 token_type_ids 参数
        let token_type_ids = Tensor::zeros_like(&token_ids_tensor)?;
        let embeddings = self.model.forward(
            &token_ids_tensor,
            &token_type_ids,
            Some(&attention_mask_tensor),
        )?;

        // 使用 mean pooling 获取句子嵌入
        let embeddings = mean_pooling(&embeddings, &attention_mask_tensor)?;

        // L2 归一化
        let normalized = normalize_l2(&embeddings)?;

        Ok(normalized)
    }

    /// 批量编码
    fn encode_batch(&self, texts: Vec<&str>, prompt_type: &str) -> Result<Tensor> {
        let mut embeddings_list = Vec::new();

        for text in texts {
            let emb = self.encode(text, prompt_type)?;
            embeddings_list.push(emb);
        }

        Ok(Tensor::stack(&embeddings_list, 0)?)
    }

    /// 计算余弦相似度
    fn similarity(query: &Tensor, documents: &Tensor) -> Result<Tensor> {
        // query: [dim], documents: [n, dim]
        // 结果：[n]
        let similarity = documents.matmul(&query.t()?)?;
        Ok(similarity.squeeze(1)?)
    }
}

/// Mean pooling: 对 token embeddings 进行平均池化
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

/// L2 归一化
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
        PathBuf::from("/Users/larry/coderesp/aphelios_cli/.cache"),
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
    let model_id = "jinaai/jina-embeddings-v5-text-small-retrieval";

    let model = JinaEmbeddingModel::load(
        model_id,
        PathBuf::from("/Users/larry/coderesp/aphelios_cli/.cache"),
    )?;

    let query = "What is machine learning?";
    let embedding = model.encode(query, "query")?;

    println!("Query: {}", query);
    println!("Embedding shape: {:?}", embedding.dims());
    println!("Embedding dim: {}", embedding.dims()[1]);

    Ok(())
}
