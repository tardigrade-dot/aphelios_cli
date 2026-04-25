//! Minimal usearch test to verify it works on Apple Silicon

use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

#[test]
fn test_usearch_basic() {
    let options = IndexOptions {
        dimensions: 1024,
        metric: MetricKind::Cos,
        quantization: ScalarKind::F32,
        ..Default::default()
    };

    let index = Index::new(&options).unwrap();

    // Reserve space BEFORE adding - this is the key!
    index.reserve(10).unwrap();

    // Create a simple vector
    let vector: Vec<f32> = (0..1024).map(|_| rand::random::<f32>()).collect();

    // Add
    index.add(0, vector.as_slice()).unwrap();
    println!("Added successfully!");

    // Search
    let results = index.search(vector.as_slice(), 1).unwrap();
    println!("Search results: {} found", results.keys.len());
}

#[test]
fn test_usearch_with_embeddings() {
    use aphelios_search::harrier_embed::HarrierEmbedModel;
    use candle_core::DType;

    let model_dir = "/Volumes/sw/pretrained_models/harrier-oss-v1-0.6b";

    // Create model
    let mut model = HarrierEmbedModel::new(model_dir).unwrap();

    // Encode
    let embedding = model.encode(vec!["Computer Programming"]).unwrap();
    println!("Embedding shape: {:?}", embedding.shape());

    // Convert to F32 and get as Vec
    let embedding_f32 = embedding
        .to_dtype(DType::F32)
        .unwrap()
        .contiguous()
        .unwrap();
    let embedding_vec = embedding_f32.to_vec2::<f32>().unwrap().remove(0);
    println!("Embedding vec len: {}", embedding_vec.len());

    // Create usearch index with reserve
    let options = IndexOptions {
        dimensions: 1024,
        metric: MetricKind::Cos,
        quantization: ScalarKind::F32,
        ..Default::default()
    };

    let index = Index::new(&options).unwrap();
    index.reserve(10).unwrap();

    // Add vector
    index.add(0, embedding_vec.as_slice()).unwrap();
    println!("Added embedding to usearch successfully!");
}
