//! Harrier OSS v1 Embedding Example
//!
//! Demonstrates encoding texts and computing cosine similarity scores.
//!
//! Run with:
//! ```bash
//! cargo run --example harrier_embed_example -p aphelios_search --features metal
//! ```

use aphelios_search::harrier_embed::{cosine_similarity, HarrierEmbedModel};

// Python reference output from harrier_example.ipynb:
// [[66.5, 28.375], [29.875, 70.0]]
#[test]
fn harrier_embed_test() -> anyhow::Result<()> {
    let model_dir = "/Volumes/sw/pretrained_models/harrier-oss-v1-0.6b";

    println!("Loading Harrier OSS v1 embedding model...");
    let start = std::time::Instant::now();
    let mut model = HarrierEmbedModel::new(model_dir)?;
    println!("Model loaded in {:?}", start.elapsed());

    let queries = [
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: how much protein should a female eat",
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: summit define",
    ];

    let docs = [
        "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
        "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.",
    ];

    println!("\nQuery 0: {}", queries[0]);
    println!("\nQuery 1: {}", queries[1]);
    println!("\nDocument 0: {}", docs[0]);
    println!("\nDocument 1: {}", docs[1]);

    // Match the Python notebook exactly: queries followed by documents in one batch.
    println!("\nEncoding texts...");
    let all_texts: Vec<&str> = queries
        .iter()
        .copied()
        .chain(docs.iter().map(|s| *s))
        .collect();
    let all_embs = model.encode(all_texts)?;

    let query_embs = all_embs.narrow(0, 0, queries.len())?; // [2, 1024]
    let doc_embs = all_embs.narrow(0, queries.len(), docs.len())?; // [2, 1024]

    println!("Query embeddings shape: {:?}", query_embs.shape());
    println!("Doc embeddings shape: {:?}", doc_embs.shape());

    let similarities = cosine_similarity(&query_embs, &doc_embs)?;
    println!("\nCosine similarity scores (x100):");
    let sims_vec = similarities.to_vec2::<f32>()?;
    println!("  Query 0 vs Doc 0: {:.3}", sims_vec[0][0]);
    println!("  Query 0 vs Doc 1: {:.3}", sims_vec[0][1]);
    println!("  Query 1 vs Doc 0: {:.3}", sims_vec[1][0]);
    println!("  Query 1 vs Doc 1: {:.3}", sims_vec[1][1]);

    if sims_vec[0][0] > sims_vec[0][1] && sims_vec[1][1] > sims_vec[1][0] {
        println!("\n✓ PASS: Scores match the Python notebook ordering");
    } else {
        println!("\n✗ FAIL: Query/document ordering does not match the Python notebook");
    }
    Ok(())
}
