//! Use a custom embedder instead of the built-in BGE-Small model.
//!
//! This example uses random vectors to avoid any network or model dependency.
//! In production you would call OpenAI, Cohere, or your own inference server.
//!
//! ```
//! cargo run --example custom_embedder
//! ```

use rustkit_semantic::{Embedder, Error, SemanticIndex};
use serde_json::json;

/// Toy embedder that returns random vectors — for illustration only.
struct RandomEmbedder {
    dims: usize,
}

impl Embedder for RandomEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>, Error> {
        // Deterministic hash-based pseudo-random so similar calls are comparable.
        let seed = text.bytes().fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
        let v: Vec<f32> = (0..self.dims)
            .map(|i| {
                let x = seed.wrapping_mul(i as u64 + 1).wrapping_add(0xdeadbeef);
                (x as f32 / u64::MAX as f32) * 2.0 - 1.0
            })
            .collect();

        // Normalise to unit vector so cosine similarity is well-defined.
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        Ok(v.iter().map(|x| x / norm).collect())
    }

    fn dimensions(&self) -> usize {
        self.dims
    }
}

fn main() -> rustkit_semantic::Result<()> {
    let embedder = RandomEmbedder { dims: 256 };

    let mut idx = SemanticIndex::builder()
        .embedder(embedder)
        .threshold(0.50)
        .build()?;

    idx.insert_batch([
        ("custom embedder — first entry",  json!({ "id": 1 })),
        ("custom embedder — second entry", json!({ "id": 2 })),
        ("something completely different", json!({ "id": 3 })),
    ])?;

    let results = idx.search("custom embedder", 3)?;
    println!("Results for \"custom embedder\":");
    for r in &results {
        println!("  [{:.3}] {} (metadata: {})", r.score, r.content, r.metadata);
    }

    Ok(())
}
