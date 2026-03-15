//! Basic usage: in-memory index, insert documents, search by meaning.
//!
//! ```
//! cargo run --example basic
//! ```

use rustkit_semantic::SemanticIndex;
use serde_json::json;

fn main() -> rustkit_semantic::Result<()> {
    // In-memory index — nothing is written to disk.
    let mut idx = SemanticIndex::in_memory()?;

    idx.insert_batch([
        ("Rust ownership prevents data races at compile time",   json!({ "lang": "rust" })),
        ("Rust borrowing: multiple readers OR one writer",       json!({ "lang": "rust" })),
        ("Python uses reference counting for memory management", json!({ "lang": "python" })),
        ("Go's garbage collector runs concurrently",             json!({ "lang": "go" })),
        ("async/await in Rust compiles to zero-cost state machines", json!({ "lang": "rust" })),
        ("Tokio is the most popular async runtime for Rust",    json!({ "lang": "rust" })),
    ])?;

    let stats = idx.stats()?;
    println!("Indexed {} entries\n", stats.total);

    for query in &[
        "how does Rust handle memory safety?",
        "concurrent programming",
        "async Rust",
    ] {
        println!("Query: \"{query}\"");
        let results = idx.search(query, 2)?;
        if results.is_empty() {
            println!("  (no results above threshold)");
        } else {
            for r in &results {
                println!("  [{:.3}] {}", r.score, r.content);
            }
        }
        println!();
    }

    Ok(())
}
