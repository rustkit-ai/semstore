//! RAG pattern: retrieve relevant context before calling an LLM.
//!
//! ```
//! cargo run --example rag
//! ```

use semstore::SemanticIndex;
use serde_json::json;

fn main() -> semstore::Result<()> {
    let mut idx = SemanticIndex::open("./rag_demo.db")?;

    // Index your docs once (skip if already populated).
    if idx.is_empty() {
        println!("Indexing knowledge base...");
        idx.insert_batch([
            (
                "Add rustkit-semantic to [dependencies] in Cargo.toml",
                json!({ "source": "docs/install.md" }),
            ),
            (
                "SemanticIndex::open(path) creates a persistent index backed by SQLite",
                json!({ "source": "docs/api.md", "section": "open" }),
            ),
            (
                "SemanticIndex::in_memory() creates an ephemeral index — useful in tests",
                json!({ "source": "docs/api.md", "section": "in_memory" }),
            ),
            (
                "insert() embeds content locally with BGE-Small and returns a u64 id",
                json!({ "source": "docs/api.md", "section": "insert" }),
            ),
            (
                "search() returns Vec<SearchResult> sorted by cosine similarity score",
                json!({ "source": "docs/api.md", "section": "search" }),
            ),
            (
                "insert_batch() embeds multiple entries in a single ONNX forward pass",
                json!({ "source": "docs/api.md", "section": "insert_batch" }),
            ),
        ])?;
        println!("Done — {} entries indexed.\n", idx.stats()?.total);
    }

    // At query time: retrieve context, inject into the LLM prompt.
    let user_question = "how do I index multiple documents efficiently?";
    println!("User: {user_question}");

    let context = idx.search(user_question, 3)?;

    println!("\n── Context injected into LLM prompt ──");
    for r in &context {
        println!(
            "  [{:.2}] {} ({})",
            r.score,
            r.content,
            r.metadata["source"].as_str().unwrap_or("?")
        );
    }
    println!("──────────────────────────────────────");
    println!("\n// → pass context + question to your LLM call");

    // Clean up demo file.
    let _ = std::fs::remove_file("./rag_demo.db");

    Ok(())
}
