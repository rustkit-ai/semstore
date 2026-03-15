# rustkit-semantic

[![Crates.io](https://img.shields.io/crates/v/rustkit-semantic.svg)](https://crates.io/crates/rustkit-semantic)
[![Docs.rs](https://docs.rs/rustkit-semantic/badge.svg)](https://docs.rs/rustkit-semantic)
[![CI](https://github.com/rustkit-ai/rustkit-semantic/actions/workflows/ci.yml/badge.svg)](https://github.com/rustkit-ai/rustkit-semantic/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Local semantic search for Rust applications — store text, search by meaning.

**No cloud. No API key. Everything runs on-device.**

---

## What it does

`rustkit-semantic` gives your Rust application a semantic search index backed by:

- **BGE-Small-EN-v1.5** — a 23 MB ONNX embedding model that runs on CPU
- **HNSW** — approximate nearest-neighbour search via [usearch](https://github.com/unum-cloud/usearch)
- **SQLite** — persistent storage for entries and embeddings (survives restarts)

Typical use cases:

| Use case | What it solves |
|----------|----------------|
| **RAG** | Retrieve relevant context before calling an LLM |
| **Semantic cache** | Avoid redundant LLM calls for similar questions |
| **Knowledge base** | Search docs / notes / code by meaning |
| **Deduplication** | Find near-duplicate entries in a dataset |

---

## Installation

```toml
[dependencies]
rustkit-semantic = "0.1"
serde_json = "1"
```

The BGE-Small model (~23 MB) is downloaded from HuggingFace on first use and cached locally.

### Feature flags

| Feature | Default | Description |
|---------|---------|-------------|
| `default-embedder` | ✓ | Includes BGE-Small-EN-v1.5 via fastembed |
| `bundled-sqlite` | ✓ | Statically links SQLite (no system library required) |

Disable `default-embedder` if you want to bring your own embedding model:

```toml
rustkit-semantic = { version = "0.1", default-features = false, features = ["bundled-sqlite"] }
```

---

## Quick start

```rust
use rustkit_semantic::SemanticIndex;
use serde_json::json;

fn main() -> rustkit_semantic::Result<()> {
    // Persistent index stored in a SQLite file
    let mut idx = SemanticIndex::open("./index.db")?;

    idx.insert("Rust ownership prevents data races at compile time",
               json!({ "lang": "rust", "topic": "memory" }))?;

    idx.insert("Python uses reference counting for memory management",
               json!({ "lang": "python", "topic": "memory" }))?;

    let results = idx.search("memory safety", 5)?;

    for r in &results {
        println!("[{:.2}] {}", r.score, r.content);
    }

    Ok(())
}
```

Output:
```
[0.87] Rust ownership prevents data races at compile time
[0.74] Python uses reference counting for memory management
```

---

## RAG pattern

```rust
use rustkit_semantic::SemanticIndex;
use serde_json::json;

fn build_prompt(question: &str, idx: &SemanticIndex) -> rustkit_semantic::Result<String> {
    let context = idx.search(question, 3)?;

    let context_block = context
        .iter()
        .map(|r| format!("- {}", r.content))
        .collect::<Vec<_>>()
        .join("\n");

    Ok(format!(
        "Context:\n{context_block}\n\nQuestion: {question}\nAnswer:"
    ))
}
```

---

## Custom embedder

Implement the [`Embedder`] trait to use any model — OpenAI, Cohere, a local
model, or anything else:

```rust
use rustkit_semantic::{Embedder, Error, SemanticIndex};

struct OpenAiEmbedder {
    client: /* your HTTP client */,
}

impl Embedder for OpenAiEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>, Error> {
        // POST to https://api.openai.com/v1/embeddings
        todo!()
    }

    fn dimensions(&self) -> usize {
        1536 // text-embedding-3-small
    }
}

let mut idx = SemanticIndex::builder()
    .embedder(OpenAiEmbedder { /* … */ })
    .path("./index.db")
    .threshold(0.80)
    .build()?;
```

---

## API overview

```rust
// Constructors
SemanticIndex::open("./index.db")?          // persistent
SemanticIndex::in_memory()?                 // ephemeral (tests)
SemanticIndex::builder()                    // full configuration
    .path("./index.db")
    .threshold(0.75)                        // minimum similarity score
    .embedder(my_embedder)                  // custom model
    .build()?

// Write
idx.insert("content", json!({ "key": "value" }))?   // → u64 id
idx.insert_batch([("a", json!({})), ("b", json!({}))])?  // → Vec<u64>
idx.remove(id)?                                     // → bool

// Read
idx.search("query", limit)?   // → Vec<SearchResult> sorted by score
idx.len()                     // → usize
idx.is_empty()                // → bool
idx.stats()?                  // → Stats { total: u64 }

// SearchResult fields
r.id        // u64
r.content   // String
r.metadata  // serde_json::Value
r.score     // f32 in [0.0, 1.0]
```

---

## Performance

| Operation | Typical latency (CPU) |
|-----------|----------------------|
| First load (model init) | ~1–2 s |
| `embed()` — single text | ~5 ms |
| `insert()` | ~6 ms |
| `search()` — 10k entries | ~6 ms |
| `search()` — cache hit (exact) | ~5 ms |

Benchmarked on Apple M2. Results vary by hardware and text length.

---

## Examples

```bash
cargo run --example basic            # in-memory index, search by meaning
cargo run --example rag              # retrieve context before LLM call
cargo run --example custom_embedder  # plug in your own model
```

---

## Contributing

Issues and pull requests are welcome.

```bash
cargo test          # run tests
cargo clippy        # lint
cargo fmt           # format
```

---

## License

MIT — see [LICENSE](LICENSE).
