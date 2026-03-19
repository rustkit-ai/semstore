<h1 align="center">semstore</h1>

<p align="center">
  Local semantic search for Rust applications — store text, search by meaning, no cloud required.
</p>

<p align="center">
  <a href="https://github.com/rustkit-ai/semstore/actions/workflows/ci.yml"><img src="https://github.com/rustkit-ai/semstore/actions/workflows/ci.yml/badge.svg" alt="CI"/></a>
  <a href="https://crates.io/crates/semstore"><img src="https://img.shields.io/crates/v/semstore.svg" alt="crates.io"/></a>
  <a href="https://docs.rs/semstore"><img src="https://docs.rs/semstore/badge.svg" alt="docs.rs"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"/></a>
</p>

---

**Semantic search in Rust applications is harder than it should be.** You either wire up a cloud embedding API (latency, cost, data leaving your machine), run a separate vector database process, or write the plumbing yourself. None of those are reasonable for a library dependency.

`semstore` is a self-contained semantic index you embed directly in your Rust binary. One struct, four methods, zero infrastructure.

```rust
let mut idx = SemanticIndex::open("./index.db")?;

idx.insert("Rust ownership prevents data races at compile time", json!({ "lang": "rust" }))?;
idx.insert("Python uses reference counting for memory management", json!({ "lang": "python" }))?;

let results = idx.search("memory safety", 5)?;
// [0.87] Rust ownership prevents data races at compile time
// [0.74] Python uses reference counting for memory management
```

**No API key. No server. No Python.** The BGE-Small model (~23 MB) runs on CPU via ONNX and is cached locally after the first use.

---

## Install

```toml
[dependencies]
semstore  = "0.1"
serde_json = "1"
```

The BGE-Small model (~23 MB) is downloaded from HuggingFace on first use and cached locally.

### Feature flags

| Feature | Default | Description |
|---------|---------|-------------|
| `default-embedder` | ✓ | Bundles BGE-Small-EN-v1.5 via fastembed |
| `bundled-sqlite` | ✓ | Statically links SQLite (no system library required) |

Bring your own embedder by disabling `default-embedder`:

```toml
semstore = { version = "0.1", default-features = false, features = ["bundled-sqlite"] }
```

---

## What's inside

- **BGE-Small-EN-v1.5** — 23 MB ONNX embedding model, runs on CPU
- **HNSW** — approximate nearest-neighbour search via [usearch](https://github.com/unum-cloud/usearch)
- **SQLite** — persistent storage for entries and embeddings (survives restarts)

| Use case | What it solves |
|---|---|
| **RAG** | Retrieve relevant context before calling an LLM |
| **Semantic cache** | Avoid redundant LLM calls for similar questions |
| **Knowledge base** | Search docs, notes, code by meaning |
| **Deduplication** | Find near-duplicate entries in a dataset |

---

## Quick start

```rust
use semstore::SemanticIndex;
use serde_json::json;

fn main() -> semstore::Result<()> {
    let mut idx = SemanticIndex::open("./index.db")?;

    idx.insert("Rust ownership prevents data races at compile time",
               json!({ "lang": "rust", "topic": "memory" }))?;
    idx.insert("Python uses reference counting for memory management",
               json!({ "lang": "python", "topic": "memory" }))?;

    for r in idx.search("memory safety", 5)? {
        println!("[{:.2}] {}", r.score, r.content);
    }
    Ok(())
}
```

```
[0.87] Rust ownership prevents data races at compile time
[0.74] Python uses reference counting for memory management
```

---

## RAG pattern

```rust
fn build_prompt(question: &str, idx: &SemanticIndex) -> semstore::Result<String> {
    let context = idx.search(question, 3)?
        .iter()
        .map(|r| format!("- {}", r.content))
        .collect::<Vec<_>>()
        .join("\n");

    Ok(format!("Context:\n{context}\n\nQuestion: {question}\nAnswer:"))
}
```

---

## Custom embedder

```rust
use semstore::{Embedder, Error, SemanticIndex};

struct OpenAiEmbedder { /* your HTTP client */ }

impl Embedder for OpenAiEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>, Error> {
        // POST to https://api.openai.com/v1/embeddings
        todo!()
    }
    fn dimensions(&self) -> usize { 1536 } // text-embedding-3-small
}

let mut idx = SemanticIndex::builder()
    .embedder(OpenAiEmbedder { /* … */ })
    .path("./index.db")
    .threshold(0.80)
    .build()?;
```

---

## API

```rust
// Constructors
SemanticIndex::open("./index.db")?     // persistent
SemanticIndex::in_memory()?            // ephemeral (tests)
SemanticIndex::builder()               // full configuration
    .path("./index.db")
    .threshold(0.75)
    .embedder(my_embedder)
    .build()?

// Write
idx.insert("content", json!({ "key": "value" }))?        // → u64
idx.insert_batch([("a", json!({})), ("b", json!({}))])?  // → Vec<u64>
idx.remove(id)?                                          // → bool

// Read
idx.search("query", limit)?  // → Vec<SearchResult> sorted by score
idx.len()                    // → usize
idx.stats()?                 // → Stats { total: u64 }

// SearchResult
r.id        // u64
r.content   // String
r.metadata  // serde_json::Value
r.score     // f32 in [0.0, 1.0]
```

---

## Performance

| Operation | Typical (Apple M2 CPU) |
|---|---|
| First load (model init) | ~1–2 s |
| `embed()` single text | ~5 ms |
| `insert()` | ~6 ms |
| `search()` 10k entries | ~6 ms |

---

## Examples

```bash
cargo run --example basic            # in-memory index
cargo run --example rag              # RAG context building
cargo run --example custom_embedder  # plug in your own model
```

---

## License

MIT — see [LICENSE](LICENSE).
