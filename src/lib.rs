//! # rustkit-semantic
//!
//! Local semantic search for Rust applications — store text, search by meaning.
//! No cloud required. Embeddings run on-device via ONNX (BGE-Small, ~23 MB).
//!
//! ## Features
//!
//! | Feature | Default | Description |
//! |---------|---------|-------------|
//! | `default-embedder` | ✓ | Bundles BGE-Small-EN-v1.5 via fastembed |
//! | `bundled-sqlite` | ✓ | Statically links SQLite (no system lib needed) |
//!
//! Disable `default-embedder` to bring your own model via the [`Embedder`] trait.
//!
//! ## Quick start
//!
//! ```no_run
//! use rustkit_semantic::SemanticIndex;
//! use serde_json::json;
//!
//! let mut idx = SemanticIndex::open("./index.db")?;
//!
//! idx.insert("Rust ownership prevents data races at compile time",
//!            json!({ "lang": "rust" }))?;
//! idx.insert("Python uses reference counting for memory management",
//!            json!({ "lang": "python" }))?;
//!
//! let results = idx.search("memory safety", 3)?;
//! for r in &results {
//!     println!("[{:.2}] {}", r.score, r.content);
//! }
//! # Ok::<(), rustkit_semantic::Error>(())
//! ```
//!
//! ## Custom embedder
//!
//! ```no_run
//! use rustkit_semantic::{Embedder, Error, SemanticIndex};
//! use serde_json::json;
//!
//! struct OpenAiEmbedder { /* your fields */ }
//!
//! impl Embedder for OpenAiEmbedder {
//!     fn embed(&self, text: &str) -> Result<Vec<f32>, Error> {
//!         // call OpenAI /v1/embeddings …
//!         Ok(vec![0.0; 1536])
//!     }
//!     fn dimensions(&self) -> usize { 1536 }
//! }
//!
//! let mut idx = SemanticIndex::builder()
//!     .embedder(OpenAiEmbedder { /* … */ })
//!     .path("./index.db")
//!     .build()?;
//! # Ok::<(), rustkit_semantic::Error>(())
//! ```

#![warn(missing_docs)]
#![forbid(unsafe_code)]
#![warn(clippy::all)]

mod embedder;
mod error;
mod index;
mod store;

pub use embedder::Embedder;
pub use error::{Error, Result};
pub use index::{Builder, SearchResult, SemanticIndex, Stats};

#[cfg(feature = "default-embedder")]
pub use embedder::BgeEmbedder;
