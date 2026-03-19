//! Main [`SemanticIndex`] struct and its builder.

use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

use crate::{
    embedder::Embedder,
    error::{Error, Result},
    store::Store,
};

#[cfg(feature = "default-embedder")]
use crate::embedder::BgeEmbedder;

// ── Public types ─────────────────────────────────────────────────────────────

/// A single result returned by [`SemanticIndex::search`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[must_use]
pub struct SearchResult {
    /// Unique id assigned when the entry was inserted.
    pub id: u64,
    /// The original stored text.
    pub content: String,
    /// JSON metadata provided at insertion time.
    pub metadata: Value,
    /// Cosine similarity in `[0.0, 1.0]`. Higher → more similar.
    pub score: f32,
}

/// Statistics about a [`SemanticIndex`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stats {
    /// Total number of stored entries.
    pub total: u64,
}

// ── Builder ───────────────────────────────────────────────────────────────────

/// Builder for [`SemanticIndex`].
///
/// Obtain one via [`SemanticIndex::builder`].
pub struct Builder<E> {
    embedder: E,
    db_path: Option<std::path::PathBuf>,
    threshold: f32,
}

#[cfg(feature = "default-embedder")]
impl Builder<()> {
    pub(crate) fn new_default() -> Builder<()> {
        Builder {
            embedder: (),
            db_path: None,
            threshold: 0.75,
        }
    }
}

impl<E> Builder<E> {
    /// Persist the index to `path` (a SQLite file).
    ///
    /// If not called, the index lives only in memory and is lost on drop.
    pub fn path(mut self, path: impl AsRef<Path>) -> Self {
        self.db_path = Some(path.as_ref().to_owned());
        self
    }

    /// Minimum similarity score `[0.0, 1.0]` required for a result to be
    /// returned from [`search`](SemanticIndex::search).
    ///
    /// Defaults to `0.75`.
    pub fn threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold.clamp(0.0, 1.0);
        self
    }
}

impl<E: Embedder + 'static> Builder<E> {
    /// Use a custom embedder.
    pub fn embedder<F: Embedder + 'static>(self, embedder: F) -> Builder<F> {
        Builder {
            embedder,
            db_path: self.db_path,
            threshold: self.threshold,
        }
    }

    /// Build the [`SemanticIndex`].
    ///
    /// # Errors
    ///
    /// Returns an error if the embedder or the store cannot be initialised.
    pub fn build(self) -> Result<SemanticIndex> {
        build_index(
            Box::new(self.embedder),
            self.db_path.as_deref(),
            self.threshold,
        )
    }
}

#[cfg(feature = "default-embedder")]
impl Builder<()> {
    /// Build the [`SemanticIndex`] with the default BGE-Small embedder.
    ///
    /// # Errors
    ///
    /// Returns an error if the model cannot be loaded or the store fails.
    pub fn build(self) -> Result<SemanticIndex> {
        let embedder = BgeEmbedder::new()?;
        build_index(Box::new(embedder), self.db_path.as_deref(), self.threshold)
    }

    /// Use a custom embedder instead of the default BGE-Small model.
    pub fn embedder<E: Embedder + 'static>(self, embedder: E) -> Builder<E> {
        Builder {
            embedder,
            db_path: self.db_path,
            threshold: self.threshold,
        }
    }
}

// ── SemanticIndex ─────────────────────────────────────────────────────────────

/// A local semantic index — store text with metadata and search by meaning.
///
/// Uses ONNX embeddings (BGE-Small by default) and an HNSW vector index
/// backed by SQLite for persistence. Everything runs locally; no API keys
/// or network access are required after the embedding model is cached.
///
/// # Quick start
///
/// ```no_run
/// use semstore::SemanticIndex;
/// use serde_json::json;
///
/// // Persistent index (SQLite file)
/// let mut idx = SemanticIndex::open("./index.db")?;
///
/// idx.insert("Rust ownership prevents data races at compile time",
///            json!({ "lang": "rust" }))?;
///
/// let results = idx.search("memory safety", 5)?;
/// for r in &results {
///     println!("[{:.2}] {}", r.score, r.content);
/// }
/// # Ok::<(), semstore::Error>(())
/// ```
///
/// # Custom embedder
///
/// ```no_run
/// use semstore::{Embedder, Error, SemanticIndex};
///
/// struct MyEmbedder;
/// impl Embedder for MyEmbedder {
///     fn embed(&self, _text: &str) -> Result<Vec<f32>, Error> { Ok(vec![0.0; 512]) }
///     fn dimensions(&self) -> usize { 512 }
/// }
///
/// let mut idx = SemanticIndex::builder()
///     .embedder(MyEmbedder)
///     .path("./index.db")
///     .build()?;
/// # Ok::<(), semstore::Error>(())
/// ```
pub struct SemanticIndex {
    embedder: Box<dyn Embedder>,
    index: Index,
    store: Store,
    threshold: f32,
}

#[cfg(feature = "default-embedder")]
impl SemanticIndex {
    /// Open a persistent index stored at `path`.
    ///
    /// Creates the file if it does not exist.
    /// Uses the default BGE-Small-EN-v1.5 embedder.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened or the model fails to load.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let embedder = BgeEmbedder::new()?;
        build_index(Box::new(embedder), Some(path.as_ref()), 0.75)
    }

    /// Create an ephemeral in-memory index (not persisted).
    ///
    /// Useful in tests and benchmarks.
    ///
    /// # Errors
    ///
    /// Returns an error if the model fails to load.
    pub fn in_memory() -> Result<Self> {
        let embedder = BgeEmbedder::new()?;
        build_index(Box::new(embedder), None, 0.75)
    }

    /// Create a [`Builder`] for fine-grained configuration.
    #[must_use]
    pub fn builder() -> Builder<()> {
        Builder::new_default()
    }
}

impl SemanticIndex {
    /// Insert one entry and return its assigned id.
    ///
    /// # Errors
    ///
    /// Returns an error if embedding or the SQLite write fails.
    pub fn insert(&mut self, content: &str, metadata: Value) -> Result<u64> {
        let embedding = self.embedder.embed(content)?;
        self.index
            .reserve(self.index.size() + 1)
            .map_err(|e| Error::Index(e.to_string()))?;
        let id = self.store.insert(content, &metadata, &embedding)?;
        self.index
            .add(id, &embedding)
            .map_err(|e| Error::Index(e.to_string()))?;
        Ok(id)
    }

    /// Insert multiple entries in one call (single embedding pass).
    ///
    /// More efficient than calling [`insert`](Self::insert) in a loop
    /// when your embedder supports batching.
    ///
    /// # Errors
    ///
    /// Returns an error on the first embedding or write failure.
    pub fn insert_batch(
        &mut self,
        entries: impl IntoIterator<Item = (impl AsRef<str>, Value)>,
    ) -> Result<Vec<u64>> {
        let entries: Vec<(String, Value)> = entries
            .into_iter()
            .map(|(s, v)| (s.as_ref().to_owned(), v))
            .collect();

        let texts: Vec<&str> = entries.iter().map(|(s, _)| s.as_str()).collect();
        let embeddings = self.embedder.embed_batch(&texts)?;

        self.index
            .reserve(self.index.size() + entries.len())
            .map_err(|e| Error::Index(e.to_string()))?;

        let mut ids = Vec::with_capacity(entries.len());
        for ((content, metadata), embedding) in entries.iter().zip(&embeddings) {
            let id = self.store.insert(content, metadata, embedding)?;
            self.index
                .add(id, embedding)
                .map_err(|e| Error::Index(e.to_string()))?;
            ids.push(id);
        }
        Ok(ids)
    }

    /// Return the `limit` most similar entries to `query` that score above the
    /// configured threshold, sorted by descending similarity.
    ///
    /// Returns an empty `Vec` when the index is empty.
    ///
    /// # Errors
    ///
    /// Returns an error if embedding `query` fails or the vector search fails.
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        if self.index.size() == 0 || limit == 0 {
            return Ok(Vec::new());
        }

        let embedding = self.embedder.embed(query)?;
        let k = limit.min(self.index.size());

        let raw = self
            .index
            .search(&embedding, k)
            .map_err(|e| Error::Index(e.to_string()))?;

        let mut results = Vec::new();
        for (&key, &distance) in raw.keys.iter().zip(&raw.distances) {
            // usearch reports cosine *distance* (0 = identical, 1 = opposite).
            let score = 1.0 - distance;
            if score < self.threshold {
                continue;
            }
            if let Some(row) = self.store.get(key)? {
                results.push(SearchResult {
                    id: row.id,
                    content: row.content,
                    metadata: row.metadata,
                    score,
                });
            }
        }

        results.sort_unstable_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(results)
    }

    /// Remove an entry by id.
    ///
    /// Returns `true` if the entry existed and was removed.
    ///
    /// # Errors
    ///
    /// Returns an error if the vector index or the store fails.
    pub fn remove(&mut self, id: u64) -> Result<bool> {
        self.index
            .remove(id)
            .map_err(|e| Error::Index(e.to_string()))?;
        self.store.delete(id)
    }

    /// Return the number of entries in the index.
    pub fn len(&self) -> usize {
        self.index.size()
    }

    /// Returns `true` if the index contains no entries.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return statistics about the index.
    ///
    /// # Errors
    ///
    /// Returns an error if the store query fails.
    pub fn stats(&self) -> Result<Stats> {
        Ok(Stats {
            total: self.store.count()?,
        })
    }

    /// Remove all entries inserted more than `seconds` ago.
    ///
    /// Returns the number of entries removed.
    ///
    /// # Errors
    ///
    /// Returns an error if the store query or delete fails.
    pub fn remove_older_than(&mut self, seconds: u64) -> Result<usize> {
        let ids = self.store.scan_ids_older_than(seconds)?;
        let count = ids.len();
        for &id in &ids {
            // Non-fatal: the vector may already have been removed
            let _ = self.index.remove(id);
            self.store.delete(id)?;
        }
        Ok(count)
    }
}

impl SemanticIndex {
    /// Return the number of stored entries at `path` without loading the
    /// embedding model — useful for stats/CLI commands.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened.
    pub fn entry_count(path: &Path) -> Result<u64> {
        let store = Store::open(path)?;
        store.count()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(all(test, feature = "default-embedder"))]
mod tests {
    use serde_json::json;

    use super::*;

    fn idx() -> SemanticIndex {
        SemanticIndex::builder().threshold(0.60).build().unwrap()
    }

    #[test]
    #[ignore = "requires fastembed model download (~23 MB) — run with --ignored locally"]
    fn empty_search_returns_nothing() {
        let idx = idx();
        assert!(idx.search("anything", 5).unwrap().is_empty());
    }

    #[test]
    #[ignore = "requires fastembed model download (~23 MB) — run with --ignored locally"]
    fn insert_and_search() {
        let mut idx = idx();
        idx.insert("Rust prevents memory errors at compile time", json!({}))
            .unwrap();
        idx.insert("Python is a high-level scripting language", json!({}))
            .unwrap();

        let results = idx.search("memory safety rust", 2).unwrap();
        assert!(!results.is_empty());
        assert!(results[0].score >= 0.60);
        assert!(results[0].content.contains("Rust"));
    }

    #[test]
    #[ignore = "requires fastembed model download (~23 MB) — run with --ignored locally"]
    fn stats_tracks_inserts() {
        let mut idx = idx();
        assert_eq!(idx.stats().unwrap().total, 0);
        idx.insert("hello", json!({})).unwrap();
        assert_eq!(idx.stats().unwrap().total, 1);
    }

    #[test]
    #[ignore = "requires fastembed model download (~23 MB) — run with --ignored locally"]
    fn len_and_is_empty() {
        let mut idx = idx();
        assert!(idx.is_empty());
        idx.insert("hello", json!({})).unwrap();
        assert_eq!(idx.len(), 1);
        assert!(!idx.is_empty());
    }

    #[test]
    #[ignore = "requires fastembed model download (~23 MB) — run with --ignored locally"]
    fn remove_entry() {
        let mut idx = idx();
        let id = idx.insert("to be deleted", json!({})).unwrap();
        assert!(idx.remove(id).unwrap());
        assert!(idx.search("to be deleted", 5).unwrap().is_empty());
    }

    #[test]
    #[ignore = "requires fastembed model download (~23 MB) — run with --ignored locally"]
    fn insert_batch() {
        let mut idx = idx();
        let ids = idx
            .insert_batch([
                ("first entry", json!({ "n": 1 })),
                ("second entry", json!({ "n": 2 })),
                ("third entry", json!({ "n": 3 })),
            ])
            .unwrap();
        assert_eq!(ids.len(), 3);
        assert_eq!(idx.stats().unwrap().total, 3);
    }

    #[test]
    #[ignore = "requires fastembed model download (~23 MB) — run with --ignored locally"]
    fn metadata_round_trips() {
        let mut idx = idx();
        idx.insert("some text", json!({ "source": "test.md", "page": 42 }))
            .unwrap();
        let results = idx.search("some text", 1).unwrap();
        assert_eq!(results[0].metadata["source"], "test.md");
        assert_eq!(results[0].metadata["page"], 42);
    }

    #[test]
    #[ignore = "requires fastembed model download (~23 MB) — run with --ignored locally"]
    fn results_sorted_descending() {
        let mut idx = idx();
        idx.insert_batch([
            ("Rust async/await with tokio", json!({})),
            ("async await syntax in Rust", json!({})),
            ("Go goroutines not async/await", json!({})),
        ])
        .unwrap();

        let results = idx.search("async Rust", 3).unwrap();
        for w in results.windows(2) {
            assert!(
                w[0].score >= w[1].score,
                "results must be sorted by score descending"
            );
        }
    }

    #[test]
    #[ignore = "requires fastembed model download (~23 MB) — run with --ignored locally"]
    fn persistence_rebuilds_index() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_owned();

        {
            let mut idx = SemanticIndex::open(&path).unwrap();
            idx.insert("persisted content", json!({ "x": 1 })).unwrap();
        }

        // Re-open: HNSW is rebuilt from the SQLite BLOB.
        let idx = SemanticIndex::open(&path).unwrap();
        let results = idx.search("persisted content", 1).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].metadata["x"], 1);
    }
}

// ── Internal constructor ──────────────────────────────────────────────────────

fn build_index(
    embedder: Box<dyn Embedder>,
    db_path: Option<&Path>,
    threshold: f32,
) -> Result<SemanticIndex> {
    let dims = embedder.dimensions();

    let index = Index::new(&IndexOptions {
        dimensions: dims,
        metric: MetricKind::Cos,
        quantization: ScalarKind::F16,
        ..Default::default()
    })
    .map_err(|e| Error::Index(e.to_string()))?;

    let store = match db_path {
        Some(path) => Store::open(path)?,
        None => Store::in_memory()?,
    };

    // Rebuild the in-memory HNSW index from the persisted embeddings.
    let existing = store.load_all()?;
    if !existing.is_empty() {
        index
            .reserve(existing.len())
            .map_err(|e| Error::Index(e.to_string()))?;
        for row in &existing {
            index
                .add(row.id, &row.embedding)
                .map_err(|e| Error::Index(e.to_string()))?;
        }
    }

    Ok(SemanticIndex {
        embedder,
        index,
        store,
        threshold,
    })
}
