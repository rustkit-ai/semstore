use anyhow::Result;
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

use crate::{embedder::Embedder, store::Store};

pub struct SemanticCache {
    embedder: Embedder,
    index: Index,
    store: Store,
    threshold: f32,
}

impl SemanticCache {
    pub fn new(db_path: &str, threshold: f32) -> Result<Self> {
        let embedder = Embedder::new()?;
        let index = Index::new(&IndexOptions {
            dimensions: 384,
            metric: MetricKind::Cos,
            quantization: ScalarKind::F16,
            ..Default::default()
        })?;
        let store = Store::new(db_path)?;

        Ok(Self { embedder, index, store, threshold })
    }

    /// Returns cached response if a semantically similar query exists.
    pub fn get(&self, query: &str) -> Result<Option<String>> {
        if self.index.size() == 0 {
            return Ok(None);
        }

        let embedding = self.embedder.embed(query)?;
        let results = self.index.search(&embedding, 1)?;

        if results.keys.is_empty() {
            return Ok(None);
        }

        // usearch cosine distance: 0 = identical, 1 = opposite
        let distance = results.distances[0];
        if distance <= (1.0 - self.threshold) {
            self.store.get_by_id(results.keys[0])
        } else {
            Ok(None)
        }
    }

    /// Stores a new query/response pair in the cache.
    pub fn set(&mut self, query: &str, response: &str, model: &str, cost_usd: f32) -> Result<()> {
        let embedding = self.embedder.embed(query)?;
        let id = self.store.count()?;

        self.index.add(id, &embedding)?;
        self.store.insert(id, query, response, model, cost_usd)?;

        Ok(())
    }
}
