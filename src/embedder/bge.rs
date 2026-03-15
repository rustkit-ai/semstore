//! BGE-Small-EN-v1.5 embedder using ONNX Runtime via fastembed.
//!
//! The model (~23 MB) is downloaded from HuggingFace on first use
//! and cached in the system's cache directory.

use std::sync::Arc;

use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use once_cell::sync::OnceCell;

use crate::error::{Error, Result};

use super::Embedder;

// Load the model once across the whole process — instantiating it several
// times would waste memory and trigger redundant downloads.
static MODEL: OnceCell<Arc<TextEmbedding>> = OnceCell::new();

fn shared_model() -> Result<Arc<TextEmbedding>> {
    MODEL
        .get_or_try_init(|| {
            let model = TextEmbedding::try_new(
                InitOptions::new(EmbeddingModel::BGESmallENV15).with_show_download_progress(true),
            )
            .map_err(|e| Error::Embed(e.to_string()))?;
            Ok(Arc::new(model))
        })
        .map(Arc::clone)
}

/// BGE-Small-EN-v1.5 — a fast, accurate 384-dimensional English embedder.
///
/// Uses ONNX Runtime locally; no API key or network connection required
/// after the first use (model is cached on disk).
///
/// This is the default embedder used when you call
/// [`SemanticIndex::open`](crate::SemanticIndex::open) or
/// [`SemanticIndex::in_memory`](crate::SemanticIndex::in_memory).
#[derive(Clone)]
pub struct BgeEmbedder {
    model: Arc<TextEmbedding>,
}

impl BgeEmbedder {
    /// Load (or reuse) the BGE-Small-EN-v1.5 model.
    ///
    /// # Errors
    ///
    /// Returns an error if the model cannot be downloaded or initialised.
    pub fn new() -> Result<Self> {
        Ok(Self {
            model: shared_model()?,
        })
    }
}

impl Embedder for BgeEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut out = self
            .model
            .embed(vec![text], None)
            .map_err(|e| Error::Embed(e.to_string()))?;
        Ok(out.remove(0))
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.model
            .embed(texts.to_vec(), None)
            .map_err(|e| Error::Embed(e.to_string()))
    }

    fn dimensions(&self) -> usize {
        384
    }
}
