//! Embedder trait and built-in implementations.
//!
//! Implement [`Embedder`] to use any embedding model (OpenAI, Cohere, local, etc.)
//! with [`SemanticIndex`](crate::SemanticIndex).

use crate::error::Result;

#[cfg(feature = "default-embedder")]
mod bge;

#[cfg(feature = "default-embedder")]
pub use bge::BgeEmbedder;

/// Converts text into fixed-size embedding vectors.
///
/// Implement this trait to use a custom embedding model with [`SemanticIndex`](crate::SemanticIndex).
///
/// # Example — OpenAI-compatible embedder
///
/// ```rust,no_run
/// use semstore::{Embedder, Error};
///
/// struct MyEmbedder;
///
/// impl Embedder for MyEmbedder {
///     fn embed(&self, text: &str) -> Result<Vec<f32>, Error> {
///         // call your embedding API here
///         Ok(vec![0.0; 1536])
///     }
///
///     fn dimensions(&self) -> usize {
///         1536
///     }
/// }
/// ```
pub trait Embedder: Send {
    /// Embed a single piece of text into a float vector.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Embed`](crate::Error::Embed) if the model fails.
    fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Embed multiple texts in one call.
    ///
    /// The default implementation calls [`embed`](Self::embed) in a loop.
    /// Override for batch-optimised models.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Embed`](crate::Error::Embed) on the first failure.
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    /// Number of dimensions produced by this embedder.
    fn dimensions(&self) -> usize;
}
