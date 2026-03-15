use thiserror::Error;

/// Errors that can occur when using [`SemanticIndex`](crate::SemanticIndex).
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    /// The embedding model failed to produce a vector.
    #[error("embedding failed: {0}")]
    Embed(String),

    /// The HNSW vector index returned an error.
    #[error("vector index error: {0}")]
    Index(String),

    /// A SQLite operation failed.
    #[error("store error: {0}")]
    Store(#[from] rusqlite::Error),

    /// A file-system operation failed.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

/// Convenience alias for `std::result::Result<T, Error>`.
pub type Result<T> = std::result::Result<T, Error>;
