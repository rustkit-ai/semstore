use anyhow::Result;
use rustkit_semantic::cache::SemanticCache;

fn main() -> Result<()> {
    let mut cache = SemanticCache::new("cache.db", 0.92)?;

    // Store a response
    cache.set(
        "How does async/await work in Rust?",
        "Async/await in Rust is built on top of futures...",
        "claude-sonnet-4-6",
        0.003,
    )?;

    // Query the cache
    let hit = cache.get("Explain async await in Rust")?;
    match hit {
        Some(response) => println!("Cache hit: {}", response),
        None => println!("Cache miss — forward to LLM"),
    }

    Ok(())
}
