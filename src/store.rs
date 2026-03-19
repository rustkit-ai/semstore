//! SQLite-backed persistence layer.

use std::path::Path;

use rusqlite::{params, Connection};
use serde_json::Value;

use crate::error::Result;

/// A persisted entry in the store.
pub(crate) struct Row {
    pub id: u64,
    pub content: String,
    pub metadata: Value,
    pub embedding: Vec<f32>,
}

pub(crate) struct Store {
    conn: Connection,
}

impl Store {
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open(path)?;
        Self::init(conn)
    }

    pub fn in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        Self::init(conn)
    }

    fn init(conn: Connection) -> Result<Self> {
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA synchronous = NORMAL;
             CREATE TABLE IF NOT EXISTS entries (
                 id        INTEGER PRIMARY KEY,
                 content   TEXT    NOT NULL,
                 metadata  TEXT    NOT NULL DEFAULT '{}',
                 embedding BLOB    NOT NULL,
                 inserted_at TEXT  NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
             );",
        )?;
        Ok(Self { conn })
    }

    /// Insert a new entry and return its id.
    pub fn insert(&self, content: &str, metadata: &Value, embedding: &[f32]) -> Result<u64> {
        let id = self.next_id()?;
        self.conn.execute(
            "INSERT INTO entries (id, content, metadata, embedding) VALUES (?1, ?2, ?3, ?4)",
            params![
                id as i64,
                content,
                metadata.to_string(),
                floats_to_bytes(embedding)
            ],
        )?;
        Ok(id)
    }

    /// Fetch a single entry by id.
    pub fn get(&self, id: u64) -> Result<Option<Row>> {
        let result = self.conn.query_row(
            "SELECT id, content, metadata, embedding FROM entries WHERE id = ?1",
            params![id as i64],
            parse_row,
        );
        match result {
            Ok(row) => Ok(Some(row)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Delete an entry. Returns `true` if a row was removed.
    pub fn delete(&self, id: u64) -> Result<bool> {
        let n = self
            .conn
            .execute("DELETE FROM entries WHERE id = ?1", params![id as i64])?;
        Ok(n > 0)
    }

    /// Load all rows — used to rebuild the HNSW index after a restart.
    pub fn load_all(&self) -> Result<Vec<Row>> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, content, metadata, embedding FROM entries ORDER BY id")?;

        let rows = stmt
            .query_map([], parse_row)?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(rows)
    }

    pub fn count(&self) -> Result<u64> {
        let n: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM entries", [], |r| r.get(0))?;
        Ok(n as u64)
    }

    /// Return ids of entries inserted more than `seconds` ago.
    ///
    /// Uses the DB-managed `inserted_at` column, not the JSON metadata.
    pub fn scan_ids_older_than(&self, seconds: u64) -> Result<Vec<u64>> {
        let mut stmt = self.conn.prepare(
            "SELECT id FROM entries
             WHERE (unixepoch('now') - unixepoch(inserted_at)) >= ?1",
        )?;
        let ids = stmt
            .query_map(params![seconds as i64], |r| r.get::<_, i64>(0))?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(ids.into_iter().map(|n| n as u64).collect())
    }

    fn next_id(&self) -> Result<u64> {
        let max: Option<i64> = self
            .conn
            .query_row("SELECT MAX(id) FROM entries", [], |r| r.get(0))?;
        Ok(max.map(|n| n as u64 + 1).unwrap_or(0))
    }
}

fn parse_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<Row> {
    let id = row.get::<_, i64>(0)? as u64;
    let content: String = row.get(1)?;
    let raw_meta: String = row.get(2)?;
    let emb_bytes: Vec<u8> = row.get(3)?;

    let metadata = serde_json::from_str(&raw_meta).unwrap_or(Value::Object(Default::default()));
    let embedding = bytes_to_floats(&emb_bytes);

    Ok(Row {
        id,
        content,
        metadata,
        embedding,
    })
}

fn floats_to_bytes(floats: &[f32]) -> Vec<u8> {
    floats.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn bytes_to_floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}
