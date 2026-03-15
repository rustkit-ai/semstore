use anyhow::Result;
use rusqlite::{Connection, params};

pub struct Store {
    conn: Connection,
}

impl Store {
    pub fn new(path: &str) -> Result<Self> {
        let conn = Connection::open(path)?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS cache (
                id      INTEGER PRIMARY KEY,
                query   TEXT NOT NULL,
                response TEXT NOT NULL,
                model   TEXT NOT NULL,
                cost_usd REAL DEFAULT 0.0,
                created_at TEXT DEFAULT (datetime('now'))
            );",
        )?;
        Ok(Self { conn })
    }

    pub fn insert(&self, id: u64, query: &str, response: &str, model: &str, cost_usd: f32) -> Result<()> {
        self.conn.execute(
            "INSERT INTO cache (id, query, response, model, cost_usd) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![id as i64, query, response, model, cost_usd],
        )?;
        Ok(())
    }

    pub fn get_by_id(&self, id: u64) -> Result<Option<String>> {
        let result = self.conn.query_row(
            "SELECT response FROM cache WHERE id = ?1",
            params![id as i64],
            |row| row.get(0),
        );
        match result {
            Ok(r) => Ok(Some(r)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    pub fn count(&self) -> Result<u64> {
        let n: i64 = self.conn.query_row("SELECT COUNT(*) FROM cache", [], |r| r.get(0))?;
        Ok(n as u64)
    }
}
