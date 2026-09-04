//! Writes a match's [`crate::match_runner::GameRecord`]s to a JSON results
//! file — one array of per-game records, so downstream tooling (a
//! leaderboard, a training-data pipeline) can `serde_json::from_reader` it
//! directly.

use std::fs;
use std::io;
use std::path::Path;

use crate::match_runner::GameRecord;

/// Write `records` to `path` as pretty-printed JSON, creating any missing
/// parent directories first.
pub fn write_results(path: &Path, records: &[GameRecord]) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(records)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    fs::write(path, json)
}

/// Read back a results file written by [`write_results`] (used by tests, and
/// available to any tooling that wants to re-analyze a past run).
pub fn read_results(path: &Path) -> io::Result<Vec<GameRecord>> {
    let json = fs::read_to_string(path)?;
    serde_json::from_str(&json).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::match_runner::play_paired_match;
    use duels_agents_api::Budget;

    #[test]
    fn round_trips_through_a_results_file() {
        let dir = std::env::temp_dir().join(format!(
            "duels-arena-test-{}-{}",
            std::process::id(),
            "round_trips_through_a_results_file"
        ));
        let path = dir.join("results.json");

        let records = play_paired_match("random", "random", &[1, 2], Budget::Nodes(1)).unwrap();
        write_results(&path, &records).expect("write should create parent dirs and succeed");

        let back = read_results(&path).expect("read back what was just written");
        assert_eq!(back.len(), records.len());
        assert_eq!(back[0].seed, records[0].seed);

        let _ = fs::remove_dir_all(&dir);
    }
}
