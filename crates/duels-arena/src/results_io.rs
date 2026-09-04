//! Writes a match's [`crate::match_runner::GameRecord`]s, plus the derived
//! tally/victory-breakdown/race-exposure summary, to a JSON results file so
//! downstream tooling (a leaderboard, a training-data pipeline, a later
//! re-analysis) can `serde_json::from_reader` the whole thing back without
//! recomputing the summary from the raw records.

use std::fs;
use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::match_runner::{
    race_exposure, tally, victory_breakdown, GameRecord, MatchTally, MatchVictoryBreakdown,
    RaceExposure,
};

/// Everything a results file holds: the raw per-game records, plus every
/// summary statistic derived from them (so a reader never needs
/// `duels-arena` itself just to see the win/loss/draw count or the
/// victory-kind breakdown).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultsFile {
    /// Every game played, in the order [`crate::match_runner::play_paired_match`]
    /// returns them.
    pub records: Vec<GameRecord>,
    /// Win/loss/draw counts from "role A"'s perspective.
    pub tally: MatchTally,
    /// How each side's wins were achieved.
    pub victory_breakdown: MatchVictoryBreakdown,
    /// How many games came within reach of an instant win, regardless of who
    /// won or how.
    pub race_exposure: RaceExposure,
}

impl ResultsFile {
    /// Compute every summary statistic from `records` and bundle it with
    /// them.
    pub fn from_records(records: &[GameRecord]) -> Self {
        Self {
            records: records.to_vec(),
            tally: tally(records),
            victory_breakdown: victory_breakdown(records),
            race_exposure: race_exposure(records),
        }
    }
}

/// Write `records` (with its derived summary) to `path` as pretty-printed
/// JSON, creating any missing parent directories first.
pub fn write_results(path: &Path, records: &[GameRecord]) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let file = ResultsFile::from_records(records);
    let json = serde_json::to_string_pretty(&file)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    fs::write(path, json)
}

/// Read back a results file written by [`write_results`] (used by tests, and
/// available to any tooling that wants to re-analyze a past run).
pub fn read_results(path: &Path) -> io::Result<ResultsFile> {
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
        assert_eq!(back.records.len(), records.len());
        assert_eq!(back.records[0].seed, records[0].seed);
        assert_eq!(back.tally, tally(&records));
        assert_eq!(back.victory_breakdown, victory_breakdown(&records));
        assert_eq!(back.race_exposure, race_exposure(&records));

        let _ = fs::remove_dir_all(&dir);
    }
}
