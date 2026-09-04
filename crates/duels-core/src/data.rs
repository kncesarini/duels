//! Static game data loading (cards, wonders, tokens).
//!
//! The repo root `data/` directory holds the factual base-game data
//! (`cards.json`, `wonders.json`, `tokens.json`, `military.json`) as the
//! source of truth, documented in `data/README.md`. That data is a
//! best-effort transcription pending a spot-check against the physical
//! rulebook/BoardGameGeek (see the data README) and is not yet consumed by
//! any code.
//!
//! M1 will add strongly-typed structs here (`Card`, `Wonder`,
//! `ProgressToken`, `MilitaryToken`, ...), parse the JSON files at build
//! time or startup, validate cross-references (chain symbols, guild/age
//! counts), and expose `'static` lookup tables keyed by id for the engine
//! and scoring modules to use.
//!
//! M0 stub only — no loading or types yet.
