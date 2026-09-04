# 0003. Public repository

## Context

This project has no proprietary business logic or confidential data — it
reimplements the public rules of a commercially published board game
(factual game data: card/wonder/token names, costs, and effects) alongside
original engine, AI-training, and UI code. There is value in being able to
share progress, accept outside contributions/feedback, and (later) point RL
research or portfolio material at the repo.

## Decision

The GitHub repository (`kncesarini/duels`) is public.

## Consequences

- No copyrighted game artwork, iconography, or verbatim rulebook text is
  committed — only factual game data (names/costs/effects), which is not
  copyrightable in the way creative expression is. `data/README.md` flags
  this data as a best-effort transcription needing a spot-check.
- Anyone can read the source and history; no secrets (API keys, tokens,
  credentials) may ever be committed. CI and any future deployment
  configuration must source secrets from GitHub Actions secrets / environment
  configuration, never from files in the repo.
- Branch protection on `main` (see the repo's branch protection ruleset) and
  `CODEOWNERS`-gated review on docs/CI/data paths are the primary controls
  against unreviewed changes landing directly on `main`, since anyone can
  fork and open a PR.
