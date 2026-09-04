# 0005. Base-game scope: no Agora, no Pantheon

## Context

7 Wonders Duel has two published expansions, Agora and Pantheon, each adding
new card types, tokens, and rules interactions on top of the base game.
Supporting them multiplies the rules surface (and therefore the engine,
data, and AI-training complexity) before the base game itself is even
implemented.

## Decision

M0 and the milestones immediately following it implement only the base
7 Wonders Duel ruleset: the 73 age cards (including the 7 base-game guild
cards, 3 of which are used per game plus 1 extra slot), the 12 base wonders,
the 10 progress tokens, and the 4 military tokens plus the base military
track. Agora and Pantheon content is explicitly out of scope until a future
ADR revisits this.

## Consequences

- Smaller, well-defined rules surface for the initial engine, scoring, and
  AI-training work — matches most published 7WD strategy/AI research, which
  also targets the base game.
- `data/cards.json`, `data/wonders.json`, and `data/tokens.json` (see
  `data/README.md`) contain only base-game content; expansion content would
  be new data files plus engine changes, not edits to these.
- The `GameState`/`Observation`/`Action` shapes in `duels-core` are designed
  against base-game mechanics only; expansion support may require additive
  (non-breaking) changes to these types, evaluated when that work is
  actually scoped.
