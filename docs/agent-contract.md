# Agent contract

`CONTRACT_VERSION = 1`

This document describes the contract between `duels-core` (the rules
engine) and any AI/bot player, defined by the `Agent` trait in the
`duels-agents-api` crate. Any future **breaking** change to the shapes
described here (removing/renaming a field or variant, changing a type in an
incompatible way, changing the meaning of an existing field) must bump
`CONTRACT_VERSION` and be called out explicitly in the PR description that
makes the change. Additive, backwards-compatible changes (e.g. a new
`Action` variant that existing exhaustive-match agents would need a wildcard
arm to ignore) should still be mentioned in the PR but do not require a
version bump.

This is a living document. As of M0, the shapes below are placeholders
(see the doc comments in `crates/duels-core/src/*.rs`); the full field list
for `GameState`/`Observation` lands in M1 together with the rules engine
that produces them. What's fixed by this document, even in M0, is the
*separation* between the full state and the public observation, and the
overall shape of the `Agent` trait.

## Why `GameState` and `Observation` are separate types

7 Wonders Duel is stochastic (card draws, deck shuffling) but has **no
player-private hidden information** — both players always see the exact
same public game state. The only thing either player doesn't know is what
hasn't been revealed yet: the order of the current age's remaining deck, and
which cards ended up face-down in the age-card structure vs. which specific
guild cards were set aside during Age III setup.

Because of this, an AI agent must never be handed anything that lets it
infer that not-yet-revealed information beyond what a human opponent could
infer (i.e. it may reason about the *pool* of cards still possible at a
hidden position, but never a concrete resolved identity before it's
revealed). Rather than rely on every future agent implementation to
carefully avoid reading a few "don't touch this" fields, this is enforced
structurally:

- `duels_core::state::GameState` is the full, authoritative representation.
  Only the engine (`duels-core::engine`) and server-side simulation code
  (`duels-server`, `duels-arena`) ever construct or hold one.
- `duels_core::observation::Observation` is derived from a `GameState` by
  the engine, with every not-yet-public value replaced by the pool of
  values it could still resolve to.
- `duels_agents_api::Agent::choose` takes `&Observation`, never
  `&GameState`, and `duels-agents-api` does not depend on `GameState` at
  all. An agent implementation is therefore structurally incapable of
  seeing hidden information, not merely asked nicely not to.

## `Observation` (conceptual shape)

Represents everything a player is allowed to know at decision time:

- Each player's built structures (cards, wonder stages), coins, and science
  symbols owned.
- The current public age-card structure: which positions are face-up
  (with a known card) vs. face-down (unknown card, but its *possible*
  identities are constrained by what's already been seen this age).
  Face-down/unrevealed positions are represented as a pool of remaining
  possible cards, never a resolved identity.
- The military track position and which military tokens have been claimed.
- Which progress tokens have been claimed, and which remain available to
  be chosen (as a set, not an order, when order isn't public).
- Whose turn it is and which age/turn-phase the game is in.

## `Action` (current shape, M0)

Defined in `crates/duels-core/src/action.rs` as an enum. As of M0:

- `PlayCard { card: CardId }` — build a card, paying its cost or building
  free via a chain symbol.
- `DiscardCardForCoins { card: CardId }` — discard instead of building.
- `BuildWonder { wonder: WonderId, card: CardId }` — spend a card to build
  a wonder stage.
- `ChooseProgressToken { token: ProgressTokenId }` — pick a progress token
  when an effect grants a choice.

`CardId`, `WonderId`, and `ProgressTokenId` are `String` ids matching the
`id` fields in `data/cards.json`, `data/wonders.json`, and
`data/tokens.json` respectively.

More variants (e.g. resolving the Economy token's "take a brown/grey card
from the discard pile" choice, or a Diplomacy-token military reset) will be
added in M1 as the engine implements the effects that need them.

## `Agent` trait

```rust
pub trait Agent {
    fn spec(&self) -> AgentSpec;
    fn choose(&mut self, obs: &Observation, legal: &[Action], budget: Budget) -> Action;
}

pub struct AgentSpec {
    pub name: String,
    pub version: String,
    pub params: String,
}

pub enum Budget {
    Nodes(u64),
    TimeMs(u64),
}
```

- `spec()` returns static identifying metadata used for logging and
  tournament leaderboards (see the future `duels-arena` tournament runner).
- `choose()` is given the current public `Observation`, the full list of
  actions the engine considers legal right now (`legal`), and a `budget`
  describing how much computation the caller is willing to let the agent
  spend. The agent must return one of the elements of `legal`.
- The engine (not the agent) is the source of truth for what's legal;
  `choose` is not expected to validate legality itself, though a
  conforming arena/server may re-validate the returned action defensively
  before applying it.

## Versioning in practice

When a PR changes any of the above in a breaking way:

1. Bump `CONTRACT_VERSION` in this file.
2. Say so explicitly in the PR description (what changed and why).
3. Update any dependent crates (`duels-arena`, `duels-server`, and later the
   PyO3 bindings and TypeScript client codegen) in the same PR where
   feasible, or file a follow-up issue if not.
