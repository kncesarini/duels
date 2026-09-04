# Agent contract

`CONTRACT_VERSION = 2`

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

## Changes in version 2 (M1)

M1 replaced the M0 placeholders with the real rules engine, which is a
breaking change to every shape below:

- **Ids are newtypes, not `String`.** `CardId`, `WonderId` and `TokenId` are
  one-byte opaque newtypes (`duels_core::data`). They still *serialise* as
  their stable JSON slug (`"lumber-yard"`), so wire messages and replays are
  unchanged in appearance, but in Rust they are `Copy` and one byte, which is
  what lets `GameState` be a 256-byte `Copy` value.
- **`Action` variants were renamed and extended.** `PlayCard`/
  `DiscardCardForCoins`/`BuildWonder` now refer to a **slot index** in the
  age structure rather than a card id, because the same card can only ever be
  in one slot and slots are what the rules make legal or illegal.
- **`Observation` is a real type**, no longer a placeholder with a turn
  counter.
- **A new `Player` enum** (`duels_core::Player`) replaces implicit seat
  indices.
- `Action` is now `Copy`, so `choose` can return `legal[i]` without cloning.

## Why `GameState` and `Observation` are separate types

7 Wonders Duel is stochastic (card draws, deck shuffling) but has **no
player-private hidden information** — both players always see the exact
same public game state. The only thing either player doesn't know is what
hasn't been revealed yet: which card is behind each face-down slot of the
current age's structure, which three cards of each age deck were returned to
the box unseen, the composition of the not-yet-dealt age decks, and (during
the first half of the draft) which four wonders will be offered next.

Because of this, an AI agent must never be handed anything that lets it
infer that not-yet-revealed information beyond what a human opponent could
infer (i.e. it may reason about the *pool* of cards still possible at a
hidden position, but never a concrete resolved identity before it's
revealed). Rather than rely on every future agent implementation to
carefully avoid reading a few "don't touch this" fields, this is enforced
structurally:

- `duels_core::state::GameState` is the full, authoritative representation.
  **Every one of its fields is private.** Its public accessors return only
  public information; the hidden parts are reachable exclusively through
  `pub(crate)` accessors, so nothing outside `duels-core` can read them even
  by accident. (`GameState` does implement `Serialize` so the authoritative
  server can persist a game — a serialised `GameState` contains hidden
  information and must never be sent to a client.)
- `duels_core::observation::Observation` is derived from a `GameState` by
  `GameState::observation()`, with every not-yet-public value replaced by the
  pool of values it could still resolve to.
- `duels_agents_api::Agent::choose` takes `&Observation`, never
  `&GameState`, and `duels-agents-api` does not depend on `GameState` at
  all. An agent implementation is therefore structurally incapable of
  seeing hidden information, not merely asked nicely not to.

The operational definition of "no leak", asserted as a property test over
thousands of positions (see `docs/rules-spec.md` R-101): **two `GameState`s
that differ only in hidden information must produce equal `Observation`s.**

## `Observation`

`Observation` carries all public state. Every collection is emitted in a
canonical order (ascending by id) so that equality is meaningful.

```rust
pub struct Observation {
    pub phase: Phase,                       // WonderDraft | Turn | ChooseFirstPlayer | GameOver
    pub age: u8,
    pub current_player: Player,
    pub turn: u32,
    pub conflict: i8,                       // positive: Player::One is ahead
    pub loot_taken: [[bool; 2]; 2],
    pub extra_turn: bool,
    pub pending: Option<Pending>,           // an outstanding effect choice
    pub last_card_taker: Player,
    pub players: [PublicPlayer; 2],
    pub slots: [SlotView; 20],              // Empty | FaceUp { card } | FaceDown
    pub discard: Vec<CardId>,
    pub wonder_fodder: Vec<CardId>,         // cards spent under wonders
    pub board_tokens: Vec<TokenId>,
    pub set_aside_tokens: Vec<TokenId>,     // public: the complement of the board five
    pub offered_wonders: Vec<WonderId>,
    pub undrafted_wonder_pool: Vec<WonderId>,
    pub draft_step: u8,
    pub draft_first: Player,
    pub unknown_slot_pool: Vec<CardId>,     // candidates for the face-down slots
    pub hidden_guild_count: u8,             // how many face-down slots hold a guild
    pub result: Option<GameResult>,
}

pub struct PublicPlayer {
    pub coins: u16,
    pub built: Vec<CardId>,
    pub wonders: Vec<WonderId>,
    pub wonders_built: Vec<WonderId>,
    pub tokens: Vec<TokenId>,
    pub shields: u8,
    pub science: [u8; 7],
    pub pairs_awarded: Vec<Science>,
}

pub enum SlotView {
    Empty,
    FaceUp { card: CardId },
    FaceDown,                               // deliberately carries no id
}
```

`SlotView::FaceDown` carries no card id at all. The candidate cards live in
`unknown_slot_pool`, which is always *strictly larger* than the number of
face-down slots (three cards of each age go back in the box unseen), so a
hidden card can never be pinned down by elimination.

`set_aside_tokens` is public because it is deducible: it is the complement of
the five tokens on the board. The only randomness those five carry is which
three The Great Library draws, and that resolves when the wonder is built.

### From `Observation` back to a state, for search

```rust
fn sample_state(&self, rng: &mut StdRng) -> GameState
```

Samples the hidden information uniformly from the pools the observation
exposes, respecting every public constraint (including that exactly three
guild cards sit in the Age III structure), and returns a valid, playable
`GameState`. `sample_state(rng).observation() == *self` always holds. This is
the bridge a determinized (PIMC) or MCTS agent needs; it never reveals what
the *actual* game is hiding.

## `Action` (version 2)

Defined in `crates/duels-core/src/action.rs`. `Slot` is `u8`, an index into
the 20-slot age structure (see `duels_core::layout`).

- `PickWonder { wonder: WonderId }` — take one of the four wonders currently
  on offer during the initial draft.
- `Build { slot: Slot }` — take the card in `slot` and construct it, paying
  its cost or building it free via a chain symbol.
- `Discard { slot: Slot }` — take the card in `slot` and discard it for
  `2 + your own yellow cards` coins.
- `BuildWonder { slot: Slot, wonder: WonderId }` — take the card in `slot`
  and spend it to construct one of your own unbuilt wonders.
- `ChooseProgressToken { token: TokenId }` — after completing a pair of
  identical scientific symbols, take one of the tokens still on the board.
- `ChooseGreatLibraryToken { token: TokenId }` — keep one of the three
  tokens The Great Library drew from the out-of-play pile.
- `MausoleumBuild { card: CardId }` — construct a card from the discard pile
  for free.
- `DestroyOpponentCard { card: CardId }` — discard one of the opponent's
  buildings of the colour a destroy effect names.
- `ChooseFirstPlayer { player: Player }` — decide who begins the new age.
  Only the militarily weaker player is ever asked.

`legal_actions` is the single source of truth for which of these are
available, and it is empty exactly when the game is over.

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
- Agents must be deterministic given their own seed. The workspace
  `clippy.toml` bans `Instant::now`, `SystemTime::now`, `rand::thread_rng`
  and `rand::random`; `duels-core` denies `clippy::disallowed_methods` at the
  crate level and future agent crates under `crates/agents/**` must do the
  same. An agent that needs wall-clock time to honour `Budget::TimeMs` will
  have to take an explicit deadline from the arena, or `allow` the lint at
  the specific call site with a comment saying why.

## Engine entry points an agent's driver uses

```rust
engine::new_game(seed: u64) -> GameState
engine::legal_actions(&GameState) -> Vec<Action>
engine::legal_actions_into(&GameState, &mut Vec<Action>)     // no allocation
engine::apply(&mut GameState, Action, &mut StdRng) -> Result<Vec<Event>, IllegalAction>
engine::apply_quiet(&mut GameState, Action, &mut StdRng) -> Result<(), IllegalAction>
engine::apply_unchecked(&mut GameState, Action, &mut StdRng) // trusts legality
engine::chance_outcomes(&GameState, Action) -> Vec<(Outcome, f64)>
engine::apply_with_outcome(&mut GameState, Action, &Outcome) -> Result<Vec<Event>, IllegalAction>
scoring::score(&GameState) -> [Breakdown; 2]
```

`chance_outcomes` / `apply_with_outcome` exist for search that wants to
expand a chance node explicitly rather than determinize: probabilities are
computed from public knowledge only, and forcing an outcome rewrites the
state's hidden layout so the result is a state that genuinely could have
arisen from that reveal.

## Versioning in practice

When a PR changes any of the above in a breaking way:

1. Bump `CONTRACT_VERSION` in this file.
2. Say so explicitly in the PR description (what changed and why).
3. Update any dependent crates (`duels-arena`, `duels-server`, and later the
   PyO3 bindings and TypeScript client codegen) in the same PR where
   feasible, or file a follow-up issue if not.
