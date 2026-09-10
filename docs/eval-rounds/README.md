# `duels-eval`: the round-by-round tuning history

Every weight in [`duels-eval`](../../crates/duels-eval) was arrived at by a
numbered *round*: one PR that states a hypothesis, builds it behind a `Config`
option, measures it against the previous generation over paired, seat-swapped
games, and then keeps it or writes down why it was not kept. Honest negatives
are recorded as carefully as the wins — the argument is usually worth more than
the number, and several rounds' most useful finding is a change that measured
worse than the model predicted.

These files are that record. They were relocated verbatim out of
`crates/duels-eval/src/lib.rs`'s module doc comment, which had grown to about
2,700 lines of history above roughly 3,800 lines of code; nothing was
summarised or dropped on the way out. Rustdoc intra-doc links (`` [`Config`] ``)
became plain code spans, since markdown has no such target.

The crate's *current* design — what the evaluation is, how the blend works,
what the rails are — stays in `lib.rs`. This directory is why it is shaped
that way.

| Round | PR | Headline |
| --- | --- | --- |
| [two](round-02.md) | #27 | Five changes, measured one at a time |
| [three](round-03.md) | #29 | A rail instead of a gradient |
| [four](round-04.md) | #31 | The turn the evaluator was scoring half of |
| [five](round-05.md) | #32 | The cards nobody was pricing |
| [six](round-06.md) | #33 | The one wonder effect that is not like the others |
| [seven](round-07.md) | #44 | The forward-looking terms were collectively over-priced |
| [eight](round-08.md) | #48 | The calibration was stale, and the ladder stopped one rung too early |
| [nine](round-09.md) | #49 | The wonder term was over-paying, and fixing it helps only the agent that does not matter |
| [ten](round-10.md) | #56 | One weight, fitted, and the only one everything agreed about |
| [eleven](round-11.md) | #59 | The science weight was frozen at the root, and unfreezing it is worth nothing |

Round one is the crate itself (#25, then extracted out of
`duels-agent-phased` in #36); it has no round file because its content *is*
the crate's design docs in `lib.rs`. Round eleven landed under a commit
subject reading "round ten" (#59) — its own doc comment, relocated here as
`round-11.md`, is the authority on the number.

[cross-round-measurements.md](cross-round-measurements.md) holds the tables
that span rounds rather than belonging to one: the round-one-to-three era Elo
and leave-one-out tables, the `mcts-uct` bar, the `military_band` sweep, the
per-decision cost breakdown and the Age I take profile.

## Reading these

Two conventions run through all of them, and neither is obvious on a first
read:

* **`Config::vN` is a generation, not a version number.** Every round that
  changes a default adds a `vN` constructor that reproduces the *previous*
  generation bit for bit, plus a `tests/vN_identity.rs` that asserts it. That
  is what lets one binary benchmark a round against the round before it
  (`phased:base=v3`, `mcts-eval:eval=v9`) instead of against a rebuild.
* **Every Elo figure is paired and seat-swapped**, on at least two disjoint
  seed ranges, and a change whose two ranges disagree in sign is reported as
  neutral rather than as the flattering half. Sample sizes are stated because
  they mattered: round four records a change that read `+2.6 / +1.7` at 800
  games per range and `-11.0 / -5.6 / -3.9` at 3,200.
