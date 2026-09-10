# Round two: five changes, measured one at a time

The first cut of this crate had one working plan and two broken ones. It
beat `greedy-ev` by six hundred Elo and it beat `mcts-uct` 8% of the
time — and *every one of those 32 wins was scientific supremacy*. Never
military, never points. Two implementation faults and three missing ideas
were diagnosed from real match data. All five are `Config` options; all
five default to on; `Config::v1` turns all five off and reproduces the
previous agent bit for bit (`tests/legacy_identity.rs`), which is what lets
`phased` and `phased:base=v1` be benchmarked against each other out of one
binary.

**1. `next_age_start` was double its intended magnitude.** The term is read
per player and then differenced, and taking the very first shield from a
centred pawn flips who is projected to start the next age — so the
*differenced* swing was twice the weight written down, about eight victory
points for a single shield. `phased` consequently kept 1.5% of the red
cards it saw in Age I. Halving the weights to `[1.5, 1.0, 0.0]` matches
`docs/strategy-backlog.md` §1.2's 1-3 VP estimate for the start-of-age
choice.

**2. `MilitaryModel::Band`.** End-of-game military scoring is a step
function (0 / 2 / 5 / 10 victory points at pawn distances 0 / 1-2 / 3-5 /
6-8) plus two loot tokens; a flat reward per step prices the shield that
crosses 2→3 exactly like the one that does nothing. The steps are smoothed
by how many shields are still in play, because what a position is worth is
the expectation of the step function over where the pawn ends up — see
`terms::MilSmoothing`.

**3. `CoinModel::Smooth`.** Three ad hoc coin terms — a floored points
channel, a capped race-liquidity bonus and a safety-floor penalty —
replaced by one continuous function, with the real `floor(coins / 3)`
restored once the rounding is about to actually happen.

**4. `EconomyModel::Bill`.** The development term prices what a city's
own production *saves it*. Nothing priced what that production *costs the
opponent*. Read per player and differenced, `terms::resource_bill` is
where monopoly value comes from, with nothing anywhere saying "grey is
good". This is the single largest gain of the round.

**5. `menu`: the opponent's menu, and chain equity.** What the next
mover's best affordable card is worth to them, and what a chain starter is
worth for the successor it unlocks. Both priced once per decision from the
root; see that module for why, and for the stand-down rule that keeps the
first one from reading a hidden card.
