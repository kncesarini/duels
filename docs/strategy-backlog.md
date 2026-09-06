# Strategy backlog

A prioritized catalog of 7 Wonders Duel strategic concepts not yet captured by any AI agent in this project, generated as a brainstorming pass separate from active development. This is reference material for future rounds of evaluation/agent work, not a spec — nothing here is validated against the arena yet, and a few specific card/chain facts are flagged for verification against `data/*.json` before anyone builds on them.

Each item is rated by **impact** (how many points/win-probability it swings when it comes up) and **frequency** (how often a game turns on it). See the "Suggested build order" at the end for a prioritized list by expected Elo per unit of implementation effort.

---

**Scope note.** Everything below is game strategy as a strong human player reasons about it, deliberately steering clear of topics already covered by the current AI work: military/science race distance and denial value, basic resource pricing, Age I color heuristics, the science-symbol ladder, and behind-so-commit dynamics.

## 0. Cross-cutting principles (highest leverage, mostly absent from 1-ply thinking)

### 0.1 Swing valuation, not own-gain valuation
Every card ends up in exactly one of three places: my city, their city, or gone. The value of taking a card is (what it does for me) + (what it would have done for them had they taken it), weighted by the probability they would actually have taken it. A 7-VP Palace both players can afford is a ~14-point decision, not a 7-point one. A card only I can use is worth exactly its face value and can often wait. Any evaluation that scores "my city after this move" without a term for "the best card I just removed from the opponent's menu" will systematically under-deny and over-greed.
*Impact: High. Frequency: Every turn.*

### 0.2 Reveal ownership — the "second cover" rule
A face-down card flips only when both cards covering it are gone. The player who removes the second cover hands the opponent first access to whatever flips. Every candidate action has a hidden cost: does this pick complete an uncover, and what is the expected value of the flipped card *to the opponent*? A strong player prefers picks that uncover nothing, or that uncover a card the remaining pool suggests will be mediocre for the opponent, and actively steers so the opponent is the one forced to remove second covers.
*Impact: High. Frequency: Every turn from mid-age onward.*

### 0.3 Extra turns are the only parity flips in the game
With 20 cards per age and strict alternation, the entire slot sequence is pre-determined — an extra turn is the one thing that re-assigns every remaining slot to the other player. Consequences: an extra turn converts a "gift reveal" into a private draw (remove the second cover with the wonder-build, then take the flipped card yourself); in a late-age parity puzzle, an available extra turn is a pass-move that breaks the zugzwang. Value an unbuilt extra-turn wonder not just for its printed effect but as a stored parity flip, more valuable when many face-down cards remain.
*Impact: High. Frequency: Medium (a few critical moments per game, decisive when they come).*

### 0.4 Turn budget
Each player gets roughly 30 actions in a game; wonder builds and forced discards eat into card-taking turns. Late in Age III the binding constraint is often turns, not resources — "5 turns left, 2 unbuilt wonders, 4 wanted blues" is a triage problem. A planner that doesn't count remaining turns will accumulate wonders it can never build.
*Impact: Medium. Frequency: High in Age III.*

### 0.5 Affordability map and tempo starvation
Track whether the opponent can pay for every accessible card right now. Leaving them a board they can't afford degrades their turn to a bad discard or purchase — a legitimate winning pattern in resource-poor Age III matchups. Corollary: keep your own coin cushion sized to the most expensive card about to become accessible.
*Impact: High. Frequency: Medium.*

### 0.6 Contested-first, safe-later
Order pickups by contestedness. A card both players want and can afford must be taken now; a card only you can use can wait unless the opponent's denial is cheap for them. Estimate denial likelihood from (card's value to you) vs. (opponent's best alternative minus their discard coins).
*Impact: Medium. Frequency: High.*

## 1. Structure geometry

- **1.1 The three ages open very differently.** Age I: 6 accessible cards, wide and safe. Age II/III: only 2 accessible cards at the start, reveal-heavy from move one.
- **1.2 Who chooses the next age's starter is decided by military.** The weaker-military player chooses; centered pawn goes to whoever played last (verify tie rule). A small military lead at an age boundary has a hidden price — it hands the opponent the start-choice. Worth pricing as a real asset (est. 1-3 VP-equivalent), and worth an arena measurement.
- **1.3 Which seat is better in Age II/III is an open empirical question** — worth resolving by forcing "always first" vs "always second" vs "choose by board" through the arena.
- **1.4 Late-age parity puzzles.** With ~6 cards left, the sequence becomes fully enumerable; only an extra turn changes who's forced into the bad reveal.
- **1.5 The wonder-build trigger card is a reveal decision too** — choose it by (denies the opponent) + (doesn't complete an uncover) + (if it does, do I have the extra turn to take the flip), not by "least valuable card."
- **1.6 The 20th card of each age is uncontested**, belonging to the age's second player absent extra turns.

*Impact: Medium throughout. Frequency: every age transition/boundary.*

## 2. Wonder draft and wonder play

- **2.1 Baseline tier list**: top tier Piraeus, Temple of Artemis, Hanging Gardens, Appian Way, Sphinx (5 of 12 wonders grant an extra turn — landing 3 is a huge tempo edge); strong/situational Great Lighthouse, Statue of Zeus, Circus Maximus, Colossus, Mausoleum; lower Pyramids, Great Library (swings on which tokens are hidden, see 2.8).
- **2.2 The 1-2-1 pick structure**: the single-picker's second wonder is forced (whatever's left) — weigh the first pick against what you'd be leaving the opponent. The double-picker can afford one pure counter-pick.
- **2.3 Cost-profile coherence** across your four wonders — a compact production base builds all of them; a scattered one means treating each wonder as "cost + coins."
- **2.4 Read the opponent's wonder costs** for denial and your own safety (public from the draft).
- **2.5 Only seven wonders get built total** — the eighth is discarded unbuilt; a race dynamic when both players hold four strong wonders.
- **2.6 Timing per wonder type**: production wonders ASAP; coin wonders right before a spending burst (Appian Way specifically timed to zero the opponent's coins); destroy wonders early (opponent can replace from Age II supply) vs. Age III (permanent lockout, no more brown/grey cards); pure-VP wonders last unless the extra turn has a concrete target.
- **2.7 Theology/Architecture reorder wonder-building priority** — build non-extra-turn wonders after Theology; value Architecture by unbuilt-wonder count.
- **2.8 Great Library's hidden five are knowable by elimination** — any specific hidden token appears with probability 3/5; worth radically different amounts depending on what's hidden.
- **2.9 Mausoleum removes discard-for-denial from the opponent's toolkit** and lets you "bank" an unaffordable card for later.
- **2.10 Wonders are the cleanest denial vehicle** when a card must be denied and you can afford a wonder build.

*Impact: High overall — decided largely in the draft and first few picks. Frequency: every game.*

## 3. Guilds and Age III

- **3.1 Guilds score the max city** — you need the card, not the majority, making them a near-symmetric ~2x swing card and an affordability race.
- **3.2 Guild affordability is a grey-production problem** — 5 of 7 guilds cost glass/papyrus.
- **3.3 Guild coins are counted at build time, VP at game end** — an uncontested guild can wait for more coins; a contested one can't.
- **3.4 Guild presence is a 3/7 prior**, updated by what's face-up at Age III start.
- **3.5 Discard a guild neither player can afford yet if the opponent will get there first.**
- **3.6 Age III has no production cards** — your economy is frozen at the end of Age II; project Age III affordability mid-Age-II and pivot early if there's a gap.
- **3.7 Age III yellow "counting" cards are highly state-dependent** — evaluate off your own city's actual counts, not a flat value.
- **3.8 The blue VP pool is finite** (~30 realistic Age III VP after removed cards) — denying blues matters as much as taking them for a points-focused opponent.
- **3.9 Pretorium converts 8 coins into 3 shields** — a coin-rich opponent can jump the military track in one action; factor into coin-cushion/threat logic.

*Impact: High, concentrated in Age III. Frequency: every game.*

## 4. Chain building

- **4.1 The chain map** (verify against data before use — flagged uncertain: Palisade→Fortifications and Tavern→Lighthouse skipping an age).
- **4.2 Chain equity as a tracked forward-value quantity**: P(successor appears) × (cost avoided + VP/effect) — real, public, forward-looking value a snapshot evaluation misses entirely.
- **4.3 Chain denial priorities** — deny the expensive successors (Aqueduct, Gardens, Pantheon, Senate, the Age III shield-chains), not the cheap ones.
- **4.4 Chain traps** — weight reveal-aversion by the opponent's outstanding chain equity in the hidden pool.
- **4.5 Chains reshape resource planning** — a player with several chain-starters can skip brown cards the opponent must buy.
- **4.6 Urbanism turns chains into a coin engine** — very strong with 3-4 starters in hand, weak with none.

*Impact: High. Frequency: every game.*

## 5. Resource economy and trade

- **5.1 Monopoly arithmetic is steep** — trade cost = 2 + opponent's production; stacking a resource the opponent's wonders/chains need is a real plan.
- **5.2 Reserves/Customs House fix price at 1** and kill monopoly value entirely once the opponent holds one — stop stacking that resource once they do.
- **5.3 Grey is the tightest market** — owning both sources of a grey is the strongest monopoly in the game; a one-source opponent is a Circus Maximus target.
- **5.4 Flexible producers (Forum, Piraeus, etc.) don't raise the opponent's price** — pure supply insurance, not a denial tool.
- **5.5 Economy token turns opponent trades into your income** — valuable against a thin economy, near-worthless against a self-sufficient one.
- **5.6 There's a "sufficiency point"** past which more production is worth less than a green/blue/yellow card — over-production is a common leak since each brown "feels" free.
- **5.7 Coins are 1/3 VP** — every trade is a points transaction, but also has a tempo side (a turn you can't afford next turn's card costs a full turn).

*Impact: High. Frequency: every game.*

## 6. Coin economy

- **6.1 Discard-for-coins scales with yellow count** — with 4 yellows every discard is 6 coins, making discard-as-denial much more attractive.
- **6.2 Coin income schedule** — plan income to land before the coin-hungry Age I/II stretch; coins hoarded into Age III are worth little unless Moneylenders/Pretorium is in play.
- **6.3 Looting is a coin attack — time it** for when the opponent is rich, ideally right before a purchase they need the coins for.
- **6.4 Coin rounding (3:1) matters exactly in the last few turns** and can flip a close game.
- **6.5 The zero-coin trap** — a board-relative (not flat) coin-liquidity term is needed since Appian Way can force this from 3.

*Impact: Medium-High. Frequency: every game.*

## 7. The remaining progress tokens

Agriculture (default early pick, liquid), Philosophy (more VP, less tempo, better late), Mathematics (value = 3 × final token count — worthless as a first token, excellent as a third), Masonry (value scales with blue cards left in the game and your own weak production), Architecture (scales with unbuilt wonders), Urbanism (scales with chain-starters held), Economy (scales with opponent's projected trading), and a token-race dynamic (only 5 of 10 are ever available; completing your pair first can matter more than which symbol you get).

*Impact: Medium overall, situationally high. Frequency: ~half of games per token.*

## 8. Military beyond race distance

- **8.1 Military VP is a step function (2/5/10 at bands 1-2/3-5/6-8)** — crossing 2→3 or 5→6 is a large swing (VP + loot); a shield that stays within a zone is worth ~0 now.
- **8.2 Zone 6 is a soft win condition** in its own right (~12-point swing including loot) when supremacy is out of reach.
- **8.3 The military tax** — matching every shield costs turns; sometimes conceding a zone and spending the turns elsewhere is correct, distinct from supremacy denial (always must-answer).
- **8.4 Remaining shield supply is countable** — gives an upper bound on whether a threat is even live this age.
- **8.5 Take-vs-discard asymmetry** — taking a red card is a bigger swing than discarding it (denies + gains vs. just denies).

*Impact: High. Frequency: every game's mid/endgame.*

## 9. Endgame calculation and tiebreaks

- **9.1 Ties go to civilian VP only** — a hidden premium on blue cards in any close projection.
- **9.2 Exact-tally mode in the last ~6 cards** — enumerate outcomes precisely; a "losing" position is often actually a win once boundaries/rounding/guild-marginals are worked out exactly.
- **9.3 Guild VP moves with every card taken** by either player in the late game.
- **9.4 Last-turn denial reflex** — a discard that removes the opponent's best card often beats taking your own second-best.
- **9.5 Unbuilt wonders in final turns** are VP + a denial vehicle only if affordable and the 7-wonder cap allows it.

*Impact: High in close games (which are the ones that matter). Frequency: every game's last few turns.*

## 10. Card counting and expected reveals

- **10.1 The hidden pool is small and fully specified** — cheap to compute exact probabilities for "is this card still in the structure."
- **10.2 Opponent-relative EV of the unseen pool** drives reveal-aversion — expensive to uncover when the hidden pool favors them, cheap when it doesn't.
- **10.3 Science supply counting** tells you whether a supremacy threat is still live, not just how many symbols are held.
- **10.4 Chain successor presence updates chain equity** with every reveal.
- **10.5 Predicting the opponent's forced move** lets you shape what your own pick uncovers for them.

*Impact: High. Frequency: every turn.*

## 11. Variance profile beyond "behind = desperate"

- **11.1 Extra-turn holders want more hidden cards alive**; non-holders want fewer.
- **11.2 When ahead: lock value, avoid reveals, shorten the game** — a comfortable lead loses to variance far more often than to good play.
- **11.3 When behind: maintain multiple forcing threats**, not just one committed race — each forces opponent denial turns that don't score for them.
- **11.4 Get your variance early in Age III**, not in the last few turns once everything is known and certain.

*Impact: Medium. Frequency: common.*

## 12. Opening theory

- **12.1 Draft to a coherent archetype** (production/tempo, military kit, points, science-enabling) rather than a sum of tier-list values.
- **12.2 First Age I picks are about grey sources and the wonder-cost base** — often first-pick material over any brown.
- **12.3 The 7-coin start is a budget** — free cards (Tavern especially) are legitimately strong early picks when costed cards exceed budget.
- **12.4 Age I military is two-edged** — free early shields cede the age-boundary start-choice (see 1.2).
- **12.5 Age I is a third of your city** — its cards are almost entirely multiplicative in value (chains, production, symbols) rather than face-VP, which current heuristics only partly capture.

*Impact: High. Frequency: every game.*

## 13. Miscellaneous one-liners

Denial cost ladder (take-and-use > wonder-build-with-it > take-and-not-use > discard-for-coins > let it go); contest what's cheap for both, leave what's expensive for them; second-copy greens (completing your own pair vs. denying theirs); yellow-density compounding; Statue of Zeus vs. doubled-brown targets; wonder-build as an escape hatch when every accessible card is a gift; Brewery/Tavern coin timing; the three "double blue" chain lines (Rostrum/Senate, Temple/Pantheon, Statue/Gardens) as trackable chain equity.

---

## Suggested build order (by expected Elo per unit of implementation effort)

**Tier 1 — cheap to compute, present every turn, large swings**
1. Swing valuation / denial term (§0.1) — value = my gain + their forgone gain.
2. Reveal ownership: second-cover detection + opponent-relative EV of the hidden pool (§0.2, §10.1-10.2).
3. Military VP as a step function with boundary/loot awareness (§8.1, §8.2, §6.3).
4. Chain equity as a tracked forward-value term (§4.2-4.4).
5. Grey monopoly / second-copy logic and Reserve-neutralization (§5.1-5.3).

**Tier 2 — moderate effort, large but less frequent**
6. Wonder cost-profile coherence and per-wonder timing rules (§2.3, §2.4, §2.6).
7. Extra turns as parity flips / double-pick planning (§0.3, §11.1).
8. Guild race valuation with max-city semantics, affordability, discard-the-guild (§3.1, §3.2, §3.5, §9.3).
9. Age II→III production-freeze projection (§3.6, §5.6).
10. Affordability map / tempo starvation (§0.5, §6.5).
11. Age-start choice policy — resolve empirically via the arena (§1.2-1.3).

**Tier 3 — refinements**
12. Token-specific valuations (§7, §2.7).
13. Turn budget and seven-wonder cap (§0.4, §2.5, §9.5).
14. Blue tiebreak premium and coin rounding (§9.1, §6.4).
15. Yellow-density compounding and discard-as-denial economics (§6.1, §13).
16. Mausoleum discard-pile tracking and Great Library hidden-token inference (§2.8-2.9).
17. Draft archetype coherence and the 1-2-1 seat calculus (§12.1, §2.2).

**Verify before building on these**: exact chain links (especially Palisade→Fortifications and Tavern→Lighthouse), the science symbol distribution split across ages, the age-start tie rule when the pawn is centered, and the Age II/III initial accessible-card count — several items above lean on them.
