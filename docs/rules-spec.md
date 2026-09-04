# Rules specification

Every non-trivial rule `duels-core` implements, numbered, with the test(s)
that pin it down. This is the traceability spine: **changing engine behaviour
means changing an `R-xxx` entry here and its tests in the same PR.**

Test names are given unqualified; they live in:

| location | contents |
| --- | --- |
| `crates/duels-core/src/*.rs` (`mod tests`) | unit tests, next to the code |
| `crates/duels-core/tests/cost_engine.rs` | hand-computed cost scenarios |
| `crates/duels-core/tests/golden_scenarios.rs` | hand-built positions with exact expected outcomes |
| `crates/duels-core/tests/properties.rs` | `proptest` invariants over thousands of full games |

Where a rule is marked **⚠ unverified**, it was inferred rather than read off
the rulebook; see "Open questions" at the bottom and the M1 PR description.

---

## Data (R-001 … R-009)

| id | rule | tested by |
| --- | --- | --- |
| R-001 | The base game has 73 age cards (23 Age I, 23 Age II, 20 Age III non-guild, 7 guilds), 12 wonders, 10 progress tokens. | `data::tests::component_counts_match_the_base_game` |
| R-002 | `data/*.json` is normalised into flat typed structs at first use; an unrecognised effect type is a hard error rather than being silently dropped. | `data::tests::unknown_effect_type_is_a_hard_error`, `data::tests::effect_valid_on_the_wrong_entity_is_a_hard_error` |
| R-003 | Chain links are 1:1 and symmetric, and a chain prerequisite always comes from an earlier age. | `data::tests::embedded_data_parses_and_validates` (via `validate`) |
| R-004 | Each of the six card-borne scientific symbols appears on exactly two cards, so a pair is always achievable and never achievable twice. The Law token's `balance` appears on no card. | `data::tests::embedded_data_parses_and_validates` |
| R-005 | Only guild cards carry majority-based effects. | `data::tests::embedded_data_parses_and_validates` |
| R-006 | The military track runs to distance 9 (a capital) with loot tokens at distance 3 (2 coins) and 6 (5 coins), and end-of-game victory points of 0 / 2 / 5 / 10 for distances 0 / 1-2 / 3-5 / 6-8. | `data::tests::military_track_scoring_table` |
| R-007 | Card, wonder and token ids are one-byte newtypes that serialise as their stable JSON slug. | `data::tests::ids_round_trip_through_slugs`, `data::tests::ids_serialise_as_slugs`, `action::tests::ids_appear_as_readable_slugs_in_json` |
| R-008 | Normalised card / wonder / token facts match the source data (spot checks on trading posts, choice production, chains, guild effects, destroy effects, rebates). | `data::tests::spot_check_normalised_cards`, `data::tests::spot_check_normalised_wonders_and_tokens` |

## Age structures (R-010 … R-019)

| id | rule | tested by |
| --- | --- | --- |
| R-010 | Each age lays out exactly 20 cards. Age I is a 2-3-4-5-6 pyramid, Age II the same inverted (6-5-4-3-2), Age III the pinched 2-3-4-**2**-4-3-2 structure. **⚠ unverified** (see Open questions). | `layout::tests::row_sizes_match_the_printed_structures` |
| R-011 | A card at `(row, col)` is covered by the cards at `(row + 1, col - 1)` and `(row + 1, col + 1)`; a slot may be taken only once every slot covering it is empty. | `layout::tests::covers_is_the_inverse_of_covered_by`, `engine::tests::covered_slots_are_never_offered` |
| R-012 | Exactly 8 of the 20 cards are dealt face down in every age; every face-down slot is covered, so a face-down card can never be taken. | `layout::tests::every_age_has_twenty_slots_and_eight_face_down`, `layout::tests::a_face_down_slot_is_never_accessible_before_it_is_uncovered` |
| R-013 | At the start of an age exactly one row is accessible, and it is face up. | `layout::tests::initially_accessible_slots_are_the_bottom_row_and_are_face_up` |
| R-014 | The covering graph is acyclic and rooted at the accessible row, so every slot becomes reachable and the structure can always be emptied. | `layout::tests::emptying_the_structure_makes_every_slot_accessible_exactly_once` |
| R-015 | Age III's upper half is gated by the two-slot middle row: the four row-3 cards are covered only by those two, and are unreachable until both are gone. | `layout::tests::age_three_upper_half_is_gated_by_the_middle_row` |
| R-016 | Removing one card can uncover at most two others, which is what bounds the chance API's reveal set. | `layout::tests::emptying_one_slot_can_uncover_at_most_two` |
| R-017 | A slot that becomes uncovered is turned face up immediately, before the next decision. | `engine::tests::apply_with_outcome_forces_the_reveal`, `invariants_hold_throughout_a_game` (via `check_invariants`) |

## Setup (R-020 … R-029)

| id | rule | tested by |
| --- | --- | --- |
| R-020 | Each player starts with 7 coins. | `engine::tests::setup_starts_in_the_wonder_draft_with_an_undealt_structure` |
| R-021 | Three cards are returned to the box unseen from *each* age deck. For Age III that happens first (20 → 17), then three of the seven guilds are shuffled in, giving exactly 20 — so exactly three guilds are always in play and no Age III card is left over. | `engine::tests::setup_partitions_every_card_exactly_once`, `engine::tests::setup_deals_exactly_three_guilds_into_age_three` |
| R-022 | Guild cards never appear in the Age I or Age II structure. | `engine::tests::setup_deals_exactly_three_guilds_into_age_three` |
| R-023 | Five of the ten progress tokens go on the board; the other five are set aside and remain out of play **except** as the source for The Great Library. | `engine::tests::setup_puts_five_tokens_on_the_board_and_five_aside`, `the_great_library_takes_a_token_that_is_out_of_play` |
| R-024 | The wonder draft offers four wonders at a time. The first player takes one, the second takes two, the first takes the last; then the same over four fresh wonders with the roles reversed. Each player ends with four, and four of the twelve never enter the game. | `state::tests::draft_order_is_one_two_one_then_reversed`, `engine::tests::the_draft_gives_each_player_four_wonders_in_one_two_one_order` |
| R-025 | The Age I structure is dealt only after the draft finishes, so drafting decisions cannot be informed by the Age I cards. | `engine::tests::setup_starts_in_the_wonder_draft_with_an_undealt_structure`, `engine::tests::the_draft_gives_each_player_four_wonders_in_one_two_one_order` |
| R-026 | The conflict pawn starts centred with all four loot tokens present. | `every_seed_produces_a_valid_setup` |
| R-027 | The first player of Age I is the player who drafted first. **⚠ unverified**: the rulebook says only "choose a first player", so the engine picks one from the seed. | `engine::tests::the_draft_gives_each_player_four_wonders_in_one_two_one_order` |
| R-028 | Setup is a pure function of the seed: no clock, no ambient RNG. | `games_are_reproducible`, `tests::the_clippy_config_still_bans_nondeterminism` |

## Cost engine (R-030 … R-039)

| id | rule | tested by |
| --- | --- | --- |
| R-030 | A card whose chain prerequisite the player already owns is built for free: coin cost *and* resource cost are ignored entirely. | `a_chain_symbol_ignores_the_cost_entirely`, `a_chain_symbol_also_waives_a_coin_cost` |
| R-031 | A resource the player's own city produces is free; anything missing costs `2 + (units the opponent's brown and grey cards produce)` per unit. | `missing_resources_cost_two_plus_the_opponents_production`, `a_card_you_can_fully_produce_is_free` |
| R-032 | Yellow cards and wonders never raise the price the opponent pays, even when they produce resources. | `yellow_cards_and_wonders_never_raise_the_opponents_prices`, `state::tests::trade_relevant_production_excludes_yellow_and_wonders` |
| R-033 | A trading post (Stone / Clay / Wood Reserve, Customs House) fixes that resource at 1 coin per unit however much the opponent produces. | `a_trading_post_fixes_the_price_at_one_however_much_the_opponent_produces`, `the_customs_house_fixes_both_manufactured_goods`, `cost::tests::trading_post_beats_a_heavily_producing_opponent` |
| R-034 | "Produce one of your choice" sources (Forum, Caravansery, Piraeus, The Great Lighthouse) are assigned at payment time, optimally for the payer, and may be assigned differently for each payment. | `a_choice_source_stands_in_for_the_priciest_unit_of_its_group`, `a_wonders_choice_production_helps_pay_for_the_next_one` |
| R-035 | Architecture reduces a wonder's, and Masonry a blue card's, resource cost by 2 units of the owner's choice. Coin costs are untouched and the reduction floors at zero. | `architecture_rebates_two_resources_of_a_wonders_cost`, `masonry_rebates_two_resources_of_the_blue_cards_cost`, `masonry_does_not_apply_to_other_colours`, `architecture_does_not_touch_a_wonders_coin_cost`, `cost::tests::surplus_coverers_never_reduce_cost_below_zero` |
| R-036 | The rebate, the choice sources and the trade prices are resolved *together*, always in the payer's favour: a player is never charged more than the cheapest legal way to pay. The fast greedy assignment is proved equal to brute force over every assignment. | `cost::tests::greedy_matches_exhaustive`, `cost::tests::rebate_is_not_wasted_on_a_unit_a_choice_source_could_cover`, `cost::tests::rebate_covers_a_group_with_no_matching_choice_source`, `cost::tests::hand_computed_multi_resource_scenarios` |
| R-037 | The trade portion of a cost is tracked separately from the printed coin cost, because only the trade portion is redirected by the Economy token. | `the_trade_portion_is_reported_separately_from_the_coin_cost` |

## Legality (R-040 … R-049)

| id | rule | tested by |
| --- | --- | --- |
| R-040 | Only accessible (uncovered) slots may be acted on. | `covered_slots_are_never_offered`, `engine::tests::covered_slots_are_never_offered` |
| R-041 | `Build` is offered only when the player can pay the computed cost; `Discard` is always available for an accessible slot. | `engine::tests::unaffordable_builds_are_not_offered`, `the_legal_move_set_is_exactly_what_the_rules_allow` |
| R-042 | `BuildWonder` is offered for each of the acting player's drafted, unbuilt, affordable wonders, paired with each accessible slot. | `the_legal_move_set_is_exactly_what_the_rules_allow` |
| R-043 | `legal_actions` is empty exactly when the game is over, and every action it returns applies successfully. | `invariants_hold_throughout_a_game`, `engine::tests::games_terminate_with_a_result` |
| R-044 | Applying an action that is not in `legal_actions` is rejected and leaves the state untouched. | `engine::tests::illegal_actions_are_rejected_without_changing_the_state` |
| R-045 | At most seven wonders are constructed in a game; once the seventh is up, the eighth can never be built. | `engine::tests::the_seventh_wonder_closes_the_wonder_option`, `the_eighth_wonder_can_never_be_built`, `engine::tests::at_most_seven_wonders_are_ever_built`, `many_full_games_terminate_and_conserve_cards` |
| R-046 | While an effect choice is pending, only actions that resolve it are legal, and the turn does not pass. | `the_great_library_takes_a_token_that_is_out_of_play`, `the_mausoleum_rebuilds_a_destroyed_card_for_free_with_full_effects`, `invariants_hold_throughout_a_game` |

## Effects (R-050 … R-069)

| id | rule | tested by |
| --- | --- | --- |
| R-050 | Effect order on constructing a card: the card enters the city, then coin effects, then shields (which may end the game), then the scientific symbol (which may end the game or create a token choice). | `a_full_turn_sequence_produces_the_expected_events_in_order` |
| R-051 | Coin losses floor at zero: a player who cannot pay a penalty simply loses what they have. | `engine::tests::coins_never_go_negative`, `state::tests::pay_up_to_floors_at_zero`, `pushing_the_pawn_to_the_capital_wins_and_still_collects_the_loot` |
| R-052 | Discarding a card pays `2 + the discarding player's own yellow cards`, and the card goes to the shared discard pile. | `engine::tests::discarding_pays_two_plus_your_yellow_cards` |
| R-053 | The Strategy token adds a shield to every **red card** the owner constructs. It does not apply to wonders, which are not buildings. | `engine::tests::strategy_adds_a_shield_to_red_cards_but_not_to_wonders`, `a_full_turn_sequence_produces_the_expected_events_in_order` |
| R-054 | A yellow Age III "coins per building you own" effect counts the player's own cards only, including the card being built if it matches its own colour. | `engine::tests::yellow_age_three_cards_count_their_own_colour_including_themselves` |
| R-055 | A guild's **coin** payout is made once, immediately, on the higher of the two players' counts *at that instant*. | `engine::tests::a_guild_pays_coins_on_the_count_at_the_moment_it_is_built`, `a_guild_pays_coins_on_the_count_when_built_and_points_on_the_final_count` |
| R-056 | The Urbanism token pays 4 coins each time its owner builds a card free via a chain symbol. It does not pay for a Mausoleum build, which is not a chain build. | `engine::tests::a_chain_symbol_makes_a_build_free_and_pays_urbanism`, `the_mausoleum_rebuilds_a_destroyed_card_for_free_with_full_effects` |
| R-057 | Trade payments go to the bank, unless the *opponent* owns the Economy token, in which case they receive them. A card's printed coin cost is never redirected. | `engine::tests::economy_redirects_the_opponents_trade_payments`, `engine::tests::a_cards_printed_coin_cost_is_not_redirected_by_economy` |
| R-058 | Shields move the conflict pawn towards the opponent's capital, clamped at distance 9. | `engine::tests::reaching_the_opposing_capital_wins_instantly`, `pushing_the_pawn_to_the_capital_wins_and_still_collects_the_loot` |
| R-059 | A loot token triggers the first time the pawn reaches or passes its space *on that side* — once per token, not once per turn spent beyond it. One large push can collect both tokens on the same side at once. | `engine::tests::military_loot_triggers_once_per_zone_entry_not_per_turn`, `engine::tests::one_big_push_collects_both_loot_tokens_at_once` |
| R-060 | A wonder with Play Again — or any wonder, once its owner holds Theology — grants an immediate extra turn. It does not stack. | `engine::tests::an_extra_turn_keeps_the_same_player_on_move`, `engine::tests::theology_grants_an_extra_turn_on_any_wonder` |
| R-061 | An extra turn granted by the last card of an age is lost, not banked. **⚠ unverified** (see Open questions). | `engine::tests::an_extra_turn_is_lost_when_the_age_ends` |
| R-062 | A destroy effect (Circus Maximus, The Statue of Zeus) discards one opponent building of the named colour **into the discard pile**, where The Mausoleum can later retrieve it. If the opponent has no such building the effect is skipped rather than creating an unresolvable choice. | `engine::tests::destroying_a_building_puts_it_in_the_discard_pile`, `engine::tests::a_destroy_effect_with_no_target_is_skipped` |
| R-063 | Destroying a building removes its production and its scientific symbol from the victim's city, but does *not* move the conflict pawn back and does not take back a progress token already earned. | `engine::tests::destroying_a_building_puts_it_in_the_discard_pile`, `state::tests::derived_caches_match_a_full_recompute` |
| R-064 | The Mausoleum constructs one card from the discard pile for free, with all of that card's effects, including a scientific symbol that completes a pair. | `engine::tests::the_mausoleum_builds_from_the_discard_pile_for_free`, `the_mausoleum_rebuilds_a_destroyed_card_for_free_with_full_effects` |
| R-065 | The Great Library draws three of the progress tokens **set aside at setup**, never from the board; the owner keeps one and the other two leave the game. | `engine::tests::the_great_library_draws_three_of_the_set_aside_tokens`, `the_great_library_takes_a_token_that_is_out_of_play` |
| R-066 | The card spent to construct a wonder is consumed: it goes under the wonder, not into the discard pile, and is not available to The Mausoleum. | `many_full_games_terminate_and_conserve_cards`, `invariants_hold_throughout_a_game` |
| R-067 | A wonder's "opponent loses N coins" effect resolves immediately and floors at zero. | `engine::tests::coins_never_go_negative` |

## Science (R-070 … R-079)

| id | rule | tested by |
| --- | --- | --- |
| R-070 | Gathering two identical scientific symbols immediately entitles the player to one of the progress tokens on the board. | `engine::tests::the_law_token_can_complete_the_sixth_symbol`, `the_mausoleum_rebuilds_a_destroyed_card_for_free_with_full_effects` |
| R-071 | If no progress tokens remain on the board, a completed pair grants nothing and the turn passes normally. | `engine::tests::a_science_pair_with_no_tokens_left_grants_nothing` |
| R-072 | A pair is rewarded at most once per symbol, so re-completing a pair after a destroy effect grants nothing further. **⚠ unverified** (see Open questions). | `re_completing_a_science_pair_grants_no_second_token` |
| R-073 | The Law token supplies a seventh symbol (`balance`), which no card carries, so it can never form a pair — but it can complete the six-distinct set. | `engine::tests::the_law_token_can_complete_the_sixth_symbol`, `data::tests::embedded_data_parses_and_validates` |

## End of age and end of game (R-080 … R-089)

| id | rule | tested by |
| --- | --- | --- |
| R-080 | Pushing the conflict pawn to the opponent's capital (distance 9) wins immediately; nothing else is scored. | `engine::tests::reaching_the_opposing_capital_wins_instantly`, `pushing_the_pawn_to_the_capital_wins_and_still_collects_the_loot` |
| R-081 | Holding six distinct scientific symbols wins immediately, short-circuiting any pending token choice and leaving cards on the board. | `engine::tests::six_distinct_symbols_wins_instantly`, `a_sixth_distinct_symbol_ends_the_game_before_anything_else_resolves` |
| R-082 | An age ends when its structure is empty. Ages I and II are followed by dealing the next structure; Age III ends the game. | `engine::tests::games_terminate_with_a_result`, `the_weaker_player_chooses_and_may_hand_the_turn_to_the_leader` |
| R-083 | The militarily weaker player — the one the conflict pawn is on the side of — chooses who begins the next age, and may choose either player. | `engine::tests::the_weaker_player_chooses_who_begins_the_next_age`, `the_weaker_player_chooses_and_may_hand_the_turn_to_the_leader` |
| R-084 | If the pawn is centred, the player who took the last card of the age begins the next one — no choice is offered. | `engine::tests::a_centred_pawn_hands_the_next_age_to_the_last_card_taker` |
| R-085 | A guild's **victory points** are computed once, at final scoring, from the higher of the two counts *at game end* — which may differ from the count its coin payout used. | `a_guild_pays_coins_on_the_count_when_built_and_points_on_the_final_count`, `scoring::tests::guild_points_use_the_higher_of_the_two_counts` |

## Scoring (R-090 … R-099)

| id | rule | tested by |
| --- | --- | --- |
| R-090 | Victory-point categories: blue cards, green cards, yellow cards, purple cards, wonders, progress tokens, the military track, and `floor(coins / 3)`. | `scoring::tests::a_fully_hand_computed_breakdown` |
| R-091 | **Green cards score only their printed victory points.** Duel has no per-symbol or per-set science bonus; pairs are rewarded during play with progress tokens instead. | `scoring::tests::green_cards_score_only_their_printed_points` |
| R-092 | Coins score one point per complete three, rounded down. | `scoring::tests::coins_score_one_point_per_three_with_a_floor` |
| R-093 | Military points go only to the player the pawn favours: 2 at distance 1-2, 5 at 3-5, 10 at 6-8, none when centred. | `scoring::tests::military_points_go_only_to_the_leading_player` |
| R-094 | Guild majority effects compare both players and pay on the higher count: commercial, civilian, scientific, military, brown+grey together, wonders built, or `floor(coins / 3)`. | `scoring::tests::guild_points_use_the_higher_of_the_two_counts`, `scoring::tests::builders_guild_pays_two_per_wonder_of_the_leader`, `scoring::tests::moneylenders_guild_pays_on_the_richer_players_coins`, `scoring::tests::shipowners_guild_counts_brown_and_grey_together` |
| R-095 | Progress tokens score their flat points; Mathematics scores 3 per progress token owned, counting itself. | `scoring::tests::mathematics_counts_itself`, `scoring::tests::agriculture_scores_four_flat_points` |
| R-096 | A tie on total points is broken on blue (civilian) points; if those are equal too the game is a genuine draw. | `scoring::tests::civilian_victory_is_decided_on_totals_then_blue_then_draw`, `a_tie_on_totals_is_broken_on_blue_points_and_otherwise_is_a_draw` |
| R-097 | Scoring is symmetric under a seat swap: mirroring a position mirrors both breakdowns and the outcome. | `scoring_is_symmetric_under_a_seat_swap`, `a_seat_swapped_playout_mirrors_the_result` |

## Hidden information and the chance API (R-100 … R-109)

| id | rule | tested by |
| --- | --- | --- |
| R-100 | `Observation` carries no card id for a face-down slot, only the pool of candidates. | `observation::tests::face_down_slots_carry_no_card_id`, `an_observation_json_never_ties_a_card_to_a_face_down_slot` |
| R-101 | Two `GameState`s that differ only in hidden information produce **equal** `Observation`s — the operational definition of "no leak". Verified for permuting face-down cards, for changing which cards were boxed, and for permuting the not-yet-offered wonders. | `observation::tests::permuting_hidden_cards_does_not_change_the_observation`, `observation::tests::changing_which_cards_were_boxed_does_not_change_the_observation`, `observation::tests::the_undrafted_wonder_pool_hides_the_second_group`, `invariants_hold_throughout_a_game` |
| R-102 | The candidate pool for the face-down slots is always strictly larger than the number of face-down slots, because three cards of the age went back in the box unseen. A hidden card can therefore never be pinned down by elimination. | `invariants_hold_throughout_a_game` |
| R-103 | The five set-aside progress tokens are *public* information: they are the complement of the five on the board. The only randomness they carry is which three The Great Library draws. | `observation::tests::a_pending_choice_is_visible_but_the_pool_behind_it_is_not` |
| R-104 | `Observation::sample_state` produces a valid, playable `GameState` whose observation is exactly the one it came from. | `observation::tests::sampled_states_reproduce_the_observation`, `observation::tests::sampled_states_are_playable`, `invariants_hold_throughout_a_game` |
| R-105 | `chance_outcomes` enumerates every possible reveal with probabilities computed from public knowledge only; they sum to 1 and none is zero. | `engine::tests::chance_outcome_probabilities_sum_to_one`, `invariants_hold_throughout_a_game` |
| R-106 | `apply_with_outcome` forces the reveal and keeps the hidden layout valid, including the public constraint that exactly three guilds sit in the Age III structure. | `engine::tests::apply_with_outcome_forces_the_reveal`, `invariants_hold_throughout_a_game` |
| R-107 | Card conservation: every card is dealt into exactly one age slot or boxed, and once taken lives in exactly one of the two cities, the discard pile, or under a wonder. | `GameState::check_invariants`, `many_full_games_terminate_and_conserve_cards`, `engine::tests::a_played_out_game_conserves_every_card` |
| R-108 | The engine is deterministic given `(seed, actions)`: no clock, no ambient RNG, no iteration-order dependence. | `games_are_reproducible`, `tests::the_clippy_config_still_bans_nondeterminism` |
| R-109 | All three endings are reachable. | `results_are_distributed_across_all_three_victory_kinds` |

---

## Performance

`cargo bench -p duels-core`, release, Apple Silicon development machine
(recorded so a regression is visible, **not** asserted in CI, where a
threshold would be flaky):

| benchmark | time | rate |
| --- | --- | --- |
| `apply_unchecked` (one action, no event log, no legality check) | 15.6 ns | ~64 M actions/s |
| `apply_validated` (`apply_quiet`, re-derives `legal_actions` to validate) | 161 ns | ~6.2 M actions/s |
| `legal_actions` (into a reused buffer, mid-game position) | 77 ns | ~13 M calls/s |
| `copy_state` (`GameState` is `Copy`, 256 bytes) | 0.87 ns | ~1.1 G copies/s |
| `observation` (builds the public view, allocates) | 415 ns | ~2.4 M/s |
| `full_playout` (setup + ~70 decisions to a result) | 21.4 µs | ~47 k games/s |

The numbers a search agent will care about are `apply_unchecked` +
`legal_actions` + `copy_state`, i.e. about 93 ns per node, or roughly 10 M
nodes/s single-threaded. `observation` is the slow one because it allocates;
it is a per-decision UI/agent-boundary cost, not a per-node cost.

---

## Open questions

Rules the implementation had to infer. A human should spot-check these
against the physical rulebook before this engine is trusted as ground truth
for AI training.

1. **R-010, Age structure geometry and face-down positions.** The rulebook
   prints these as diagrams. The `(row, col)` tables in
   `crates/duels-core/src/layout.rs` were taken from an independent
   open-source implementation and cross-checked against the invariants the
   physical structures are known to satisfy (20 slots, the row sizes above,
   exactly 8 face down per age, exactly one accessible row at the start).
   The row *sizes* are high-confidence; the exact face-up/face-down row
   pattern (rows 2/4/6 in Ages I and II, rows 1/3/5/7 in Age III) is the part
   most worth eyeballing against the rulebook's Game Aid page.
2. **R-027, first player of Age I.** The rulebook says "choose a first
   player" without a mechanism. The engine picks one from the seed and uses
   that same player as the first drafter. If the physical game has a
   different convention (e.g. the second player drafts first), only the
   draft's seat assignment changes.
3. **R-061, an extra turn at the end of an age.** Modelled as lost rather
   than carried into the next age. This follows from "the age ends when the
   structure is empty", but is not stated in the rulebook text located.
4. **R-072, re-completing a science pair.** If a green card is destroyed and
   the pair later re-formed (only possible via The Mausoleum), the engine
   grants nothing a second time. The rulebook does not address this; the
   conservative reading was taken.
5. **R-062 vs. R-064 interaction.** Destroyed buildings are put in the
   discard pile and are Mausoleum-eligible. This is well attested in
   secondary sources but was not read off the card text.
6. **R-091, green-card scoring.** Confirmed from the rulebook that Duel has
   no science set bonus — but note that the M0 stub doc comments in
   `scoring.rs` asserted the opposite ("science-card point-scoring which is
   nonlinear per symbol count"). That was a 7 Wonders (base game) rule
   leaking in, and has been corrected.
7. **R-053, Strategy and wonders.** Applied to red cards only, on the
   reading that "building" means a card. An independent implementation we
   cross-checked applies it to wonder shields too; we believe that is a bug
   in that implementation, but the printed card text should settle it.
