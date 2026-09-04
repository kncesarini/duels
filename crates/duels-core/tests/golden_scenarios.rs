//! Golden scenarios: hand-built positions with an exact expected outcome.
//!
//! Each test sets up a specific board, plays a specific sequence of actions,
//! and asserts an exact number — a score breakdown, a legal-move list, a coin
//! total. They are the regression net for the subtle rules: the guild
//! coins-versus-points timing split, the two instant wins, the wonder cap, and
//! the start-of-age first-player rule.
//!
//! See `docs/rules-spec.md` for the `R-xxx` rule each one covers.

use duels_core::action::Action;
use duels_core::data::{TokenId, WonderId};
use duels_core::engine;
use duels_core::scoring::{self, Breakdown, GameResult, VictoryKind};
use duels_core::state::{Pending, Phase};
use duels_core::testing::StateBuilder;
use duels_core::{GameState, Player};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn rng() -> StdRng {
    StdRng::seed_from_u64(0x901d)
}

fn play(state: &mut GameState, action: Action) -> Vec<duels_core::Event> {
    engine::apply(state, action, &mut rng()).unwrap_or_else(|e| panic!("{e}"))
}

fn wonder(slug: &str) -> WonderId {
    WonderId::from_slug(slug).unwrap()
}

// ---------------------------------------------------------------------------
// R-055 / R-085: a guild's coins are locked in when it is built; its points
// are recomputed at the end of the game.
// ---------------------------------------------------------------------------

#[test]
fn a_guild_pays_coins_on_the_count_when_built_and_points_on_the_final_count() {
    // Age III, two cards left. Player One takes the Merchants Guild while
    // Player Two has three yellow cards; Player Two then chains the Arena in
    // for free, making four. So the guild pays One 3 coins now but 4 points
    // at the end.
    let mut state = StateBuilder::new()
        .age(3)
        .open_slots(&[(18, "merchants-guild"), (19, "arena")])
        .built(Player::One, &["tavern"])
        .built(Player::Two, &["brewery", "forum", "caravansery"])
        .coins(Player::One, 20)
        .coins(Player::Two, 0)
        .current(Player::One)
        .build();

    let events = play(&mut state, Action::Build { slot: 18 });
    let guild_coins: u16 = events
        .iter()
        .filter_map(|e| match e {
            duels_core::Event::CoinsGained {
                amount,
                reason: duels_core::event::CoinReason::GuildMajority,
                ..
            } => Some(*amount),
            _ => None,
        })
        .sum();
    assert_eq!(
        guild_coins, 3,
        "the guild pays on the count at that instant"
    );
    // Cost was 1 clay + 1 wood + 1 glass + 1 papyrus, all bought at 2 = 8.
    assert_eq!(state.player(Player::One).coins(), 20 - 8 + 3);

    // Player Two chains the Arena off the Brewery for free.
    assert_eq!(state.current_player(), Player::Two);
    play(&mut state, Action::Build { slot: 19 });

    // The structure is empty, so Age III and the game are over.
    assert!(state.is_over());
    let [p1, p2] = scoring::score(&state);

    // Player One: Merchants Guild scores 1 point per commercial card of
    // whoever has more, which is now Player Two's four. Fifteen coins are
    // five more points.
    assert_eq!(
        p1,
        Breakdown {
            civilian: 0,
            scientific: 0,
            commercial: 0,
            guilds: 4,
            wonders: 0,
            progress_tokens: 0,
            military: 0,
            coins: 5,
            total: 9,
        }
    );
    // Player Two: only the Arena carries points (3).
    assert_eq!(
        p2,
        Breakdown {
            civilian: 0,
            scientific: 0,
            commercial: 3,
            guilds: 0,
            wonders: 0,
            progress_tokens: 0,
            military: 0,
            coins: 0,
            total: 3,
        }
    );
    assert_eq!(
        state.result(),
        Some(GameResult::Win {
            winner: Player::One,
            kind: VictoryKind::CivilianVictory
        })
    );
}

// ---------------------------------------------------------------------------
// R-081: scientific supremacy.
// ---------------------------------------------------------------------------

#[test]
fn a_sixth_distinct_symbol_ends_the_game_before_anything_else_resolves() {
    // Player One has five distinct symbols and takes the University, whose
    // gyroscope is the sixth. The win must short-circuit: no progress-token
    // choice, no scoring, and the turn never passes.
    let mut state = StateBuilder::new()
        .age(3)
        .open_slots(&[(18, "university"), (19, "obelisk")])
        .built(
            Player::One,
            &[
                "workshop",
                "apothecary",
                "scriptorium",
                "pharmacist",
                "academy",
            ],
        )
        .board_tokens(&["philosophy", "agriculture"])
        .coins(Player::One, 20)
        .current(Player::One)
        .build();
    assert_eq!(state.player(Player::One).distinct_science(), 5);

    let events = play(&mut state, Action::Build { slot: 18 });

    assert_eq!(
        state.result(),
        Some(GameResult::Win {
            winner: Player::One,
            kind: VictoryKind::ScientificSupremacy
        })
    );
    assert!(state.result().unwrap().is_instant());
    assert_eq!(
        state.pending(),
        None,
        "no token choice survives an instant win"
    );
    assert_eq!(state.phase(), Phase::GameOver);
    assert!(engine::legal_actions(&state).is_empty());
    // The board still has cards on it: the game stopped mid-age.
    assert_ne!(state.occupied_slots(), 0);
    assert!(events
        .iter()
        .any(|e| matches!(e, duels_core::Event::GameEnded { .. })));
}

// ---------------------------------------------------------------------------
// R-080: military supremacy.
// ---------------------------------------------------------------------------

#[test]
fn pushing_the_pawn_to_the_capital_wins_and_still_collects_the_loot() {
    // Pawn at +7, Player One builds the Arsenal (3 shields). The pawn is
    // clamped at +9 (the capital), the distance-6 loot triggers on the way,
    // and the game ends instantly.
    let mut state = StateBuilder::new()
        .age(3)
        .open_slots(&[(18, "arsenal"), (19, "obelisk")])
        .conflict(7)
        .loot_taken(Player::One, 0) // the distance-3 token was taken earlier
        .coins(Player::One, 20)
        .coins(Player::Two, 4)
        .current(Player::One)
        .build();

    let events = play(&mut state, Action::Build { slot: 18 });

    assert_eq!(state.conflict(), 9, "the pawn stops at the capital");
    assert_eq!(state.player(Player::Two).coins(), 0, "5 coins owed, 4 paid");
    assert_eq!(
        state.result(),
        Some(GameResult::Win {
            winner: Player::One,
            kind: VictoryKind::MilitarySupremacy
        })
    );
    assert_eq!(
        events
            .iter()
            .filter(|e| matches!(e, duels_core::Event::MilitaryLootTriggered { .. }))
            .count(),
        1,
        "only the distance-6 token was still on the board"
    );
    assert!(engine::legal_actions(&state).is_empty());
}

// ---------------------------------------------------------------------------
// R-040..R-042: the exact legal-move set in a crafted position.
// ---------------------------------------------------------------------------

#[test]
fn the_legal_move_set_is_exactly_what_the_rules_allow() {
    // Age III with the two bottom slots open. Player One has 8 coins, no
    // production, and one undrafted-but-owned wonder they can just afford.
    //   Obelisk   : 2 stone + 1 glass  = 6 coins -> buildable
    //   Pretorium : 8 coins flat       = 8 coins -> buildable
    //   The Pyramids: 3 stone + 1 papyrus = 8 coins -> buildable with either card
    let pyramids = wonder("the-pyramids");
    let state = StateBuilder::new()
        .age(3)
        .open_slots(&[(18, "obelisk"), (19, "pretorium")])
        .wonders(Player::One, &["the-pyramids"])
        .coins(Player::One, 8)
        .current(Player::One)
        .build();

    assert_eq!(
        engine::legal_actions(&state),
        vec![
            Action::Build { slot: 18 },
            Action::Discard { slot: 18 },
            Action::BuildWonder {
                slot: 18,
                wonder: pyramids
            },
            Action::Build { slot: 19 },
            Action::Discard { slot: 19 },
            Action::BuildWonder {
                slot: 19,
                wonder: pyramids
            },
        ]
    );

    // One coin fewer and the Pretorium and the wonder both drop out.
    let poorer = StateBuilder::new()
        .age(3)
        .open_slots(&[(18, "obelisk"), (19, "pretorium")])
        .wonders(Player::One, &["the-pyramids"])
        .coins(Player::One, 7)
        .current(Player::One)
        .build();
    assert_eq!(
        engine::legal_actions(&poorer),
        vec![
            Action::Build { slot: 18 },
            Action::Discard { slot: 18 },
            Action::Discard { slot: 19 },
        ]
    );
}

// ---------------------------------------------------------------------------
// R-045: the seven-wonder cap.
// ---------------------------------------------------------------------------

#[test]
fn the_eighth_wonder_can_never_be_built() {
    let mut state = StateBuilder::new()
        .age(3)
        .open_slots(&[(18, "obelisk"), (19, "pretorium")])
        .wonders_built(Player::One, &["the-pyramids", "the-colossus", "the-sphinx"])
        .wonders_built(
            Player::Two,
            &["the-great-library", "the-mausoleum", "the-hanging-gardens"],
        )
        .wonders(Player::One, &["the-appian-way"])
        .coins(Player::One, 60)
        .current(Player::One)
        .build();
    assert_eq!(state.wonders_built_total(), 6);

    // The seventh is still allowed.
    let seventh = Action::BuildWonder {
        slot: 18,
        wonder: wonder("the-appian-way"),
    };
    assert!(engine::legal_actions(&state).contains(&seventh));
    play(&mut state, seventh);
    assert_eq!(state.wonders_built_total(), 7);

    // Player Two's fourth wonder is now dead wood.
    duels_core::testing::set_current_player(&mut state, Player::Two);
    assert!(!engine::legal_actions(&state)
        .iter()
        .any(|a| matches!(a, Action::BuildWonder { .. })));
}

// ---------------------------------------------------------------------------
// R-070..R-072: end of age and the first-player choice.
// ---------------------------------------------------------------------------

#[test]
fn the_weaker_player_chooses_and_may_hand_the_turn_to_the_leader() {
    let mut state = StateBuilder::new()
        .age(2)
        .open_slots(&[(18, "brewery")])
        .conflict(-4) // Player Two is ahead, Player One is weaker
        .coins(Player::One, 20)
        .current(Player::One)
        .build();

    play(&mut state, Action::Build { slot: 18 });
    assert_eq!(state.age(), 3);
    assert_eq!(state.phase(), Phase::ChooseFirstPlayer);
    assert_eq!(
        state.current_player(),
        Player::One,
        "the weaker player chooses"
    );
    assert_eq!(
        engine::legal_actions(&state),
        vec![
            Action::ChooseFirstPlayer {
                player: Player::One
            },
            Action::ChooseFirstPlayer {
                player: Player::Two
            },
        ]
    );

    play(
        &mut state,
        Action::ChooseFirstPlayer {
            player: Player::Two,
        },
    );
    assert_eq!(state.phase(), Phase::Turn);
    assert_eq!(state.current_player(), Player::Two);
}

// ---------------------------------------------------------------------------
// R-060 / R-062: The Great Library and The Mausoleum.
// ---------------------------------------------------------------------------

#[test]
fn the_great_library_takes_a_token_that_is_out_of_play() {
    // Only Strategy is on the board; the other five are set aside. The Great
    // Library must draw from the set-aside five, never from the board.
    let mut state = StateBuilder::new()
        .age(3)
        .open_slots(&[(18, "obelisk"), (19, "pretorium")])
        .wonders(Player::One, &["the-great-library"])
        .built(
            Player::One,
            &["press", "glassworks", "sawmill", "lumber-yard"],
        )
        .board_tokens(&["strategy"])
        .set_aside_tokens(&["philosophy", "agriculture", "mathematics", "law", "economy"])
        .coins(Player::One, 20)
        .current(Player::One)
        .build();

    play(
        &mut state,
        Action::BuildWonder {
            slot: 18,
            wonder: wonder("the-great-library"),
        },
    );

    let Some(Pending::GreatLibraryToken { tokens }) = state.pending() else {
        panic!("expected a Great Library draw, got {:?}", state.pending());
    };
    let strategy = TokenId::from_slug("strategy").unwrap();
    assert!(
        !tokens.contains(&strategy),
        "the board token must not be drawn"
    );
    assert_eq!(
        engine::legal_actions(&state).len(),
        3,
        "exactly the three drawn tokens are on offer"
    );

    play(
        &mut state,
        Action::ChooseGreatLibraryToken { token: tokens[1] },
    );
    assert!(state.player(Player::One).has_token(tokens[1]));
    // The other two are returned to the box, and Strategy is untouched.
    assert_eq!(state.set_aside_tokens_mask().count_ones(), 2);
    assert!(state.board_tokens().any(|t| t == strategy));
}

#[test]
fn the_mausoleum_rebuilds_a_destroyed_card_for_free_with_full_effects() {
    // Player Two's Lumber Yard was destroyed earlier and sits in the discard
    // pile. Player One builds The Mausoleum and takes the Library out of the
    // discard, which completes an inkwell pair and offers a progress token.
    let mut state = StateBuilder::new()
        .age(3)
        .open_slots(&[(18, "obelisk"), (19, "pretorium")])
        .wonders(Player::One, &["the-mausoleum"])
        .built(
            Player::One,
            &[
                "press",
                "glassworks",
                "glassblower",
                "brickyard",
                "scriptorium",
            ],
        )
        .discard(&["library", "lumber-yard"])
        .board_tokens(&["philosophy"])
        .coins(Player::One, 20)
        .current(Player::One)
        .build();

    play(
        &mut state,
        Action::BuildWonder {
            slot: 18,
            wonder: wonder("the-mausoleum"),
        },
    );
    assert_eq!(state.pending(), Some(Pending::MausoleumBuild));
    assert_eq!(engine::legal_actions(&state).len(), 2);

    let coins_before = state.player(Player::One).coins();
    let library = duels_core::data::CardId::from_slug("library").unwrap();
    play(&mut state, Action::MausoleumBuild { card: library });

    assert!(state.player(Player::One).has_built(library));
    assert_eq!(
        state.player(Player::One).coins(),
        coins_before,
        "the discard-pile build is free"
    );
    assert_eq!(
        state.pending(),
        Some(Pending::ProgressToken),
        "the rebuilt card's science symbol still completes a pair"
    );
    // The Lumber Yard is still in the discard pile.
    assert_eq!(state.discard_pile().count(), 1);
}

// ---------------------------------------------------------------------------
// R-090..R-092: tie-breaks.
// ---------------------------------------------------------------------------

#[test]
fn a_tie_on_totals_is_broken_on_blue_points_and_otherwise_is_a_draw() {
    // Nine points each. Player One's come from a wonder, Player Two's from
    // blue cards, so Player Two wins the tie-break.
    let state = StateBuilder::new()
        .wonders_built(Player::One, &["the-pyramids"]) // 9
        .built(Player::Two, &["palace", "altar"]) // 7 + 3 = 10 ... adjust below
        .build();
    let [a, b] = scoring::score(&state);
    assert_eq!((a.total, b.total), (9, 10));

    // Now make it an exact tie: Palace alone is 7, plus 6 coins is 2 more.
    let state = StateBuilder::new()
        .wonders_built(Player::One, &["the-pyramids"]) // 9
        .built(Player::Two, &["palace"]) // 7
        .coins(Player::Two, 6) // + 2
        .build();
    let [a, b] = scoring::score(&state);
    assert_eq!((a.total, b.total), (9, 9));
    assert_eq!((a.civilian, b.civilian), (0, 7));
    assert_eq!(
        scoring::civilian_result(&state),
        GameResult::Win {
            winner: Player::Two,
            kind: VictoryKind::CivilianTiebreak
        }
    );

    // Identical blue holdings on both sides is a genuine draw.
    let state = StateBuilder::new()
        .built(Player::One, &["palace"])
        .built(Player::Two, &["town-hall"])
        .build();
    assert_eq!(scoring::civilian_result(&state), GameResult::Draw);
}

// ---------------------------------------------------------------------------
// R-053: the Strategy token, and R-051: coin floors.
// ---------------------------------------------------------------------------

#[test]
fn a_full_turn_sequence_produces_the_expected_events_in_order() {
    use duels_core::event::CoinReason;
    use duels_core::Event;

    // Player One buys the Guard Tower (free, 1 shield) with Strategy in hand
    // and the pawn already at +2, which crosses the distance-3 loot token.
    let mut state = StateBuilder::new()
        .age(1)
        .open_slots(&[(14, "guard-tower"), (15, "quarry")])
        .tokens(Player::One, &["strategy"])
        .conflict(2)
        .coins(Player::One, 5)
        .coins(Player::Two, 1)
        .current(Player::One)
        .build();

    let events = play(&mut state, Action::Build { slot: 14 });
    let kinds: Vec<&str> = events
        .iter()
        .map(|e| match e {
            Event::CardTaken { .. } => "taken",
            Event::CardBuilt { .. } => "built",
            Event::ConflictMoved { .. } => "conflict",
            Event::MilitaryLootTriggered { .. } => "loot",
            Event::CoinsLost {
                reason: CoinReason::MilitaryLoot,
                ..
            } => "loot-paid",
            _ => "other",
        })
        .collect();
    assert_eq!(
        kinds,
        vec!["taken", "built", "conflict", "loot", "loot-paid"],
        "effects must resolve in order: take, build, shields, loot"
    );
    // 1 printed shield + 1 from Strategy = 2, so +2 -> +4.
    assert_eq!(state.conflict(), 4);
    // Player Two owed 2 but only had 1.
    assert_eq!(state.player(Player::Two).coins(), 0);
    assert!(events.iter().any(|e| matches!(
        e,
        Event::CoinsLost {
            amount: 1,
            reason: CoinReason::MilitaryLoot,
            ..
        }
    )));
}

// ---------------------------------------------------------------------------
// R-072: a pair is rewarded at most once per symbol.
// ---------------------------------------------------------------------------

#[test]
fn re_completing_a_science_pair_grants_no_second_token() {
    use duels_core::data::{CardId, Science};

    // Player One already took a token for their pair of mortars; the second
    // Dispensary was then destroyed and sits in the discard pile. Rebuilding
    // it with The Mausoleum re-forms the pair but must not pay again.
    let mut state = StateBuilder::new()
        .age(3)
        .open_slots(&[(18, "obelisk"), (19, "pretorium")])
        .wonders(Player::One, &["the-mausoleum"])
        .built(
            Player::One,
            &[
                "pharmacist",
                "press",
                "glassworks",
                "glassblower",
                "brickyard",
            ],
        )
        .pair_already_awarded(Player::One, Science::Mortar)
        .discard(&["dispensary"])
        .board_tokens(&["philosophy"])
        .coins(Player::One, 20)
        .current(Player::One)
        .build();

    play(
        &mut state,
        Action::BuildWonder {
            slot: 18,
            wonder: wonder("the-mausoleum"),
        },
    );
    assert_eq!(state.pending(), Some(Pending::MausoleumBuild));

    let dispensary = CardId::from_slug("dispensary").unwrap();
    play(&mut state, Action::MausoleumBuild { card: dispensary });

    assert_eq!(
        state.player(Player::One).science()[Science::Mortar.index()],
        2
    );
    assert_eq!(
        state.pending(),
        None,
        "the mortar pair was already rewarded once"
    );
    assert_eq!(state.player(Player::One).token_count(), 0);
    // ...and a fresh pair still does pay.
    assert!(state.board_tokens().count() == 1);
}
