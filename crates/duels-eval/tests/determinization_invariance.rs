//! The mandatory correctness property for anything in this repository that
//! touches game state: **nothing may depend on which hidden world it is
//! looking at.**
//!
//! `duels-agent-phased`, this evaluation's caller, is handed an
//! [`Observation`], which carries no hidden information, and immediately
//! invents one concrete [`GameState`] from it — the engine's chance API needs
//! a real state to work on. That invented world assigns a specific identity to
//! every face-down card. If any part of the commitment blend, any weight, or
//! any term read one of those identities, the agent would be scoring the
//! sampler's luck rather than the position, and two runs of the same decision
//! could disagree for no reason a player could see.
//!
//! So the property is asserted here, of the evaluation, rather than of any one
//! caller: every caller inherits it.
//!
//! Two independent attacks, matching `duels-strategy`'s own
//! `tests/determinization_invariance.rs`:
//!
//! 1. **Resampling.** Draw two unrelated concrete worlds from the same real
//!    observation and assert everything agrees bit for bit.
//! 2. **In-place hidden mutation.** Permute what sits behind the face-down
//!    slots of the *original* state, or swap one of the cards boxed at setup
//!    into play, with `duels_core::testing`'s helpers. Those change hidden
//!    information and nothing public at all, so they are a sharper probe than
//!    resampling, which can coincidentally draw the same world twice.
//!
//! Every comparison is exact, including on `f64`: identical inputs through
//! identical arithmetic produce identical bits, and a discrepancy of any size
//! means something read what it should not have.

use duels_core::observation::Observation;
use duels_core::testing::{swap_a_boxed_card_into_play, swap_two_hidden_cards, StateBuilder};
use duels_core::{engine, Action, GameState, Player};
use duels_eval::{
    evaluate, expected_value, rail_owner, win_probability, Blend, CoinModel, Config, CountPricing,
    EconomyModel, EvalWeights, GuildPricing, MenuFloor, MenuShieldPricing, MilitaryModel,
    PendingModel, RailModel, Root, ScienceWeights, SupplyModel, WonderModel,
};
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Two `f64`s from identical inputs through identical arithmetic must be
/// identical *bit for bit*. Deliberately not an epsilon comparison.
#[track_caller]
fn same_bits(a: f64, b: f64, what: &str) {
    assert_eq!(
        a.to_bits(),
        b.to_bits(),
        "{what}: {a} vs {b} differ in bits"
    );
}

/// Walk a real game `steps` decisions in, with a deterministic policy that
/// depends on `mix` so different cases explore different lines.
fn advance(seed: u64, steps: usize, mix: u64) -> GameState {
    let mut st = engine::new_game(seed);
    let mut rng = StdRng::seed_from_u64(seed ^ 0x5A5A);
    for _ in 0..steps {
        let actions = engine::legal_actions(&st);
        if actions.is_empty() {
            break;
        }
        let i = u64::from(st.turn())
            .wrapping_mul(mix)
            .wrapping_add(mix >> 9) as usize;
        let action = actions[i % actions.len()];
        engine::apply_quiet(&mut st, action, &mut rng).unwrap();
    }
    st
}

/// Assert that every commitment scalar, every root-fixed weight, and the full
/// expected value of every legal action agree on `a` and `b`, which must be
/// two states with the same public view.
fn assert_everything_agrees(a: &GameState, b: &GameState, ctx: &str, config: Config) {
    assert_eq!(
        a.observation(),
        b.observation(),
        "{ctx}: the two states are not publicly identical, so the test would be vacuous"
    );

    let me = a.current_player();
    let root_a = Root::new(a, me, config);
    let root_b = Root::new(b, me, config);

    for p in Player::ALL {
        let (ca, cb) = (root_a.commitment(p), root_b.commitment(p));
        same_bits(ca.c_sci, cb.c_sci, &format!("{ctx}: c_sci for {p:?}"));
        same_bits(ca.c_mil, cb.c_mil, &format!("{ctx}: c_mil for {p:?}"));
        same_bits(ca.c, cb.c, &format!("{ctx}: c for {p:?}"));
        same_bits(ca.c0_eff, cb.c0_eff, &format!("{ctx}: c0_eff for {p:?}"));
        same_bits(ca.s, cb.s, &format!("{ctx}: S(c) for {p:?}"));
        same_bits(ca.s_sci, cb.s_sci, &format!("{ctx}: S(c_sci) for {p:?}"));
        same_bits(ca.s_mil, cb.s_mil, &format!("{ctx}: S(c_mil) for {p:?}"));

        let (wa, wb) = (root_a.weights(p), root_b.weights(p));
        for (name, x, y) in [
            ("vp", wa.vp, wb.vp),
            ("liquidity", wa.liquidity, wb.liquidity),
            ("development", wa.development, wb.development),
            ("science", wa.science, wb.science),
            ("military", wa.military, wb.military),
            ("race_liquidity", wa.race_liquidity, wb.race_liquidity),
            ("economy", wa.economy, wb.economy),
        ] {
            same_bits(x, y, &format!("{ctx}: {name} weight for {p:?}"));
        }
    }
    same_bits(
        root_a.deny_scale(),
        root_b.deny_scale(),
        &format!("{ctx}: denial scale"),
    );
    assert_eq!(
        root_a.supply(),
        root_b.supply(),
        "{ctx}: the development supply statistics disagree"
    );

    // The static evaluation of the two worlds themselves...
    for p in Player::ALL {
        same_bits(
            evaluate(a, p, &root_a),
            evaluate(b, p, &root_b),
            &format!("{ctx}: evaluate for {p:?}"),
        );
        // ...and its win-probability mapping, which reads state.age() on top
        // of evaluate() itself — a second thing that must not depend on which
        // hidden world produced `a`/`b`.
        same_bits(
            win_probability(a, p, &root_a),
            win_probability(b, p, &root_b),
            &format!("{ctx}: win_probability for {p:?}"),
        );
    }

    // ...and, the thing that actually decides a move, the expected value of
    // every legal action, chance outcomes and all.
    let legal_a = engine::legal_actions(a);
    let legal_b = engine::legal_actions(b);
    assert_eq!(legal_a, legal_b, "{ctx}: legal actions");
    for &action in &legal_a {
        same_bits(
            expected_value(a, action, me, &root_a),
            expected_value(b, action, me, &root_b),
            &format!("{ctx}: expected value of {action:?}"),
        );
        same_bits(
            root_a.denial_term(action),
            root_b.denial_term(action),
            &format!("{ctx}: denial term of {action:?}"),
        );
    }
}

#[test]
fn resampling_the_same_observation_changes_nothing() {
    let mut cases = 0usize;
    let mut multi_outcome_cases = 0usize;
    for seed in 0..12u64 {
        for &steps in &[4usize, 11, 19, 27, 36, 45] {
            let st = advance(seed, steps, 0x9E37_79B9_7F4A_7C15);
            if st.is_over() {
                continue;
            }
            let obs: Observation = st.observation();
            let mut rng_a = StdRng::seed_from_u64(seed * 7919 + steps as u64);
            let mut rng_b = StdRng::seed_from_u64(0xFFFF_0000_FFFF_0000 ^ seed ^ steps as u64);
            let a = obs.sample_state(&mut rng_a);
            let b = obs.sample_state(&mut rng_b);

            if engine::legal_actions(&a)
                .iter()
                .any(|&x| engine::chance_outcomes(&a, x).len() > 1)
            {
                multi_outcome_cases += 1;
            }
            assert_everything_agrees(
                &a,
                &b,
                &format!("seed {seed} steps {steps}"),
                Config::default(),
            );
            cases += 1;
        }
    }
    assert!(cases > 20, "only {cases} positions exercised");
    assert!(
        multi_outcome_cases > 0,
        "no position offered an action with a real chance node, so the \
         expectation half of the property was never tested"
    );
}

#[test]
fn permuting_the_hidden_cards_in_place_changes_nothing() {
    let mut permuted = 0usize;
    let mut boxed = 0usize;
    for seed in 0..12u64 {
        for &steps in &[6usize, 15, 24, 33, 42] {
            let base = advance(seed, steps, 0xD1B5_4A32_D192_ED03);
            if base.is_over() {
                continue;
            }

            let mut shuffled = base;
            if swap_two_hidden_cards(&mut shuffled) {
                assert_everything_agrees(
                    &base,
                    &shuffled,
                    &format!("seed {seed} steps {steps} (hidden swap)"),
                    Config::default(),
                );
                permuted += 1;
            }

            let mut smuggled = base;
            if swap_a_boxed_card_into_play(&mut smuggled) {
                assert_everything_agrees(
                    &base,
                    &smuggled,
                    &format!("seed {seed} steps {steps} (boxed card)"),
                    Config::default(),
                );
                boxed += 1;
            }
        }
    }
    assert!(permuted > 0, "never managed to permute two hidden cards");
    assert!(boxed > 0, "never managed to swap a boxed card into play");
}

/// The property has to hold with the blend switched off too — otherwise the
/// un-blended baseline the bit-identity test rests on would itself be
/// sample-dependent.
#[test]
fn the_property_holds_with_the_blend_switched_off() {
    let config = Config {
        blend: Blend::off(),
        ..Config::default()
    };
    for seed in 0..6u64 {
        for &steps in &[9usize, 21, 38] {
            let st = advance(seed, steps, 0x2545_F491_4F6C_DD1D);
            if st.is_over() {
                continue;
            }
            let obs = st.observation();
            let mut rng_a = StdRng::seed_from_u64(seed);
            let mut rng_b = StdRng::seed_from_u64(!seed);
            let a = obs.sample_state(&mut rng_a);
            let b = obs.sample_state(&mut rng_b);
            assert_everything_agrees(
                &a,
                &b,
                &format!("blend off, seed {seed} steps {steps}"),
                config,
            );
        }
    }
}

/// The whole-decision restatement: two callers whose internal `sample_state`
/// draws differ must score every candidate identically. The action each one
/// finally returns may still differ, because ties are broken from each
/// agent's own RNG — it is the scores feeding that choice which must not
/// depend on the sample. (That an agent always returns something legal is
/// `duels-agent-phased`'s own test; this crate has no agent.)
#[test]
fn two_differently_seeded_samples_score_every_candidate_identically() {
    let st = advance(11, 18, 0x9E37_79B9_7F4A_7C15);
    assert!(!st.is_over());
    let obs = st.observation();
    let legal = engine::legal_actions(&st);
    assert!(legal.len() > 1, "test setup: need a real choice");
    let me = obs.current_player;

    let mut rng_a = StdRng::seed_from_u64(0x1111);
    let mut rng_b = StdRng::seed_from_u64(0x2222_2222_2222);
    let a = obs.sample_state(&mut rng_a);
    let b = obs.sample_state(&mut rng_b);
    let root_a = Root::new(&a, me, Config::default());
    let root_b = Root::new(&b, me, Config::default());

    for &action in &legal {
        same_bits(
            expected_value(&a, action, me, &root_a),
            expected_value(&b, action, me, &root_b),
            &format!("candidate {action:?}"),
        );
    }
}

/// Every configuration option this crate offers, exercised over the same real
/// positions.
///
/// The property is not "the default configuration does not leak"; it is "no
/// configuration does". Each new model is a different set of reads on the
/// post-outcome state, and one of them (the opponent menu) reads cards in the
/// structure, so each has to be attacked separately rather than trusted to
/// inherit the default's clean bill of health.
#[test]
fn the_property_holds_under_every_model_combination() {
    let mut configs = vec![Config::default(), Config::v1(), Config::v2()];
    for rails in [RailModel::Off, RailModel::On] {
        for menu_shield_pricing in [MenuShieldPricing::OneSided, MenuShieldPricing::Differenced] {
            for military_horizon in [None, Some(2.0), Some(5.0)] {
                configs.push(Config {
                    rails,
                    menu_shield_pricing,
                    military_horizon,
                    ..Config::default()
                });
            }
        }
    }
    for military_model in [MilitaryModel::Legacy, MilitaryModel::Band] {
        for coin_model in [CoinModel::Legacy, CoinModel::Smooth] {
            for economy_model in [EconomyModel::Legacy, EconomyModel::Bill] {
                configs.push(Config {
                    military_model,
                    coin_model,
                    economy_model,
                    ..Config::default()
                });
            }
        }
    }
    // Round four's three options, in every combination: the pending
    // resolution applies the engine's own actions to the post-outcome state
    // and the wonder budget reads the discard pile, the opponent's city and
    // the set-aside token pile, so each is a new set of reads that has to be
    // attacked rather than trusted to inherit the default's clean bill.
    for pending_model in [PendingModel::Unresolved, PendingModel::Completed] {
        for wonder_model in [WonderModel::Flat, WonderModel::Budget] {
            for destroy_replace_discount in [false, true] {
                configs.push(Config {
                    pending_model,
                    wonder_model,
                    destroy_replace_discount,
                    ..Config::default()
                });
            }
        }
    }
    // Round five's options. Guild pricing walks both cities and both players'
    // wonder holdings, the menu floor reads `discard_reward` and every unbuilt
    // wonder's cost on the post-outcome state, soft affordability widens the
    // menu to cards the mover cannot pay for — which is *more* cards read out
    // of the structure, not fewer — and the supply weighting changes the pool
    // walk itself. Every one of those is a new set of reads.
    for guild_pricing in [GuildPricing::Unpriced, GuildPricing::Projected] {
        for menu_floor in [
            MenuFloor::None,
            MenuFloor::Discard,
            MenuFloor::DiscardAndWonder,
        ] {
            for supply_model in [SupplyModel::Raw, SupplyModel::Dealt] {
                for menu_afford_soft in [0.0, 3.0] {
                    configs.push(Config {
                        guild_pricing,
                        menu_floor,
                        supply_model,
                        menu_afford_soft,
                        eval: EvalWeights {
                            guild_projection: 1.0,
                            yellow_equity: 1.0,
                            ..Config::default().eval
                        },
                        ..Config::default()
                    });
                }
            }
        }
    }
    // Round six's one option. The extra-turn premium is a constant added to a
    // wonder's flat power, keyed off `WonderDef::play_again` -- a static
    // property of the wonder, not of the position -- so it has no plausible
    // route to a hidden identity. It is swept anyway, at a magnitude far
    // larger than the sweep's peak, because "no configuration leaks" is the
    // property and a knob exempted from it on an argument is a knob nobody
    // checked. Both wonder models are included: `Budget` must be provably
    // *untouched* by it, and a leak-free run at both settings is part of
    // saying so.
    for wonder_model in [WonderModel::Flat, WonderModel::Budget] {
        for premium in [0.0, 9.0, 30.0] {
            configs.push(Config {
                wonder_model,
                eval: EvalWeights {
                    wonder_extra_turn_premium: premium,
                    ..Config::default().eval
                },
                ..Config::default()
            });
        }
    }
    // Round seven's options. The dead-race gate walks every symbol's cards
    // against the discard pile, both cities and the wonder-fodder pile; the
    // owned-token table reads the discard pile through `resource_bill`, both
    // players' wonder holdings and the chain-starter mask; the count-priced
    // menu reads the taker's own city counts; and `to_move` reads the phase and
    // `current_player`, which on an age-ending action is exactly the thing the
    // throwaway sample decides. All four are switched on together and at
    // magnitudes well past their swept peaks, because "no configuration leaks"
    // is the property.
    for count_pricing in [CountPricing::Unpriced, CountPricing::Counted] {
        for dead_race_scale in [0.0, 0.25, 1.0] {
            configs.push(Config {
                count_pricing,
                eval: EvalWeights {
                    token_equity: 2.0,
                    to_move: 12.0,
                    value_scale: 2.5,
                    science: ScienceWeights {
                        dead_race_scale,
                        ..Config::default().eval.science
                    },
                    ..Config::default().eval
                },
                ..Config::default()
            });
        }
    }
    configs.push(Config::v6());
    for (i, config) in configs.iter().enumerate() {
        for seed in 0..6u64 {
            for &steps in &[7usize, 17, 29, 43] {
                let st = advance(seed, steps, 0x1234_5678_9ABC_DEF0);
                if st.is_over() {
                    continue;
                }
                let obs = st.observation();
                let mut rng_a = StdRng::seed_from_u64(seed * 31 + steps as u64);
                let mut rng_b = StdRng::seed_from_u64(0xDEAD_BEEF ^ seed ^ (steps as u64) << 8);
                let a = obs.sample_state(&mut rng_a);
                let b = obs.sample_state(&mut rng_b);
                assert_everything_agrees(
                    &a,
                    &b,
                    &format!("config {i}, seed {seed} steps {steps}"),
                    *config,
                );
            }
        }
    }
}

/// **The age-ending case**, which is the one that actually bites.
///
/// A move that empties the structure makes the engine deal a whole new age
/// out of a deck no `Observation` can see. `Observation::sample_state` invents
/// that deck, so two samples of the *same* observation produce two different
/// Age II structures — and any term that reads a card in the structure on the
/// post-action state would score the age-ending move differently in the two
/// worlds. This is exactly the bug that was found while
/// `terms::chain_gift_exposure` was being built, and it is what
/// `menu::menu_term`'s stand-down rule exists to prevent.
///
/// So this test does not merely hope an age-ending action turns up somewhere
/// in a sweep: it builds a position with one card left, asserts that the move
/// really does end the age, asserts that the two sampled worlds really do
/// deal *different* next ages (otherwise the test would be vacuous), and only
/// then asserts the scores agree bit for bit.
#[test]
fn an_age_ending_action_scores_identically_in_two_invented_futures() {
    for (age, last_slot) in [(1u8, 19u8), (2u8, 19u8)] {
        let st = StateBuilder::new()
            .age(age)
            .open_slots(&[(last_slot, "clay-pool")])
            .built(Player::One, &["scriptorium", "palisade", "tavern"])
            .built(Player::Two, &["theater", "altar"])
            .coins(Player::One, 12)
            .coins(Player::Two, 9)
            .conflict(2)
            .current(Player::One)
            .build();

        let obs = st.observation();
        let mut rng_a = StdRng::seed_from_u64(0x1111_2222);
        let mut rng_b = StdRng::seed_from_u64(0x8888_9999);
        let a = obs.sample_state(&mut rng_a);
        let b = obs.sample_state(&mut rng_b);
        assert_eq!(a.observation(), b.observation());

        let action = Action::Discard { slot: last_slot };
        assert!(engine::legal_actions(&a).contains(&action));

        // The two worlds really do invent different next ages.
        let mut after_a = a;
        let mut after_b = b;
        let outcome = engine::chance_outcomes(&a, action);
        assert_eq!(
            outcome.len(),
            1,
            "an age-ending discard of a face-up card has no chance node of its own"
        );
        engine::apply_with_outcome(&mut after_a, action, &outcome[0].0).unwrap();
        engine::apply_with_outcome(&mut after_b, action, &outcome[0].0).unwrap();
        assert_eq!(after_a.age(), age + 1, "the move must end the age");
        let structure = |s: &GameState| -> Vec<Option<duels_core::data::CardId>> {
            (0..20u8).map(|i| s.face_up_card(i)).collect()
        };
        assert_ne!(
            structure(&after_a),
            structure(&after_b),
            "age {age}: both samples invented the same next age, so this test is vacuous"
        );

        // ...and every configuration scores the age-ending move identically
        // regardless of which future it happened to invent.
        for (i, config) in [Config::default(), Config::v1()].iter().enumerate() {
            let me = a.current_player();
            let root_a = Root::new(&a, me, *config);
            let root_b = Root::new(&b, me, *config);
            same_bits(
                expected_value(&a, action, me, &root_a),
                expected_value(&b, action, me, &root_b),
                &format!("age {age}, config {i}: the age-ending discard"),
            );
            // And the whole slate of candidates, for good measure.
            for &candidate in &engine::legal_actions(&a) {
                same_bits(
                    expected_value(&a, candidate, me, &root_a),
                    expected_value(&b, candidate, me, &root_b),
                    &format!("age {age}, config {i}: {candidate:?}"),
                );
            }
        }
    }
}

/// **The age-ending case, for the rails specifically.**
///
/// [`rail_owner`] asks whether an accessible card would end the game outright,
/// which means reading cards in the structure, which makes it the third thing
/// in this crate that could score a move by which world the throwaway sample
/// happened to invent. A move that empties the structure deals a whole new age
/// from a deck no `Observation` can see, so two samples of the *same* position
/// disagree about what is on the table afterwards.
///
/// Provoking that takes some care, because the military half of the rails
/// cannot reach across an age boundary at all: an age that ends with the pawn
/// off centre puts the engine into `Phase::ChooseFirstPlayer`, where the rails
/// stand down anyway, and an age that ends with the pawn centred leaves the
/// capital nine shields away, which is further than any single action can
/// travel. The **science** half can: a player sitting on five distinct symbols
/// with a centred pawn closes the moment a green card carrying their sixth
/// lands face up and affordable — and whether the next age deals one there is
/// precisely the sample's invention.
///
/// So both players are given five symbols, missing different ones, both of
/// which are printed on an Age II card. The test then asserts, over many
/// draws: that the two invented Age IIs really differ, that at least one of
/// them really would fire a rail if the stand-down were lifted (otherwise the
/// case is vacuous), and that every candidate scores identically anyway.
#[test]
fn an_age_ending_action_cannot_let_the_rails_read_the_next_age() {
    // Player One holds every symbol but Mortar (Age II: the Dispensary);
    // Player Two every symbol but Inkwell (Age II: the Library).
    const ONE: [&str; 5] = [
        "workshop",
        "apothecary",
        "scriptorium",
        "academy",
        "university",
    ];
    const TWO: [&str; 5] = ["laboratory", "school", "dispensary", "study", "observatory"];

    let mut provoked = 0usize;
    let mut pairs = 0usize;
    for draw in 0..80u64 {
        // The pawn is centred, so the age ends with the last card's taker
        // simply starting the next one — no `ChooseFirstPlayer` in the way.
        let st = StateBuilder::new()
            .age(1)
            .open_slots(&[(19, "clay-pool")])
            .built(Player::One, &ONE)
            .built(Player::Two, &TWO)
            .conflict(0)
            .coins(Player::One, 40)
            .coins(Player::Two, 40)
            .current(Player::One)
            .build();
        assert_eq!(st.player(Player::One).distinct_science(), 5);
        assert_eq!(st.player(Player::Two).distinct_science(), 5);

        let obs = st.observation();
        let mut rng_a = StdRng::seed_from_u64(0x51A7_E000 + draw);
        let mut rng_b = StdRng::seed_from_u64(0x0DDB_A110 ^ (draw << 17));
        let a = obs.sample_state(&mut rng_a);
        let b = obs.sample_state(&mut rng_b);
        assert_eq!(a.observation(), b.observation());

        let action = Action::Discard { slot: 19 };
        let outcome = engine::chance_outcomes(&a, action);
        assert_eq!(outcome.len(), 1);
        let (mut after_a, mut after_b) = (a, b);
        engine::apply_with_outcome(&mut after_a, action, &outcome[0].0).unwrap();
        engine::apply_with_outcome(&mut after_b, action, &outcome[0].0).unwrap();
        assert_eq!(after_a.age(), 2, "the move must end the age");
        assert_eq!(after_a.phase(), duels_core::state::Phase::Turn);

        let structure = |s: &GameState| -> Vec<Option<duels_core::data::CardId>> {
            (0..20u8).map(|i| s.face_up_card(i)).collect()
        };
        if structure(&after_a) == structure(&after_b) {
            // The two draws happened to coincide; nothing to compare.
            continue;
        }
        pairs += 1;

        // Vacuity guard: passing the *post-action* age as the root age is what
        // lifting the stand-down would mean, and at least one invented future
        // must then hand somebody a closing card.
        for world in [&after_a, &after_b] {
            if rail_owner(world, world.age(), RailModel::On).is_some() {
                provoked += 1;
            }
        }

        // ...and with the root age where it really is, both futures score
        // every candidate identically.
        for (i, config) in [Config::default(), Config::v1(), Config::v2()]
            .iter()
            .enumerate()
        {
            let me = a.current_player();
            let root_a = Root::new(&a, me, *config);
            let root_b = Root::new(&b, me, *config);
            assert_eq!(
                rail_owner(&after_a, root_a.age(), config.rails),
                None,
                "draw {draw}, config {i}: a rail read a card from an age the \
                 observation cannot see"
            );
            for &candidate in &engine::legal_actions(&a) {
                same_bits(
                    expected_value(&a, candidate, me, &root_a),
                    expected_value(&b, candidate, me, &root_b),
                    &format!("draw {draw}, config {i}: {candidate:?}"),
                );
            }
        }
    }
    assert!(
        pairs > 40,
        "only {pairs} distinct pairs of futures compared"
    );
    assert!(
        provoked > 0,
        "no invented Age II ever contained a closing card, so the stand-down \
         was never actually under test"
    );
}

// ---------------------------------------------------------------------------
// Round four: resolving a pending effect must read only public information.
// ---------------------------------------------------------------------------

/// Every configuration rounds four and five add, exercised together.
///
/// Round five's entries belong here as much as in the model sweep: the
/// `DiscardAndWonder` floor prices a wonder off the *post-outcome* state, and
/// the age-ending and pending-effect tests are the two places where that state
/// is one the throwaway sample invented.
fn round_four_configs() -> Vec<Config> {
    let all_on = EvalWeights {
        guild_projection: 1.0,
        yellow_equity: 1.0,
        ..Config::default().eval
    };
    vec![
        Config {
            pending_model: PendingModel::Completed,
            ..Config::default()
        },
        Config {
            pending_model: PendingModel::Completed,
            wonder_model: WonderModel::Budget,
            ..Config::default()
        },
        Config {
            pending_model: PendingModel::Completed,
            destroy_replace_discount: true,
            ..Config::default()
        },
        Config {
            wonder_model: WonderModel::Budget,
            ..Config::default()
        },
        Config {
            guild_pricing: GuildPricing::Projected,
            menu_floor: MenuFloor::DiscardAndWonder,
            menu_afford_soft: 3.0,
            supply_model: SupplyModel::Dealt,
            eval: all_on,
            ..Config::default()
        },
        Config {
            pending_model: PendingModel::Completed,
            wonder_model: WonderModel::Budget,
            destroy_replace_discount: true,
            guild_pricing: GuildPricing::Projected,
            menu_floor: MenuFloor::DiscardAndWonder,
            menu_afford_soft: 3.0,
            supply_model: SupplyModel::Dealt,
            eval: all_on,
            ..Config::default()
        },
        Config::v4(),
        Config::v5(),
        // Round six: the extra-turn premium, at a magnitude well past the
        // sweep's peak, through the pending-effect path as well.
        Config {
            pending_model: PendingModel::Completed,
            eval: EvalWeights {
                wonder_extra_turn_premium: 30.0,
                ..Config::default().eval
            },
            ..Config::default()
        },
        // Round seven, everything on at once, through the pending-effect path:
        // that is where a progress token is actually *chosen*, and so where the
        // owned-token table has to be provably blind to which world the sample
        // invented. `to_move` is here for the age-ending case, where
        // `current_player` is decided by the deal the sample makes up.
        Config {
            pending_model: PendingModel::Completed,
            count_pricing: CountPricing::Counted,
            eval: EvalWeights {
                token_equity: 2.0,
                to_move: 12.0,
                value_scale: 2.5,
                ..Config::default().eval
            },
            ..Config::default()
        },
        Config::v6(),
    ]
}

/// A real Age II deal, so some slots are genuinely face down and two sampled
/// worlds really do differ.
const AGE_TWO_DEAL: [&str; 20] = [
    "sawmill",
    "brickyard",
    "shelf-quarry",
    "glassblower",
    "drying-room",
    "walls",
    "horse-breeders",
    "barracks",
    "archery-range",
    "parade-ground",
    "library",
    "dispensary",
    "school",
    "laboratory",
    "courthouse",
    "statue",
    "temple",
    "aqueduct",
    "rostrum",
    "forum",
];

/// **The pending-effect case.**
///
/// [`PendingModel::Completed`] resolves a pending choice by walking the
/// engine's own `legal_actions` for it, which for the four wonders that create
/// one means reading the opponent's city (a destroy), the discard pile (the
/// Mausoleum), the board tokens (a science pair) or the three tokens the Great
/// Library drew. Every one of those is public — but "is public" is a claim,
/// and this is the test that checks it: the same position, two unrelated
/// invented worlds, and the resolved score has to agree bit for bit.
///
/// The Great Library is the interesting one, because its draw *is* random: it
/// arrives as a `chance_outcomes` enumeration over the set-aside pile, and the
/// ten three-token draws are exactly the ten chance outcomes the expectation
/// already averages over. If the resolution read the sampled state's idea of
/// the draw rather than the forced outcome's, this would catch it.
#[test]
fn resolving_a_pending_effect_reads_nothing_the_observation_cannot_see() {
    /// `(wonder, opponent's city, discard pile, tokens set aside)`.
    type Case = (
        &'static str,
        &'static [&'static str],
        &'static [&'static str],
        &'static [&'static str],
    );
    let cases: [Case; 4] = [
        ("circus-maximus", &["glassworks", "press"], &[], &[]),
        ("the-statue-of-zeus", &["lumber-yard", "clay-pit"], &[], &[]),
        ("the-mausoleum", &[], &["palace", "theater", "altar"], &[]),
        (
            "the-great-library",
            &[],
            &[],
            &["law", "theology", "mathematics", "philosophy", "urbanism"],
        ),
    ];

    let mut checked = 0usize;
    for (slug, city, discard, aside) in cases {
        let st = StateBuilder::new()
            .age(2)
            .deal(&AGE_TWO_DEAL)
            .wonders(Player::One, &[slug])
            .built(Player::Two, city)
            .discard(discard)
            .set_aside_tokens(aside)
            .board_tokens(&["strategy", "urbanism"])
            .coins(Player::One, 40)
            .coins(Player::Two, 40)
            .current(Player::One)
            .build();

        let obs = st.observation();
        let mut rng_a = StdRng::seed_from_u64(0xFEED_0001);
        let mut rng_b = StdRng::seed_from_u64(0x0BAD_C0DE);
        let a = obs.sample_state(&mut rng_a);
        let b = obs.sample_state(&mut rng_b);
        assert_eq!(a.observation(), b.observation());

        // Vacuity guard: the build really does leave a pending effect.
        let build = engine::legal_actions(&a)
            .into_iter()
            .find(|x| matches!(x, Action::BuildWonder { .. }))
            .unwrap_or_else(|| panic!("{slug}: no wonder build was legal"));
        let outcomes = engine::chance_outcomes(&a, build);
        let mut pending_seen = 0usize;
        for (outcome, _) in &outcomes {
            let mut next = a;
            engine::apply_with_outcome(&mut next, build, outcome).unwrap();
            if next.pending().is_some() {
                pending_seen += 1;
            }
        }
        assert!(
            pending_seen > 0,
            "{slug}: no chance outcome left a pending effect, so the case is vacuous"
        );
        if slug == "the-great-library" {
            assert!(
                outcomes.len() >= 10,
                "{slug}: the ten three-token draws are missing ({} outcomes)",
                outcomes.len()
            );
        }

        for (i, config) in round_four_configs().iter().enumerate() {
            assert_everything_agrees(&a, &b, &format!("{slug}, config {i}"), *config);
            checked += 1;
        }
    }
    assert!(checked >= 16, "only {checked} comparisons made");
}

/// **The age-ending case, for the pending resolution specifically.**
///
/// The dangerous combination: a candidate that empties the structure *and*
/// leaves a pending effect. Resolving it runs `finish_turn`, which then deals a
/// whole new age out of a deck no `Observation` can see — so if anything
/// downstream of the resolution read a card in that structure, the two invented
/// futures would score the move differently.
///
/// One card is left in the structure, and burying it under the wonder both
/// builds the wonder and ends the age. Both players are given the wonder in
/// turn, since a pending effect belonging to the *waiting* player takes a
/// different path through the resolution.
#[test]
fn an_age_ending_action_that_leaves_a_pending_effect_scores_identically() {
    let mut compared = 0usize;
    for owner in Player::ALL {
        for (slug, city, discard) in [
            ("the-mausoleum", &[][..], &["palace", "theater"][..]),
            ("circus-maximus", &["glassworks", "press"][..], &[][..]),
        ] {
            let st = StateBuilder::new()
                .age(1)
                .open_slots(&[(19, "clay-pool")])
                .wonders(owner, &[slug])
                .built(owner.other(), city)
                .discard(discard)
                .board_tokens(&["strategy", "urbanism"])
                .conflict(0)
                .coins(Player::One, 40)
                .coins(Player::Two, 40)
                .current(owner)
                .build();

            let obs = st.observation();
            let salt = u64::from(u32::try_from(owner.index()).unwrap_or(0));
            let mut rng_a = StdRng::seed_from_u64(0x1234_0000 ^ salt);
            let mut rng_b = StdRng::seed_from_u64(0x9876_5432 ^ salt);
            let a = obs.sample_state(&mut rng_a);
            let b = obs.sample_state(&mut rng_b);
            assert_eq!(a.observation(), b.observation());

            let build = engine::legal_actions(&a)
                .into_iter()
                .find(|x| matches!(x, Action::BuildWonder { .. }))
                .unwrap_or_else(|| panic!("{slug}: no wonder build was legal"));

            // Vacuity guards: the move really leaves a pending effect, and
            // resolving it really ends the age into two different Age IIs.
            let mut after_a = a;
            let mut after_b = b;
            let outcome = engine::chance_outcomes(&a, build);
            assert_eq!(outcome.len(), 1);
            engine::apply_with_outcome(&mut after_a, build, &outcome[0].0).unwrap();
            engine::apply_with_outcome(&mut after_b, build, &outcome[0].0).unwrap();
            assert!(after_a.pending().is_some(), "{slug}: no pending effect");
            assert_eq!(after_a.age(), 1, "the age has not turned over yet");

            let resolve = engine::legal_actions(&after_a);
            let mut done_a = after_a;
            let mut done_b = after_b;
            engine::apply_with_outcome(&mut done_a, resolve[0], &engine::Outcome::default())
                .unwrap();
            engine::apply_with_outcome(&mut done_b, resolve[0], &engine::Outcome::default())
                .unwrap();
            assert_eq!(
                done_a.age(),
                2,
                "{slug}: resolving the effect was supposed to end the age"
            );
            let structure = |s: &GameState| -> Vec<Option<duels_core::data::CardId>> {
                (0..20u8).map(|i| s.face_up_card(i)).collect()
            };
            if structure(&done_a) == structure(&done_b) {
                continue;
            }

            for (i, config) in round_four_configs().iter().enumerate() {
                let me = a.current_player();
                let root_a = Root::new(&a, me, *config);
                let root_b = Root::new(&b, me, *config);
                for &candidate in &engine::legal_actions(&a) {
                    same_bits(
                        expected_value(&a, candidate, me, &root_a),
                        expected_value(&b, candidate, me, &root_b),
                        &format!("{slug}, {owner:?}, config {i}: {candidate:?}"),
                    );
                }
                compared += 1;
            }
        }
    }
    assert!(
        compared >= 8,
        "only {compared} age-ending pending comparisons were made"
    );
}
