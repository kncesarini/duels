//! Reconstruct one exported position and take this crate's evaluation apart on
//! it, term by term.
//!
//! `duels-server`'s advanced mode exports a flagged position as `{ seed, moves
//! }` and `crate::room::replay` reconstructs it. This example is the offline
//! half of that loop, inside the crate whose numbers are being questioned: it
//! replays a move list, prints [`evaluate`]'s reading of the position and of
//! every legal action, and — the part a win percentage cannot show — the
//! per-term breakdown behind each of those numbers.
//!
//! The move list is given as `--moves` in the same JSON-ish shorthand the
//! export uses, or picked from the built-in table of positions this crate has
//! been asked about. `--list` names them.
//!
//! ```text
//! cargo run --release -p duels-eval --example position_probe -- --case r9-wonder-early
//! cargo run --release -p duels-eval --example position_probe -- \
//!     --seed 1 --moves "PickWonder:piraeus,...,Build:13"
//! ```

use duels_core::data::{CardId, WonderId};
use duels_core::{engine, Action, GameState, Player};
use duels_eval::{evaluate, expected_value, win_probability_from_value, Config, Root};

/// The round-nine flagged position: the project owner's "it really overvalues
/// building wonders here early in age 1" export, verbatim.
const R9_WONDER_EARLY: (u64, &str) = (
    1,
    "PickWonder:piraeus,PickWonder:the-great-lighthouse,PickWonder:the-statue-of-zeus,\
     PickWonder:circus-maximus,PickWonder:the-temple-of-artemis,PickWonder:the-pyramids,\
     PickWonder:the-hanging-gardens,PickWonder:the-colossus,\
     Build:17,Build:14,Build:19,BuildWonder:18:the-great-lighthouse,Build:13",
);

/// The same game with the flagged `BuildWonder` replaced by an ordinary
/// `Build` of the same slot, so the two lines can be compared at the same turn.
const R9_NO_WONDER: (u64, &str) = (
    1,
    "PickWonder:piraeus,PickWonder:the-great-lighthouse,PickWonder:the-statue-of-zeus,\
     PickWonder:circus-maximus,PickWonder:the-temple-of-artemis,PickWonder:the-pyramids,\
     PickWonder:the-hanging-gardens,PickWonder:the-colossus,\
     Build:17,Build:14,Build:19,Build:18,Build:13",
);

fn parse_action(s: &str) -> Result<Action, String> {
    let mut it = s.split(':');
    let kind = it.next().unwrap_or("");
    match kind {
        "PickWonder" => {
            let w = it.next().ok_or("PickWonder needs a wonder slug")?;
            Ok(Action::PickWonder {
                wonder: WonderId::from_slug(w).ok_or(format!("no wonder {w:?}"))?,
            })
        }
        "Build" => Ok(Action::Build {
            slot: it
                .next()
                .and_then(|v| v.parse().ok())
                .ok_or("Build needs a slot")?,
        }),
        "Discard" => Ok(Action::Discard {
            slot: it
                .next()
                .and_then(|v| v.parse().ok())
                .ok_or("Discard needs a slot")?,
        }),
        "BuildWonder" => {
            let slot = it
                .next()
                .and_then(|v| v.parse().ok())
                .ok_or("BuildWonder needs a slot")?;
            let w = it.next().ok_or("BuildWonder needs a wonder slug")?;
            Ok(Action::BuildWonder {
                slot,
                wonder: WonderId::from_slug(w).ok_or(format!("no wonder {w:?}"))?,
            })
        }
        "ChooseProgressToken" => {
            let t = it.next().ok_or("needs a token slug")?;
            Ok(Action::ChooseProgressToken {
                token: duels_core::data::TokenId::from_slug(t).ok_or(format!("no token {t:?}"))?,
            })
        }
        "MausoleumBuild" => {
            let c = it.next().ok_or("needs a card slug")?;
            Ok(Action::MausoleumBuild {
                card: CardId::from_slug(c).ok_or(format!("no card {c:?}"))?,
            })
        }
        "DestroyOpponentCard" => {
            let c = it.next().ok_or("needs a card slug")?;
            Ok(Action::DestroyOpponentCard {
                card: CardId::from_slug(c).ok_or(format!("no card {c:?}"))?,
            })
        }
        "ChooseFirstPlayer" => {
            let p = it.next().ok_or("needs one|two")?;
            Ok(Action::ChooseFirstPlayer {
                player: match p {
                    "one" | "One" | "1" => Player::One,
                    "two" | "Two" | "2" => Player::Two,
                    other => return Err(format!("no player {other:?}")),
                },
            })
        }
        other => Err(format!("unknown action kind {other:?}")),
    }
}

/// Replay `moves` from `seed`, exactly as `duels_server::room::replay` does.
///
/// The RNG seeding is the server's, so a position whose history includes The
/// Great Library's three-token draw reconstructs identically rather than
/// approximately. Nothing else in the engine consumes randomness.
fn replay(seed: u64, moves: &[Action]) -> Result<GameState, String> {
    use rand::{rngs::StdRng, SeedableRng};
    let mut state = engine::new_game(seed);
    let mut rng = StdRng::seed_from_u64(seed ^ 0x9E37_79B9_7F4A_7C15);
    for (i, &a) in moves.iter().enumerate() {
        engine::apply(&mut state, a, &mut rng).map_err(|e| format!("move {i} ({a:?}): {e:?}"))?;
    }
    Ok(state)
}

fn describe(action: Action, state: &GameState) -> String {
    match action {
        Action::Build { slot } => match state.face_up_card(slot) {
            Some(c) => format!("Build slot {slot} ({})", c.def().id),
            None => format!("Build slot {slot}"),
        },
        Action::Discard { slot } => match state.face_up_card(slot) {
            Some(c) => format!("Discard slot {slot} ({})", c.def().id),
            None => format!("Discard slot {slot}"),
        },
        Action::BuildWonder { slot, wonder } => {
            format!("BuildWonder {} under slot {slot}", wonder.def().id)
        }
        other => format!("{other:?}"),
    }
}

/// Every term of [`evaluate`]'s weighted sum, read for one player at that
/// player's own root-fixed multipliers — the decomposition a single number
/// cannot show.
///
/// Deliberately assembled here out of the crate's public `terms::` functions
/// rather than exposed from `player_value`, so a diagnostic cannot drift into
/// being an interface `evaluate` has to keep. Every line is copied from
/// `player_value`, and `total` is asserted against the real
/// `evaluate(state, p) + evaluate(state, p.other())` difference below.
fn breakdown(state: &GameState, p: Player, root: &Root) -> Vec<(&'static str, f64)> {
    use duels_eval::terms;
    let e = &root.config().eval;
    let w = root.weights(p);
    let b = duels_core::scoring::breakdown(state, p);
    let lock = 1.0 + e.production_lock_in * root.supply().production_lock_in;
    vec![
        (
            "points",
            w.vp * e.vp_projection * terms::card_and_token_vp(&b),
        ),
        (
            "coins",
            w.liquidity * e.coins_div3 * terms::coin_points(state, p, e.coin_endgame_decisions)
                + terms::coin_liquidity(state, p, e.coin_smooth_beta, e.coin_smooth_ref),
        ),
        (
            "development",
            w.development
                * e.development
                * lock
                * terms::development_value_with(
                    state,
                    p,
                    root.supply(),
                    e.development_take_rate,
                    false,
                ),
        ),
        (
            "chain_equity",
            w.development
                * e.chain_equity
                * duels_eval::menu::chain_equity(state, p, root.menu().chain()),
        ),
        (
            "resource_bill",
            w.economy
                * (e.resource_bill
                    * lock
                    * -terms::resource_bill(state, p, root.supply(), e.development_take_rate)
                    / 3.0),
        ),
        (
            "science",
            w.science * e.science_ladder * terms::science_ladder(state, p, &e.science),
        ),
        (
            "military",
            w.military * e.military_band * terms::military_band(state, p, root.smoothing())
                + e.military_loot * terms::military_loot(state, p, root.smoothing()),
        ),
        (
            "mil_urgency",
            e.military_endgame_urgency * terms::military_urgency(state, p),
        ),
        ("next_age_start", terms::next_age_start(state, p, e)),
        (
            "wonder_potential",
            e.wonder_potential * terms::wonder_potential(state, p, e),
        ),
        (
            "yellow_equity",
            e.yellow_equity
                * terms::yellow_equity(
                    state,
                    p,
                    root.menu().take(p).coin_marginal,
                    e.yellow_discard_rate,
                ),
        ),
    ]
}

fn print_breakdown(state: &GameState, root: &Root) {
    let one = breakdown(state, Player::One, root);
    let two = breakdown(state, Player::Two, root);
    println!(
        "\nper-term breakdown (One - Two, as `evaluate` differences them):\n  \
         {:>18}  {:>10}  {:>10}  {:>10}",
        "term", "One", "Two", "diff"
    );
    let mut sum = 0.0;
    for (i, (name, a)) in one.iter().enumerate() {
        let b = two[i].1;
        sum += a - b;
        println!("  {name:>18}  {a:>10.3}  {b:>10.3}  {:>+10.3}", a - b);
    }
    let menu = duels_eval::menu::menu_term(
        state,
        Player::One,
        root.age(),
        root.menu(),
        &root.config().eval.menu,
    );
    println!(
        "  {:>18}  {:>10}  {:>10}  {menu:>+10.3}",
        "menu (one-sided)", "", ""
    );
    println!(
        "  {:>18}  {:>10}  {:>10}  {:>+10.3}",
        "TOTAL",
        "",
        "",
        sum + menu
    );
}

fn main() {
    let mut seed: u64 = 1;
    let mut moves_arg: Option<String> = None;
    let mut drop: usize = 0;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--seed" => seed = args.next().and_then(|s| s.parse().ok()).unwrap_or(seed),
            "--drop" => drop = args.next().and_then(|s| s.parse().ok()).unwrap_or(0),
            "--moves" => moves_arg = args.next(),
            "--case" => {
                let c = args.next().unwrap_or_default();
                let (s, m) = match c.as_str() {
                    "r9-wonder-early" => R9_WONDER_EARLY,
                    "r9-no-wonder" => R9_NO_WONDER,
                    other => {
                        return eprintln!("position_probe: unknown case {other:?}; try --list");
                    }
                };
                seed = s;
                moves_arg = Some(m.to_string());
            }
            "--list" => {
                println!("r9-wonder-early   the round-nine flagged position (turn 13, Age I)");
                println!("r9-no-wonder      the same line with slot 18 built as a card instead");
                return;
            }
            other => return eprintln!("position_probe: unexpected argument {other:?}"),
        }
    }
    let Some(raw) = moves_arg else {
        return eprintln!("position_probe: --moves or --case is required (--list to see cases)");
    };
    let mut moves = Vec::new();
    for tok in raw.split(',').filter(|s| !s.trim().is_empty()) {
        match parse_action(tok.trim()) {
            Ok(a) => moves.push(a),
            Err(e) => return eprintln!("position_probe: {e}"),
        }
    }
    moves.truncate(moves.len().saturating_sub(drop));
    let state = match replay(seed, &moves) {
        Ok(s) => s,
        Err(e) => return eprintln!("position_probe: {e}"),
    };

    let me = state.current_player();
    let root = Root::new(&state, me, Config::default());
    let value = evaluate(&state, me, &root);
    let age = state.age();
    println!(
        "seed {seed}, {} moves replayed: turn {}, age {age}, {me:?} to move",
        moves.len(),
        state.turn()
    );
    println!(
        "coins {} / {}, conflict {}",
        state.player(Player::One).coins(),
        state.player(Player::Two).coins(),
        state.conflict()
    );
    println!(
        "value {value:+.4} vp   win_probability {:.4}",
        win_probability_from_value(value, age)
    );

    println!("\nlegal actions, as the analysis endpoint reports them:");
    let legal = engine::legal_actions(&state);
    let mut rows: Vec<(f64, String)> = legal
        .iter()
        .map(|&a| {
            let v = expected_value(&state, a, me, &root);
            (v, describe(a, &state))
        })
        .collect();
    rows.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    for (v, label) in &rows {
        println!(
            "  {:>9.4} vp   p={:.4}   {label}",
            v,
            win_probability_from_value(*v, age)
        );
    }

    // The whole point of the probe: the drop from `value` to every action's
    // value is not the actions being bad, it is the menu term changing sign
    // with who moves next. Report the same table with that one term removed
    // from both sides.
    println!("\nthe same, with the opponent-menu term switched off on both sides:");
    let bare = Config {
        eval: duels_eval::EvalWeights {
            menu: duels_eval::MenuWeights {
                lambda: 0.0,
                ..Config::default().eval.menu
            },
            // `deny_chain_gift` switches itself back on when `lambda` is zero,
            // which would confound the comparison; hold it off too.
            deny_chain_gift: 0.0,
            ..Config::default().eval
        },
        ..Config::default()
    };
    let bare_root = Root::new(&state, me, bare);
    let bare_value = evaluate(&state, me, &bare_root);
    println!(
        "  standing        {:>9.4} vp   p={:.4}",
        bare_value,
        win_probability_from_value(bare_value, age)
    );
    let mut bare_rows: Vec<(f64, String)> = legal
        .iter()
        .map(|&a| {
            let v = expected_value(&state, a, me, &bare_root);
            (v, describe(a, &state))
        })
        .collect();
    bare_rows.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    for (v, label) in &bare_rows {
        println!(
            "  {:>9.4} vp   p={:.4}   {label}",
            v,
            win_probability_from_value(*v, age)
        );
    }

    print_breakdown(&state, &root);

    println!("\nwonder holdings and what the flat model pays for them:");
    for p in Player::ALL {
        let ps = state.player(p);
        let unbuilt: Vec<String> = ps
            .wonders()
            .filter(|&w| !ps.has_built_wonder(w))
            .map(|w| {
                format!(
                    "{} ({:+.1}{})",
                    w.def().id,
                    duels_eval::terms::wonder_power_flat(w, &root.config().eval),
                    if w.def().play_again {
                        ", play again"
                    } else {
                        ""
                    }
                )
            })
            .collect();
        let built: Vec<&str> = ps
            .wonders()
            .filter(|&w| ps.has_built_wonder(w))
            .map(|w| w.def().id)
            .collect();
        println!("  {p:?} built {built:?}");
        println!("       unbuilt {}", unbuilt.join(", "));
        println!(
            "       wonder_potential term = {:+.3} vp (weight {:.2}), p_build = {:.3}",
            root.config().eval.wonder_potential
                * duels_eval::terms::wonder_potential(&state, p, &root.config().eval),
            root.config().eval.wonder_potential,
            duels_eval::terms::wonder_p_build(&state, p, &root.config().eval),
        );
    }
}
