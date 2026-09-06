//! **Does `phased` ever miss a terminal rail?**
//!
//! The win rate against a stronger opponent is not the number that says
//! whether the rails in `duels-agent-phased`'s `rails` module work. A rail is
//! a *guarantee*: take an available win, never hand one over, never pass up an
//! undeniable close. A guarantee is audited, not sampled — the target is
//! 100%, and any counterexample is a bug rather than a data point.
//!
//! So this replays real games and, at every one of `phased`'s own decisions,
//! recomputes from the **engine** — never from the agent's own reads — what
//! was actually available:
//!
//! ```text
//! Rail A   a candidate ends the game in this player's favour
//!          -> was one of them chosen?                          must be 100%
//! Rail B   some candidate removes every game-ending action the
//!          opponent would otherwise have had, and some other
//!          candidate does not
//!          -> was a removing candidate chosen?                  must be 100%
//! Rail C   some candidate leaves this player a close that no
//!          single opposing reply can take away, and Rail B did
//!          not already decide the move
//!          -> was one of them chosen?                           must be 100%
//! ```
//!
//! and, separately, the failure that actually costs games: **every military or
//! scientific supremacy loss**, checked against whether a Rail-B-resolving
//! candidate was on the table at the last decision before it. That count must
//! be zero.
//!
//! # Why this lives in `duels-arena` and not in the agent crate
//!
//! The audit has to run `phased` against `alphabeta` and `mcts-uct` as well as
//! against itself, and this repository's standing invariant is that **no agent
//! crate depends on another agent crate**. `duels-arena` is where every agent
//! is already visible, and where `matchup_profile.rs` — the other
//! cross-agent diagnostic — already lives.
//!
//! # Cost, and the one shortcut taken
//!
//! Ground truth for Rail B means applying every opposing reply to every
//! candidate's post-state; for Rail C it means applying every reply *and* then
//! every one of this player's answers. That is far too much to do at every
//! decision of a few hundred games, so it is done only at decisions that could
//! possibly matter — where somebody is within one action's shields of a
//! capital, or holds five distinct symbols. Both conditions are functions of
//! the *pre-reveal* part of the position, which no chance outcome changes, so
//! skipping the rest cannot hide a case: a position that is not hot has no
//! game-ending action available to anybody, in any outcome.
//!
//! ```text
//! cargo run --release -p duels-arena --example rail_audit -- phased phased 100
//! cargo run --release -p duels-arena --example rail_audit -- \
//!     phased mcts-uct 100 --seed 5001 --budget nodes:2000
//! ```
//!
//! Arguments: the audited specification, the opponent's, then optionally the
//! number of games (default 100, rounded to an even number of paired seeds),
//! `--seed <N>` and `--budget <spec>`.

use duels_agent_phased::rails::first_resolving_reply;
use duels_agent_phased::{rail_owner, RailModel};
use duels_agents_api::Budget;
use duels_arena::agent_spec::make_agent_from_spec;
use duels_arena::match_runner::parse_budget;
use duels_core::scoring::VictoryKind;
use duels_core::state::Phase;
use duels_core::{engine, Action, GameResult, GameState, Player};
use rand::{rngs::StdRng, SeedableRng};

/// Salts matching `match_runner`'s, so an audit run and a `duels-arena match`
/// run of the same seeds see the same games.
const AGENT_A_SALT: u64 = 0xA011_7A9E_5B21_0001;
const AGENT_B_SALT: u64 = 0xB022_8C3F_6D42_0002;
const ENGINE_RNG_SALT: u64 = 0x9E37_79B9_7F4A_7C15;

/// Distinct scientific symbols that win the game outright.
const SYMBOLS_TO_WIN: u8 = 6;

/// The most shields any single action can add: the biggest printed red card
/// plus the Strategy token's bonus, or the biggest shield-granting wonder.
/// Read off the data rather than written down.
fn max_single_shield_gain() -> u8 {
    let card = (0..duels_core::data::NUM_CARDS)
        .map(|i| duels_core::data::CardId::from_index(i).def().shields)
        .max()
        .unwrap_or(0);
    let wonder = duels_core::data::WonderId::all()
        .map(|w| w.def().shields)
        .max()
        .unwrap_or(0);
    card.saturating_add(1).max(wonder)
}

#[derive(Default, Clone, Copy)]
struct Audit {
    decisions: u64,
    hot_decisions: u64,
    /// Rail A: a winning candidate existed / was taken.
    a_available: u64,
    a_taken: u64,
    /// Rail B: a resolving candidate existed / was taken.
    b_available: u64,
    b_taken: u64,
    /// Rail C: an undeniable close was creatable / was created.
    c_available: u64,
    c_taken: u64,
    /// The subset of those where the closing action was *already on the
    /// table* when the candidate was played, rather than one the opponent's
    /// own reply had to uncover first. Only this subset is something a rail
    /// with no search can see.
    c_now_available: u64,
    c_now_taken: u64,
    /// How often the agent's own rail *fired* on a candidate's post-state,
    /// and how often ground truth agreed — the precision half, which matters
    /// more than recall: a rail that fires wrongly misvalues a move by 500
    /// points. `b_*` is "the mover can close" (Rails B and C′), `c_*` is "the
    /// waiter holds a close no reply removes" (Rail C).
    b_fired: u64,
    b_fired_correct: u64,
    c_fired: u64,
    c_fired_correct: u64,
    /// Supremacy losses, and the ones that were avoidable at the last
    /// decision.
    supremacy_losses: u64,
    avoidable_supremacy_losses: u64,
}

impl Audit {
    fn merge(&mut self, other: &Audit) {
        self.decisions += other.decisions;
        self.hot_decisions += other.hot_decisions;
        self.a_available += other.a_available;
        self.a_taken += other.a_taken;
        self.b_available += other.b_available;
        self.b_taken += other.b_taken;
        self.c_available += other.c_available;
        self.c_taken += other.c_taken;
        self.c_now_available += other.c_now_available;
        self.c_now_taken += other.c_now_taken;
        self.b_fired += other.b_fired;
        self.b_fired_correct += other.b_fired_correct;
        self.c_fired += other.c_fired;
        self.c_fired_correct += other.c_fired_correct;
        self.supremacy_losses += other.supremacy_losses;
        self.avoidable_supremacy_losses += other.avoidable_supremacy_losses;
    }
}

/// Whether any race is close enough that a single action could settle it.
///
/// Neither half depends on which face-down card a chance outcome turns over,
/// which is what makes it a sound filter rather than a sampling shortcut.
fn hot(state: &GameState) -> bool {
    let reach = i16::from(max_single_shield_gain());
    let cap = i16::from(duels_core::data::military().capital_distance);
    let pawn = i16::from(state.conflict()).abs();
    if cap - pawn <= reach {
        return true;
    }
    Player::ALL
        .iter()
        .any(|&p| state.player(p).distinct_science() + 1 >= SYMBOLS_TO_WIN)
}

/// Every state `action` can lead to, with no probabilities attached: the
/// audit asks "in every outcome" and "in some outcome", never "on average".
fn futures(state: &GameState, action: Action) -> Vec<GameState> {
    engine::chance_outcomes(state, action)
        .into_iter()
        .filter_map(|(outcome, _)| {
            let mut next = *state;
            engine::apply_with_outcome(&mut next, action, &outcome)
                .ok()
                .map(|_| next)
        })
        .collect()
}

/// Any win at all, however it arrives. Rail A is "take the win", and a
/// civilian victory on the last card of Age III is as much a win as a race.
fn is_win(state: &GameState, p: Player) -> bool {
    matches!(state.result(), Some(GameResult::Win { winner, .. }) if winner == p)
}

/// A win by *race*, which is the only kind the rails claim to see coming: a
/// civilian victory is not something one action creates out of nothing, and
/// not something a single reply can take away.
fn is_race_win(state: &GameState, p: Player) -> bool {
    matches!(
        state.result(),
        Some(GameResult::Win {
            winner,
            kind: VictoryKind::MilitarySupremacy | VictoryKind::ScientificSupremacy,
        }) if winner == p
    )
}

/// Whether `p` can end the game outright from `state`, checked by applying
/// every legal action.
fn can_close(state: &GameState, p: Player) -> bool {
    if state.is_over() || state.current_player() != p || state.phase() != Phase::Turn {
        return false;
    }
    engine::legal_actions(state)
        .into_iter()
        .any(|a| futures(state, a).iter().all(|s| is_race_win(s, p)))
}

/// Whether the player to move in `state` can close, in **any** of the futures
/// `action` produces.
fn hands_over_a_close(state: &GameState, action: Action, opponent: Player) -> bool {
    futures(state, action)
        .into_iter()
        .any(|s| can_close(&s, opponent))
}

/// Whether `holder` is left with a close that survives every single reply, in
/// **every** future `action` produces.
///
/// Ground truth, two plies of the real engine: for each reply the opponent can
/// make, and each way that reply's own randomness resolves, is `holder` still
/// able to end the game?
fn creates_an_undeniable_close(state: &GameState, action: Action, holder: Player) -> bool {
    let worlds = futures(state, action);
    if worlds.is_empty() {
        return false;
    }
    worlds.into_iter().all(|s| {
        if s.is_over() || s.current_player() == holder || s.phase() != Phase::Turn {
            return false;
        }
        if !can_close_after_any_reply(&s, holder) {
            return false;
        }
        true
    })
}

fn can_close_after_any_reply(state: &GameState, holder: Player) -> bool {
    let replies = engine::legal_actions(state);
    if replies.is_empty() {
        return false;
    }
    // It must survive *every* reply, in every one of that reply's outcomes.
    replies.into_iter().all(|r| {
        futures(state, r)
            .iter()
            .all(|s| can_close_eventually(s, holder, PENDING_PLIES))
    })
}

/// How many further plies the ground truth walks before giving up on reaching
/// `holder`'s turn.
///
/// A reply does not always hand the turn straight back: the Great Library
/// draws three tokens and the builder picks one, a destroy effect asks which
/// building, a play-again wonder simply gives them the move again. Asking
/// "can the holder close?" in the middle of that answers "no" for a reason
/// that has nothing to do with whether the close survives, which is exactly
/// the false positive this walk exists to stop the audit from reporting.
const PENDING_PLIES: u8 = 3;

/// Whether `p` can still end the game outright once the position has been
/// walked forward to a turn of theirs.
///
/// Choices made by `p` are theirs to make, so one good line is enough;
/// choices made by the opponent have to *all* leave the close alive, which is
/// what "undeniable" means.
fn can_close_eventually(state: &GameState, p: Player, depth: u8) -> bool {
    if state.is_over() {
        return false;
    }
    if state.current_player() == p && state.phase() == Phase::Turn && state.pending().is_none() {
        return can_close(state, p);
    }
    if depth == 0 {
        return false;
    }
    let actions = engine::legal_actions(state);
    if actions.is_empty() {
        return false;
    }
    let mine = state.current_player() == p;
    let mut any = false;
    for a in actions {
        let ok = futures(state, a)
            .iter()
            .all(|s| can_close_eventually(s, p, depth - 1));
        if mine {
            any |= ok;
        } else if !ok {
            return false;
        }
    }
    !mine || any
}

/// The same question, restricted to closes that were **already available**
/// when `action` was played.
///
/// The broader question above counts a position where every opposing reply
/// happens to uncover a winning card for the holder. That really is an
/// undeniable close, and a searching agent would find it — but a 1-ply rail
/// that reads the post-action state cannot, because at that moment the card is
/// still covered. Splitting the two is what turns "Rail C recall is 50%" into
/// a statement about the design rather than a bug report.
fn creates_a_close_already_on_the_table(state: &GameState, action: Action, holder: Player) -> bool {
    let worlds = futures(state, action);
    if worlds.is_empty() {
        return false;
    }
    worlds.into_iter().all(|world| {
        if world.is_over() || world.current_player() == holder || world.phase() != Phase::Turn {
            return false;
        }
        let on_the_table = world.accessible_slots();
        let replies = engine::legal_actions(&world);
        !replies.is_empty()
            && replies.into_iter().all(|r| {
                futures(&world, r)
                    .iter()
                    .all(|s| closes_from_a_slot_in(s, holder, on_the_table, PENDING_PLIES))
            })
    })
}

/// Audit one decision. Returns whether a Rail-B-resolving candidate existed
/// and whether it was taken, for the loss post-mortem.
#[allow(clippy::too_many_arguments)]
fn audit_decision(
    state: &GameState,
    me: Player,
    legal: &[Action],
    chosen: Action,
    out: &mut Audit,
    explain: Option<&str>,
) -> (bool, bool) {
    out.decisions += 1;
    if legal.len() < 2 || state.phase() != Phase::Turn || !hot(state) {
        return (false, false);
    }
    out.hot_decisions += 1;
    let opp = me.other();

    // --- Rail A ---------------------------------------------------------
    let wins: Vec<bool> = legal
        .iter()
        .map(|&a| {
            let f = futures(state, a);
            !f.is_empty() && f.iter().all(|s| is_win(s, me))
        })
        .collect();
    if wins.iter().any(|&w| w) {
        out.a_available += 1;
        let i = legal.iter().position(|&a| a == chosen).unwrap_or(0);
        if wins[i] {
            out.a_taken += 1;
        } else if let Some(ctx) = explain {
            let winners: Vec<&Action> = legal
                .iter()
                .zip(&wins)
                .filter(|(_, &w)| w)
                .map(|(a, _)| a)
                .collect();
            println!(
                "  MISS A  {ctx} turn {} age {}: chose {chosen:?}, winning: {winners:?}",
                state.turn(),
                state.age()
            );
        }
        // Rail A settles the decision; B and C are not asked.
        return (false, false);
    }

    // --- Rail B ---------------------------------------------------------
    let concedes: Vec<bool> = legal
        .iter()
        .map(|&a| hands_over_a_close(state, a, opp))
        .collect();

    // Precision: what the agent's own rail said about each candidate, against
    // what the engine says. Recall below 100% costs a missed bonus; precision
    // below 100% would misvalue a move by five hundred points, so it is the
    // half that has to be exact.
    //
    // Which rail fired is read off the position rather than off which player
    // it named: `rail_owner` returns the *mover* when the mover can close
    // (Rails B and C'), and the *waiter* when the waiter holds a close no
    // single reply removes (Rail C). Either player can be either, so an
    // extra-turn candidate really can leave the opponent holding an
    // undeniable close.
    for (&action, &wins_it) in legal.iter().zip(&wins) {
        if wins_it {
            continue;
        }
        for world in futures(state, action) {
            let Some(owner) = rail_owner(&world, state.age(), RailModel::On) else {
                continue;
            };
            if owner == world.current_player() {
                out.b_fired += 1;
                if can_close(&world, owner) {
                    out.b_fired_correct += 1;
                } else if let Some(ctx) = explain {
                    println!(
                        "  FALSE B {ctx} turn {}: after {action:?} the mover was said \
                         to have a close and does not",
                        state.turn()
                    );
                }
            } else {
                out.c_fired += 1;
                let really = !world.is_over()
                    && world.phase() == Phase::Turn
                    && can_close_after_any_reply(&world, owner);
                if really {
                    out.c_fired_correct += 1;
                } else if let Some(ctx) = explain {
                    println!(
                        "  FALSE C {ctx} turn {}: after {action:?} {owner:?} was said to \
                         hold an undeniable close and a reply takes it away",
                        state.turn()
                    );
                    let sources = duels_strategy::closing_sources_with(
                        &world,
                        owner,
                        world.age() == state.age(),
                    );
                    println!(
                        "          world: age {} phase {:?} mover {:?}; {owner:?} need {} \
                         slots {:#x} wonders {:#x} science {:#x} coins {}",
                        world.age(),
                        world.phase(),
                        world.current_player(),
                        sources.need,
                        sources.military_slots,
                        sources.military_wonders,
                        sources.science_slots,
                        world.player(owner).coins()
                    );
                    for r in engine::legal_actions(&world) {
                        if futures(&world, r)
                            .iter()
                            .all(|s| can_close_eventually(s, owner, PENDING_PLIES))
                        {
                            continue;
                        }
                        println!("          refuted by {r:?}");
                    }
                }
            }
        }
    }

    let threatened = concedes.iter().any(|&c| c);
    let avoidable = concedes.iter().any(|&c| !c);
    if threatened && avoidable {
        out.b_available += 1;
        let i = legal.iter().position(|&a| a == chosen).unwrap_or(0);
        if !concedes[i] {
            out.b_taken += 1;
        } else if let Some(ctx) = explain {
            let safe: Vec<&Action> = legal
                .iter()
                .zip(&concedes)
                .filter(|(_, &c)| !c)
                .map(|(a, _)| a)
                .collect();
            println!(
                "  MISS B  {ctx} turn {} age {} pawn {}: chose {chosen:?}, safe: {safe:?}",
                state.turn(),
                state.age(),
                state.conflict()
            );
        }
        return (
            true,
            !concedes[legal.iter().position(|&a| a == chosen).unwrap_or(0)],
        );
    }
    if threatened {
        // Every candidate concedes: nothing to audit, and nothing to blame.
        return (false, false);
    }

    // --- Rail C ---------------------------------------------------------
    let decisive: Vec<bool> = legal
        .iter()
        .zip(&concedes)
        .map(|(&a, &c)| !c && creates_an_undeniable_close(state, a, me))
        .collect();
    if decisive.iter().any(|&d| d) {
        out.c_available += 1;
        let i = legal.iter().position(|&a| a == chosen).unwrap_or(0);
        if decisive[i] {
            out.c_taken += 1;
        }
        if !decisive[i] {
            if let Some(ctx) = explain {
                let good: Vec<&Action> = legal
                    .iter()
                    .zip(&decisive)
                    .filter(|(_, &d)| d)
                    .map(|(a, _)| a)
                    .collect();
                println!(
                    "  MISS C  {ctx} turn {} age {} pawn {}: chose {chosen:?}, decisive: {good:?}",
                    state.turn(),
                    state.age(),
                    state.conflict()
                );
                // Why did the rail not see it? Report the reply its own check
                // was worried about, in each decisive candidate's post-state.
                for (&a, &d) in legal.iter().zip(&decisive) {
                    if !d {
                        continue;
                    }
                    for world in futures(state, a) {
                        let sources = duels_strategy::closing_sources(&world, me);
                        println!(
                            "          after {a:?}: {} closing source(s), rail says {:?}, \
                             worried about {:?}",
                            sources.count(),
                            rail_owner(&world, state.age(), RailModel::On),
                            first_resolving_reply(&world, me, &sources)
                        );
                    }
                }
            }
        }
        let on_table: Vec<bool> = legal
            .iter()
            .zip(&decisive)
            .map(|(&a, &d)| d && creates_a_close_already_on_the_table(state, a, me))
            .collect();
        if on_table.iter().any(|&d| d) {
            out.c_now_available += 1;
            if on_table[i] {
                out.c_now_taken += 1;
            }
        }
    }
    (false, false)
}

/// Play one game, auditing every decision the audited seat makes.
#[allow(clippy::too_many_arguments)]
fn play_and_audit(
    audited: &str,
    opponent: &str,
    audited_seat: Player,
    audited_seed: u64,
    opponent_seed: u64,
    setup_seed: u64,
    budget: Budget,
    out: &mut Audit,
    explain: bool,
) -> Result<(), String> {
    let mut mine = make_agent_from_spec(audited, audited_seed)?;
    let mut theirs = make_agent_from_spec(opponent, opponent_seed)?;
    let mut state = engine::new_game(setup_seed);
    let mut rng = StdRng::seed_from_u64(setup_seed ^ ENGINE_RNG_SALT);
    let mut last_b: Option<(bool, bool)> = None;
    let mut moves = 0u32;

    while !state.is_over() {
        let legal = engine::legal_actions(&state);
        if legal.is_empty() {
            break;
        }
        let obs = state.observation();
        let me = state.current_player();
        let action = if me == audited_seat {
            mine.choose(&obs, &legal, budget)
        } else {
            theirs.choose(&obs, &legal, budget)
        };
        if me == audited_seat {
            let ctx = format!("seed {setup_seed} seat {audited_seat:?}");
            let verdict = audit_decision(
                &state,
                me,
                &legal,
                action,
                out,
                explain.then_some(ctx.as_str()),
            );
            if verdict.0 {
                last_b = Some(verdict);
            }
        }
        engine::apply(&mut state, action, &mut rng).map_err(|e| e.to_string())?;
        moves += 1;
        if moves > 400 {
            return Err("game did not terminate".to_string());
        }
    }

    if let Some(GameResult::Win { winner, kind }) = state.result() {
        let raced = matches!(
            kind,
            VictoryKind::MilitarySupremacy | VictoryKind::ScientificSupremacy
        );
        if raced && winner != audited_seat {
            out.supremacy_losses += 1;
            // The last decision at which a resolving candidate was on the
            // table: if there was one and it was not taken, this loss was
            // avoidable and the rail failed.
            if matches!(last_b, Some((true, false))) {
                out.avoidable_supremacy_losses += 1;
            }
        }
    }
    Ok(())
}

/// [`can_close_eventually`], restricted to closing actions whose slot was
/// already in `on_the_table`.
fn closes_from_a_slot_in(state: &GameState, p: Player, on_the_table: u32, depth: u8) -> bool {
    if state.is_over() {
        return false;
    }
    if state.current_player() == p && state.phase() == Phase::Turn && state.pending().is_none() {
        return engine::legal_actions(state).into_iter().any(|a| {
            let already = match a {
                Action::Build { slot } | Action::BuildWonder { slot, .. } => {
                    on_the_table & (1u32 << slot) != 0
                }
                _ => false,
            };
            already && futures(state, a).iter().all(|t| is_race_win(t, p))
        });
    }
    if depth == 0 {
        return false;
    }
    let actions = engine::legal_actions(state);
    if actions.is_empty() {
        return false;
    }
    let mine = state.current_player() == p;
    let mut any = false;
    for a in actions {
        let ok = futures(state, a)
            .iter()
            .all(|s| closes_from_a_slot_in(s, p, on_the_table, depth - 1));
        if mine {
            any |= ok;
        } else if !ok {
            return false;
        }
    }
    !mine || any
}

fn line(label: &str, taken: u64, available: u64) {
    let pct = if available == 0 {
        100.0
    } else {
        100.0 * taken as f64 / available as f64
    };
    let verdict = if taken == available { "OK  " } else { "MISS" };
    println!("  {verdict} {label:<44} {taken:>5} / {available:<5}  ({pct:.1}%)");
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let positional: Vec<&String> = args.iter().take_while(|a| !a.starts_with("--")).collect();
    let flag = |name: &str| -> Option<String> {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };

    let audited = positional
        .first()
        .map(|s| s.as_str())
        .unwrap_or("phased")
        .to_string();
    let opponent = positional
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("phased")
        .to_string();
    let games: u64 = positional
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);
    let seed: u64 = flag("--seed").and_then(|s| s.parse().ok()).unwrap_or(1);
    let explain = args.iter().any(|a| a == "--explain");
    let budget = match parse_budget(&flag("--budget").unwrap_or_else(|| "nodes:2000".to_string())) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };

    let pairs = (games / 2).max(1);
    println!(
        "rail audit: {audited} vs {opponent}, {} games from seed {seed}, budget {budget:?}\n",
        pairs * 2
    );

    let mut total = Audit::default();
    for i in 0..pairs {
        let s = seed + i;
        for seat in [Player::One, Player::Two] {
            let mut one = Audit::default();
            let (a_seed, b_seed) = (s ^ AGENT_A_SALT, s ^ AGENT_B_SALT);
            let (audited_seed, opponent_seed) = match seat {
                Player::One => (a_seed, b_seed),
                Player::Two => (b_seed, a_seed),
            };
            if let Err(e) = play_and_audit(
                &audited,
                &opponent,
                seat,
                audited_seed,
                opponent_seed,
                s,
                budget,
                &mut one,
                explain,
            ) {
                eprintln!("seed {s} seat {seat:?}: {e}");
                std::process::exit(1);
            }
            total.merge(&one);
        }
    }

    println!(
        "  {} decisions audited, {} of them close enough to a race to matter\n",
        total.decisions, total.hot_decisions
    );
    line(
        "Rail A  an available win was taken",
        total.a_taken,
        total.a_available,
    );
    line(
        "Rail B  an available block was taken",
        total.b_taken,
        total.b_available,
    );
    line(
        "Rail C  an available undeniable close was taken",
        total.c_taken,
        total.c_available,
    );
    line(
        "Rail C  ...of those already on the table",
        total.c_now_taken,
        total.c_now_available,
    );
    println!();
    println!("  the agent's own rails, against the same ground truth:");
    line(
        "B/C'   a position handed to a closer really was",
        total.b_fired_correct,
        total.b_fired,
    );
    line(
        "C      a position called decisive really was",
        total.c_fired_correct,
        total.c_fired,
    );
    println!();
    let clean = total.supremacy_losses - total.avoidable_supremacy_losses;
    let verdict = if total.avoidable_supremacy_losses == 0 {
        "OK  "
    } else {
        "BUG "
    };
    println!(
        "  {verdict} {:<44} {:>5} / {:<5}",
        "supremacy losses that were NOT blockable", clean, total.supremacy_losses
    );
    if total.avoidable_supremacy_losses > 0 {
        println!(
            "         {} supremacy loss(es) had a resolving candidate on the table \
             at the last decision",
            total.avoidable_supremacy_losses
        );
    }
}
