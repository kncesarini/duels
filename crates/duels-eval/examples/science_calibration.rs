//! What does a science position *actually* win, and what does this crate say
//! it wins?
//!
//! [`duels_eval::win_probability`] maps [`duels_eval::evaluate`]'s
//! victory-point score onto a win probability through the age-calibrated
//! logistic `examples/calibrate.rs` fits. That fit is an *aggregate* one: it
//! asks whether the evaluation is well calibrated over all positions at once,
//! and a term can be badly mispriced in one corner of the state space while the
//! aggregate stays honest, because the corner is a small share of the corpus.
//!
//! This example asks the same question restricted to one corner: **the mover's
//! distinct scientific symbol count**. It replays real self-play games exactly
//! the way `calibrate.rs` does — a verbatim copy of `PhasedAgent::choose`, the
//! same `seed ^ 0xF00D` engine stream, both seat orders — and, at every
//! decision, records for **each** player
//!
//! * how many distinct symbols that player holds
//!   ([`duels_core::state::PublicPlayer::distinct_science`]),
//! * what [`duels_eval::win_probability`] says their chances are,
//! * and whether they went on to actually win.
//!
//! Bucketing those by `(age, symbols)` and comparing the mean prediction with
//! the realised win rate is a direct, empirical test of whether the science
//! terms are priced right, as against an argument about whether a hand-built
//! position "feels" strong enough.
//!
//! Both players are recorded from one [`duels_eval::Root`], which is exact
//! rather than an approximation: [`duels_eval::evaluate`] is antisymmetric, so
//! the non-mover's score is the mover's negated and the two probabilities sum
//! to one.
//!
//! # Three tables and a fit, because they answer different questions
//!
//! * **every position** weights a symbol count by how many decisions a player
//!   spends holding it, which is the distribution a leaf value actually sees.
//! * **first reach** keeps one sample per (game, seat, age, count) — the
//!   position at which that player first got to that many symbols in that age
//!   — which is much closer to the question "a position with four symbols in
//!   Age II is worth what?" and does not let one long game dominate a bucket.
//! * **the fitted correction** ([`rung_correction`]) turns the gap between
//!   predicted and actual into **victory points per rung**, which is the form a
//!   [`duels_eval::ScienceWeights::ladder`] entry is actually written in. A
//!   bucket gap says "this is wrong"; `δ_k` says by how much and in what units.
//!
//! # `--factors`: is the error explained by something the evaluation cannot see?
//!
//! Round nine added a second question on top, because a bucket gap alone does
//! not name a term. Every `(age, symbols)` cell is split by a *second* factor —
//! cards left in the structure, how many symbols are still assemblable, how
//! many missing symbols are face up right now, the unbuilt-play-again-wonder
//! differential, and who took the first decision of the age — and
//! [`rung_correction_where`] re-runs the `δ_k` fit inside each bin.
//!
//! The reading is the whole point. **A gap that is the same in every bin of a
//! factor is a factor the evaluation already prices**, however strongly that
//! factor predicts the outcome. **A gap that moves across the bins is a term
//! the evaluation is missing**, and that is the only kind of finding that names
//! one. Round nine's own use of it is in the crate docs: the `reachable` split
//! is what said round eight's named "age-scale the ladder rung" follow-up was
//! aimed at the wrong population, and the `extra-turn diff` split is what
//! pointed at the wonder term.
//!
//! # The reading configuration is separate from the playing ones
//!
//! `--read` is the [`Config`] every position is *scored* with — the evaluation
//! under judgement — and it is fixed for the whole run. The two positional
//! arguments are the [`Config`]s the two **seats play with**, and they only
//! decide which positions get visited. Keeping them separate is the whole
//! design: letting the reading config follow the seat compares each evaluation
//! against its own games, which is a different and much less useful question.
//! All three take the `label:key=value,...` form `examples/head_to_head.rs`
//! parses, `base=vN` included.
//!
//! # Populating the thin buckets honestly
//!
//! The default agent reaches four distinct symbols rarely — round seven traded
//! the science lottery for civilian points and says so — so plain self-play
//! leaves exactly the interesting buckets at tens of samples. Rather than
//! reweighting anything, run the seats **science-tilted**, with the ladder
//! weight well above its default. Tilt *both* seats: the corpus then stays
//! symmetric, so an unconditional win rate is still 50% and a bucket's realised
//! rate is a rate against an equal opponent, where tilting one seat conflates
//! "this position is strong" with "this seat is weaker".
//!
//! It is off-policy for the evaluation being read, and that is the honest
//! trade, stated rather than hidden: hundreds to thousands of samples in the
//! buckets that matter, against tens.
//!
//! ```text
//! # on-policy, and thin at the top
//! cargo run --release -p duels-eval --example science_calibration -- --games 400
//!
//! # one fixed science-heavy corpus, read by two generations in turn --
//! # the exact before/after in the round-eight crate docs
//! cargo run --release -p duels-eval --example science_calibration -- \
//!     --games 1000 --read "v7:base=v7" "a:base=v7,sci=3.0" "b:base=v7,sci=3.0"
//! cargo run --release -p duels-eval --example science_calibration -- \
//!     --games 1000 --read "r8:"       "a:base=v7,sci=3.0" "b:base=v7,sci=3.0"
//!
//! # round nine's factor tables, off the same fixed corpus
//! cargo run --release -p duels-eval --example science_calibration -- \
//!     --games 2000 --factors --read "v8:base=v8" \
//!     "a:base=v7,sci=3.0" "b:base=v7,sci=3.0"
//! ```

use duels_core::scoring::GameResult;
use duels_core::{engine, Action, Observation, Player};
use duels_eval::{evaluate, expected_value, win_probability, Config, Root};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// `PhasedAgent`'s tie window, copied because the tie set feeds the RNG draw
/// and so decides which move comes out.
const TIE_EPSILON: f64 = 1e-6;
/// `duels_arena::match_runner`'s per-role agent salts, verbatim.
const AGENT_A_SALT: u64 = 0xA011_7A9E_5B21_0001;
const AGENT_B_SALT: u64 = 0xB022_8C3F_6D42_0002;

/// A verbatim copy of `PhasedAgent::choose`, RNG usage included.
struct Driver {
    rng: StdRng,
    config: Config,
}

impl Driver {
    fn new(seed: u64, config: Config) -> Driver {
        Driver {
            rng: StdRng::seed_from_u64(seed),
            config,
        }
    }

    fn choose(&mut self, obs: &Observation, legal: &[Action]) -> Action {
        if legal.len() == 1 {
            return legal[0];
        }
        let me = obs.current_player;
        let base_state = obs.sample_state(&mut self.rng);
        let root = Root::new(&base_state, me, self.config);

        let mut scored: Vec<(Action, f64)> = Vec::with_capacity(legal.len());
        for &action in legal {
            scored.push((action, expected_value(&base_state, action, me, &root)));
        }
        let Some(best_score) = scored.iter().map(|&(_, s)| s).fold(None, |m, s| match m {
            Some(b) if b >= s => Some(b),
            _ => Some(s),
        }) else {
            return legal[self.rng.gen_range(0..legal.len())];
        };
        let best: Vec<Action> = scored
            .iter()
            .filter(|&&(_, s)| (best_score - s).abs() <= TIE_EPSILON)
            .map(|&(a, _)| a)
            .collect();
        best[self.rng.gen_range(0..best.len())]
    }
}

/// How many of the distinct symbols `p` does **not** hold are printed on a
/// card that is face up in the structure at this instant.
///
/// The distinction the project owner drew in Age II — "all four of the age's
/// distinct symbols still achievable, face-up and known, or face-down but
/// plausibly reachable" — is exactly this against
/// [`duels_eval::terms::supremacy_reachable`], which counts a symbol reachable
/// whenever *some* card printing it is not provably gone, face down or not.
fn faceup_missing_symbols(state: &duels_core::GameState, p: Player) -> u8 {
    let held = state.player(p).science();
    let mut n = 0u8;
    for sym in duels_strategy::masks::ALL_SCIENCE {
        if held[sym.index()] > 0 {
            continue;
        }
        let mut mask = state.occupied_slots();
        let mut found = false;
        while mask != 0 {
            let slot = mask.trailing_zeros() as u8;
            mask &= mask - 1;
            if let Some(card) = state.face_up_card(slot) {
                if card.def().science == Some(sym) {
                    found = true;
                    break;
                }
            }
        }
        if found {
            n += 1;
        }
    }
    n
}

/// Play-again wonders `p` has drafted and not yet built — the extra turns they
/// still have in hand, which is the half of the parity question that is still
/// to come.
fn extra_turn_wonders(state: &duels_core::GameState, p: Player) -> u8 {
    let ps = state.player(p);
    ps.wonders()
        .filter(|&w| !ps.has_built_wonder(w) && w.def().play_again)
        .count() as u8
}

/// One observation for the comparison.
#[derive(Clone, Copy)]
struct Sample {
    age: u8,
    symbols: u8,
    /// The *other* player's distinct symbol count in the same position.
    opp_symbols: u8,
    /// `evaluate(state, this player, root)`, on the victory-point scale.
    value: f64,
    predicted: f64,
    won: bool,
    /// Whether this player was the one to move in the position.
    mover: bool,
    /// Whether this player's seat was the overridden ("A") configuration.
    side_a: bool,
    /// First position in this game at which this seat held this many symbols
    /// in this age.
    first: bool,
    // --- round nine: the factors the project owner named, one field each ----
    /// Cards still in the current age's structure, taken *and* untaken slots
    /// excluded — `GameState::occupied_slots().count_ones()`. Twenty at the
    /// start of an age, zero at its end, so it is "which turn of the age is
    /// this" in the units the board actually offers.
    cards_left: u8,
    /// How many distinct symbols this player could still *end the game*
    /// holding, from [`duels_eval::terms::supremacy_reachable`] — six wins, so
    /// anything below six means the supremacy race is dead for them.
    reach: u8,
    /// Of the symbols this player does not hold, how many are printed on a
    /// card that is **face up in the structure right now**. The project
    /// owner's "face-up and known" against "face-down but plausibly
    /// reachable": `reach` counts the second, this counts the first.
    faceup_missing: u8,
    /// Unbuilt play-again wonders this player holds minus the opponent's — the
    /// parity lever, since an extra turn is the only thing that re-assigns the
    /// remaining slot sequence.
    extra_turn_diff: i8,
    /// Whether this player took the first decision of the current age.
    age_starter: bool,
}

/// Play one game and emit every `(player, position)` sample from it.
fn play_one(
    seat_one: (Config, u64),
    seat_two: (Config, u64),
    read: Config,
    setup_seed: u64,
    a_seat: Player,
    out: &mut Vec<Sample>,
) {
    let config = read;
    let mut driver = [
        Driver::new(seat_one.1, seat_one.0),
        Driver::new(seat_two.1, seat_two.0),
    ];
    let mut state = engine::new_game(setup_seed);
    let mut rng = StdRng::seed_from_u64(setup_seed ^ 0xF00D);
    // (age, symbols) pairs already seen, per seat.
    let mut seen: [Vec<(u8, u8)>; 2] = [Vec::new(), Vec::new()];
    let mut trace: Vec<(Player, Sample)> = Vec::new();
    // Who took the first decision of each age, indexed by age minus one.
    // Recorded rather than derived: the age-boundary first-player choice runs
    // through `Phase::ChooseFirstPlayer` and the military track, and the only
    // reliable read of "who actually started" is to watch it happen.
    let mut age_starter: [Option<Player>; 3] = [None; 3];

    while !state.is_over() {
        let legal = engine::legal_actions(&state);
        if legal.is_empty() {
            break;
        }
        let me = state.current_player();
        if state.phase() == duels_core::state::Phase::Turn {
            let slot = &mut age_starter[usize::from(state.age().clamp(1, 3)) - 1];
            if slot.is_none() {
                *slot = Some(me);
            }
        }
        // One `Root` serves both players: `evaluate` is antisymmetric, so the
        // non-mover's probability is exactly one minus the mover's.
        //
        // The configuration a position is *read* with is `--read`, and is fixed
        // for the whole run — it is the evaluation under judgement — while the
        // two seats' configurations only decide which positions get visited.
        // Letting the reading config follow the seat would compare each seat's
        // evaluation against its own games, which is a different question.
        let root = Root::new(&state, me, config);
        let age = state.age();
        for p in [Player::One, Player::Two] {
            let symbols = state.player(p).distinct_science();
            let opp_symbols = state.player(p.other()).distinct_science();
            let key = (age, symbols);
            let first = !seen[p.index()].contains(&key);
            if first {
                seen[p.index()].push(key);
            }
            trace.push((
                p,
                Sample {
                    age,
                    symbols,
                    opp_symbols,
                    value: evaluate(&state, p, &root),
                    predicted: win_probability(&state, p, &root),
                    won: false,
                    mover: p == me,
                    side_a: p == a_seat,
                    first,
                    cards_left: state.occupied_slots().count_ones() as u8,
                    reach: duels_eval::terms::supremacy_reachable(&state, p),
                    faceup_missing: faceup_missing_symbols(&state, p),
                    extra_turn_diff: extra_turn_wonders(&state, p) as i8
                        - extra_turn_wonders(&state, p.other()) as i8,
                    age_starter: age_starter[usize::from(age.clamp(1, 3)) - 1] == Some(p),
                },
            ));
        }

        let obs: Observation = state.observation();
        let action = driver[me.index()].choose(&obs, &legal);
        engine::apply_quiet(&mut state, action, &mut rng).expect("the driver plays legally");
    }

    // Drawn games are dropped, exactly as `calibrate.rs` drops them: there are
    // very few and a win rate has nowhere to put them.
    if let Some(GameResult::Win { winner, .. }) = state.result() {
        for (p, mut s) in trace {
            s.won = p == winner;
            out.push(s);
        }
    }
}

/// One bucket of the comparison table.
#[derive(Default, Clone, Copy)]
struct Bucket {
    n: u32,
    predicted: f64,
    won: u32,
}

impl Bucket {
    fn add(&mut self, s: &Sample) {
        self.n += 1;
        self.predicted += s.predicted;
        self.won += u32::from(s.won);
    }
}

fn table(title: &str, samples: &[Sample], keep: impl Fn(&Sample) -> bool) {
    println!("\n{title}");
    println!(
        "  {:>3}  {:>4}  {:>9}  {:>10}  {:>9}  {:>8}",
        "age", "sym", "n", "predicted", "actual", "gap"
    );
    for age in 1..=3u8 {
        for symbols in 0..=6u8 {
            let mut b = Bucket::default();
            for s in samples
                .iter()
                .filter(|s| s.age == age && s.symbols == symbols && keep(s))
            {
                b.add(s);
            }
            if b.n == 0 {
                continue;
            }
            let predicted = b.predicted / f64::from(b.n);
            let actual = f64::from(b.won) / f64::from(b.n);
            // A binomial 95% half-width on the realised rate, so a bucket that
            // is only thin can be told from one that actually disagrees.
            let se = (actual * (1.0 - actual) / f64::from(b.n)).sqrt();
            println!(
                "  {age:>3}  {symbols:>4}  {:>9}  {predicted:>10.3}  {actual:>9.3}  {:>+8.3}  ±{:.3}",
                b.n,
                actual - predicted,
                1.96 * se
            );
        }
    }
}

/// Split one `(age, symbols)` cell by a second factor, and report what the
/// evaluation predicts against what actually happens in each bin.
///
/// This is the round-nine extension, and the question it answers is narrower
/// and sharper than the plain bucket table's. A gap in `table` says the
/// evaluation is wrong at that symbol count. A gap that *differs across the
/// bins here* says the evaluation is wrong **because it cannot see this
/// factor** — which is the only kind of finding that names a new term. A
/// factor the evaluation already prices correctly shows the same gap in every
/// bin, however strongly it predicts the outcome.
fn factor_table(
    title: &str,
    samples: &[Sample],
    name: &str,
    bin: impl Fn(&Sample) -> Option<(usize, &'static str)>,
    bins: usize,
) {
    println!("\n{title}  (factor: {name})");
    println!(
        "  {:>3}  {:>4}  {:>16}  {:>7}  {:>10}  {:>9}  {:>8}",
        "age", "sym", name, "n", "predicted", "actual", "gap"
    );
    for age in 1..=3u8 {
        for symbols in 2..=5u8 {
            let mut rows: Vec<(String, Bucket)> = Vec::new();
            for _ in 0..bins {
                rows.push((String::new(), Bucket::default()));
            }
            for s in samples
                .iter()
                .filter(|s| s.age == age && s.symbols == symbols && s.first && s.mover)
            {
                if let Some((i, label)) = bin(s) {
                    if i < bins {
                        rows[i].0 = label.to_string();
                        rows[i].1.add(s);
                    }
                }
            }
            if rows.iter().map(|r| r.1.n).sum::<u32>() < 30 {
                continue;
            }
            for (label, b) in &rows {
                if b.n == 0 {
                    continue;
                }
                let predicted = b.predicted / f64::from(b.n);
                let actual = f64::from(b.won) / f64::from(b.n);
                let se = (actual * (1.0 - actual) / f64::from(b.n)).sqrt();
                println!(
                    "  {age:>3}  {symbols:>4}  {label:>16}  {:>7}  {predicted:>10.3}  \
                     {actual:>9.3}  {:>+8.3}  ±{:.3}",
                    b.n,
                    actual - predicted,
                    1.96 * se
                );
            }
        }
    }
}

/// Every round-nine factor table, off the same corpus.
fn factor_tables(samples: &[Sample]) {
    println!(
        "\n=== round nine: is the evaluation's science error explained by a factor \
         it cannot see? ===\n(first position at that count, the player to move. A gap that \
         moves across the bins of one\n factor is a term the evaluation is missing; a gap that \
         is flat across them is not.)"
    );

    factor_table(
        "how far into the age the count was reached",
        samples,
        "cards left",
        |s| {
            Some(match s.cards_left {
                0..=6 => (0, "0-6 (late)"),
                7..=13 => (1, "7-13 (mid)"),
                _ => (2, "14-20 (early)"),
            })
        },
        3,
    );

    factor_table(
        "whether six distinct symbols are still assemblable",
        samples,
        "reachable",
        |s| {
            Some(match s.reach {
                6..=7 => (0, "6+ (race live)"),
                5 => (1, "5 (one short)"),
                _ => (2, "<=4 (dead)"),
            })
        },
        3,
    );

    factor_table(
        "missing symbols that are face up on the board right now",
        samples,
        "face-up missing",
        |s| {
            Some(match s.faceup_missing {
                0 => (0, "0"),
                1 => (1, "1"),
                _ => (2, "2+"),
            })
        },
        3,
    );

    factor_table(
        "unbuilt play-again wonders, this player minus the opponent",
        samples,
        "extra-turn diff",
        |s| {
            Some(match s.extra_turn_diff {
                i8::MIN..=-1 => (0, "behind"),
                0 => (1, "level"),
                _ => (2, "ahead"),
            })
        },
        3,
    );

    factor_table(
        "who took the first decision of this age",
        samples,
        "age starter",
        |s| {
            Some(if s.age_starter {
                (0, "this player")
            } else {
                (1, "the opponent")
            })
        },
        2,
    );
}

/// The residual value, in victory points, that the evaluation is *missing* at
/// each distinct-symbol count.
///
/// The bucket tables above say a science-heavy position wins less often than
/// [`win_probability`] claims, but they cannot say how much of that is the
/// science terms and how much is everything else about the positions a science
/// player reaches. This can: hold the evaluation and the calibrated temperature
/// fixed, and fit one additive correction `δ_k` per symbol count by maximum
/// likelihood, on
///
/// ```text
/// P(this player wins) = σ( ( evaluate + δ_{sym(me)} − δ_{sym(opp)} ) / T(age) )
/// ```
///
/// with `δ_0 ≡ 0` as the reference. `δ_k` is then, in victory points, exactly
/// what would have to be *added* to a player holding `k` symbols for the
/// evaluation to predict the outcomes actually observed — so a negative `δ_k`
/// is an over-priced rung and a positive one an under-priced rung. Written as a
/// difference of the two players' counts, it is antisymmetric like everything
/// else here.
///
/// Fitted on the **mover's** sample only: the two players' samples from one
/// position are exact complements, so counting both would halve every standard
/// error while adding no information.
fn rung_correction(samples: &[Sample], age: u8) -> ([f64; 6], [f64; 6], [u32; 6]) {
    rung_correction_where(samples, age, |_| true)
}

/// [`rung_correction`] restricted to a subset of the corpus.
///
/// Added in round nine so the same fit can be run *inside* one bin of a second
/// factor — the actionable form of a factor table, since it reports the
/// missing victory points per rung rather than a probability gap, and a
/// [`duels_eval::ScienceWeights::ladder`] entry is written in victory points.
fn rung_correction_where(
    samples: &[Sample],
    age: u8,
    keep: impl Fn(&Sample) -> bool,
) -> ([f64; 6], [f64; 6], [u32; 6]) {
    let t = duels_eval::win_probability_temperature(age);
    // Rails are magnitudes rather than judgements (`±imminent` = 500), so they
    // would swamp a likelihood; drop them, as `calibrate.rs` drops terminals.
    let rows: Vec<&Sample> = samples
        .iter()
        .filter(|s| s.mover && s.age == age && s.value.abs() < 100.0 && keep(s))
        .collect();
    let mut delta = [0.0f64; 6];
    let mut n = [0u32; 6];
    for s in &rows {
        n[usize::from(s.symbols).min(5)] += 1;
    }
    // Coordinate-wise Newton. The likelihood is concave in `delta`, and five
    // free coordinates over a few hundred thousand rows converges in a handful
    // of sweeps.
    let mut information = [0.0f64; 6];
    for _ in 0..60 {
        for k in 1..6usize {
            let (mut grad, mut hess) = (0.0f64, 0.0f64);
            for s in &rows {
                let a = usize::from(s.symbols).min(5);
                let b = usize::from(s.opp_symbols).min(5);
                let d = f64::from(u8::from(a == k)) - f64::from(u8::from(b == k));
                if d == 0.0 {
                    continue;
                }
                let z = (s.value + delta[a] - delta[b]) / t;
                let p = 1.0 / (1.0 + (-z).exp());
                grad += (f64::from(u8::from(s.won)) - p) * d / t;
                hess += p * (1.0 - p) * d * d / (t * t);
            }
            if hess > 1e-12 {
                delta[k] += grad / hess;
                information[k] = hess;
            }
        }
    }
    let mut se = [f64::INFINITY; 6];
    for k in 1..6usize {
        if information[k] > 0.0 {
            se[k] = 1.0 / information[k].sqrt();
        }
    }
    (delta, se, n)
}

fn correction_table(samples: &[Sample]) {
    println!(
        "\nfitted additive correction per symbol count, in victory points\n  \
         (negative = the evaluation over-credits that rung; delta_0 is the reference)"
    );
    println!(
        "  {:>3}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}",
        "age", "d1", "d2", "d3", "d4", "d5"
    );
    for age in 1..=3u8 {
        let (delta, se, n) = rung_correction(samples, age);
        println!(
            "  {age:>3}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}",
            format!("{:+.1}", delta[1]),
            format!("{:+.1}", delta[2]),
            format!("{:+.1}", delta[3]),
            format!("{:+.1}", delta[4]),
            format!("{:+.1}", delta[5]),
        );
        println!(
            "       {:>8}  {:>8}  {:>8}  {:>8}  {:>8}   (95% half-width)",
            format!("±{:.1}", 1.96 * se[1]),
            format!("±{:.1}", 1.96 * se[2]),
            format!("±{:.1}", 1.96 * se[3]),
            format!("±{:.1}", 1.96 * se[4]),
            format!("±{:.1}", 1.96 * se[5]),
        );
        println!(
            "       {:>8}  {:>8}  {:>8}  {:>8}  {:>8}   (mover positions)",
            n[1], n[2], n[3], n[4], n[5]
        );
    }
}

/// One bin of a second factor: its label, and the predicate that selects it.
type FactorBin = (&'static str, fn(&Sample) -> bool);

/// [`correction_table`] run inside each bin of a second factor, so the missing
/// victory points per rung can be read against that factor directly.
fn correction_by_factor(samples: &[Sample], name: &str, bins: &[FactorBin]) {
    println!("\nfitted additive correction per symbol count, split by {name}, in victory points");
    println!(
        "  {:>3}  {:>16}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}",
        "age", name, "d1", "d2", "d3", "d4", "d5"
    );
    for age in 1..=3u8 {
        for (label, keep) in bins {
            let (delta, _se, n) = rung_correction_where(samples, age, keep);
            if n.iter().sum::<u32>() < 200 {
                continue;
            }
            println!(
                "  {age:>3}  {label:>16}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}",
                format!("{:+.1}", delta[1]),
                format!("{:+.1}", delta[2]),
                format!("{:+.1}", delta[3]),
                format!("{:+.1}", delta[4]),
                format!("{:+.1}", delta[5]),
            );
            println!(
                "       {:>16}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}   (mover positions)",
                "", n[1], n[2], n[3], n[4], n[5]
            );
        }
    }
}

/// Apply one `key=value` override — deliberately the same table as
/// `examples/head_to_head.rs`, so a configuration screened on one instrument
/// can be inspected on the other by the same string.
fn apply(cfg: &mut Config, key: &str, raw: &str) -> Result<(), String> {
    if key == "base" {
        *cfg = match raw {
            "v1" => Config::v1(),
            "v2" => Config::v2(),
            "v3" => Config::v3(),
            "v4" => Config::v4(),
            "v5" => Config::v5(),
            "v6" => Config::v6(),
            "v7" => Config::v7(),
            "v8" => Config::v8(),
            "v9" => Config::v9(),
            "default" => Config::default(),
            other => return Err(format!("unknown base \"{other}\"")),
        };
        return Ok(());
    }
    if key == "reach" {
        cfg.eval.science.reach_model = match raw {
            "optimistic" | "0" => duels_eval::ReachModel::Optimistic,
            "structure" | "1" => duels_eval::ReachModel::Structure,
            other => return Err(format!("unknown reach model \"{other}\"")),
        };
        return Ok(());
    }
    if key == "wonder" {
        cfg.wonder_model = match raw {
            "flat" => duels_eval::WonderModel::Flat,
            "budget" => duels_eval::WonderModel::Budget,
            "rationed" => duels_eval::WonderModel::Rationed,
            other => return Err(format!("unknown wonder model \"{other}\"")),
        };
        return Ok(());
    }
    let v: f64 = raw
        .parse()
        .map_err(|_| format!("\"{raw}\" is not a number (key {key})"))?;
    let e = &mut cfg.eval;
    match key {
        "wonder_potential" => e.wonder_potential = v,
        "sci" => e.science_ladder = v,
        "dead" => e.science.dead_race_scale = v,
        "pairthreat" => e.science.pair_threat_weight = v,
        "pairshare" => e.science.pair_token_share = v,
        "pairtax" => e.science.pair_tempo_tax = v,
        "strongtok" => e.science.strong_token_mult = v,
        "ladder1" => e.science.ladder[1] = v,
        "ladder2" => e.science.ladder[2] = v,
        "ladder3" => e.science.ladder[3] = v,
        "ladder4" => e.science.ladder[4] = v,
        "ladder5" => e.science.ladder[5] = v,
        "yellow" => e.yellow_equity = v,
        "vp" => e.vp_projection = v,
        "band" => e.military_band = v,
        "scale" => e.value_scale = v,
        other => return Err(format!("unknown key \"{other}\"")),
    }
    Ok(())
}

fn parse_variant(arg: &str) -> Result<(String, Config), String> {
    let (label, params) = arg.split_once(':').unwrap_or((arg, ""));
    let mut cfg = Config::default();
    for kv in params.split(',').filter(|s| !s.trim().is_empty()) {
        let (k, v) = kv
            .split_once('=')
            .ok_or_else(|| format!("\"{kv}\" is not key=value"))?;
        apply(&mut cfg, k.trim(), v.trim())?;
    }
    Ok((label.to_string(), cfg))
}

fn main() {
    let mut games: u64 = 400;
    let mut seed: u64 = 0;
    let mut threads: usize = 8;
    let mut sides: Vec<String> = Vec::new();
    let mut read_arg: Option<String> = None;
    let mut factors = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--games" => games = args.next().and_then(|s| s.parse().ok()).unwrap_or(games),
            "--seed" => seed = args.next().and_then(|s| s.parse().ok()).unwrap_or(seed),
            "--threads" => threads = args.next().and_then(|s| s.parse().ok()).unwrap_or(threads),
            "--read" => read_arg = args.next(),
            "--factors" => factors = true,
            _ => sides.push(a),
        }
    }
    let (label_read, read) = match read_arg.as_deref() {
        None => ("default".to_string(), Config::default()),
        Some(s) => match parse_variant(s) {
            Ok(v) => v,
            Err(e) => return eprintln!("science_calibration: {e}"),
        },
    };
    let (label_a, cfg_a) = match sides.first() {
        None => ("default".to_string(), Config::default()),
        Some(s) => match parse_variant(s) {
            Ok(v) => v,
            Err(e) => return eprintln!("science_calibration: {e}"),
        },
    };
    let (label_b, cfg_b) = match sides.get(1) {
        None => ("default".to_string(), Config::default()),
        Some(s) => match parse_variant(s) {
            Ok(v) => v,
            Err(e) => return eprintln!("science_calibration: {e}"),
        },
    };

    let threads = threads.max(1);
    let mut samples: Vec<Sample> = Vec::new();
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let t = t as u64;
                let nthreads = threads as u64;
                scope.spawn(move || {
                    let mut out: Vec<Sample> = Vec::new();
                    let mut i = t;
                    while i < games {
                        let s = seed + i;
                        let a_seed = s ^ AGENT_A_SALT;
                        let b_seed = s ^ AGENT_B_SALT;
                        let a = (cfg_a, a_seed);
                        let b = (cfg_b, b_seed);
                        play_one(a, b, read, s, Player::One, &mut out);
                        play_one(b, a, read, s, Player::Two, &mut out);
                        i += nthreads;
                    }
                    out
                })
            })
            .collect();
        for h in handles {
            samples.extend(h.join().expect("a worker thread panicked"));
        }
    });

    println!(
        "science_calibration: seat A = {label_a}, seat B = {label_b}  \
         ({} games from base seed {seed}, both seat orders)",
        games * 2
    );
    println!(
        "{} player-position samples from decided games",
        samples.len()
    );
    println!(
        "the evaluation being read is {label_read}: {}",
        read.params_string()
    );

    table("every position, both seats:", &samples, |_| true);
    table("first position at that count, both seats:", &samples, |s| {
        s.first
    });
    if label_a != label_b {
        table(
            "first position at that count, seat A only:",
            &samples,
            |s| s.first && s.side_a,
        );
    }
    table(
        "first position at that count, the player to move:",
        &samples,
        |s| s.first && s.mover,
    );
    correction_table(&samples);
    if factors {
        factor_tables(&samples);
        correction_by_factor(
            &samples,
            "cards left",
            &[
                ("0-6 (late)", |s| s.cards_left <= 6),
                ("7-13 (mid)", |s| s.cards_left >= 7 && s.cards_left <= 13),
                ("14-20 (early)", |s| s.cards_left >= 14),
            ],
        );
        correction_by_factor(
            &samples,
            "reachable",
            &[
                ("6+ (race live)", |s| s.reach >= 6),
                ("<=5 (dead)", |s| s.reach <= 5),
            ],
        );
        correction_by_factor(
            &samples,
            "extra-turn diff",
            &[
                ("behind", |s| s.extra_turn_diff < 0),
                ("level", |s| s.extra_turn_diff == 0),
                ("ahead", |s| s.extra_turn_diff > 0),
            ],
        );
    }
}
