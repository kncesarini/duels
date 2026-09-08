//! One configuration of this crate against another, over a paired,
//! seat-swapped match — without a spec key for every knob.
//!
//! `duels-arena` is this project's measurement instrument and nothing here
//! replaces it. What it cannot do is reach a [`Config`] field that
//! `duels_arena::agent_spec::parse_phased_config` has no key for, and a round
//! that adds a knob adds it *here*, in a code-owner-review path, while the
//! parser lives in a crate a `duels-eval` round is not supposed to touch. Every
//! previous round therefore either got a new parser key in the same PR or could
//! not sweep its own knob at all.
//!
//! So this is the arena's `phased`-versus-`phased` match, reproduced inside
//! this crate over two `Config` values passed on the command line: the same
//! paired seat-swap, the same salts (`AGENT_A_SALT` / `AGENT_B_SALT` /
//! `ENGINE_RNG_SALT`), the same `PhasedAgent::choose`, and the same
//! Bradley-Terry fit with a half-game prior that `duels_arena::elo::fit_elo`
//! performs. `tests/head_to_head_agrees_with_the_arena` is not a thing this
//! crate can write — it would need the arena — so the agreement is checked by
//! hand instead, and the check is recorded in the round-seven section of the
//! crate docs: two runs of `phased:science_ladder=0.2` against `phased` at
//! 3200 games, one through each instrument, agreeing to the game.
//!
//! ```text
//! cargo run --release -p duels-eval --example head_to_head -- \
//!     --games 3200 --seed 1 "candidate:sci=0.2" "base:"
//! ```
//!
//! Both sides are `label:key=value,...` over [`Config::default`], with
//! `base=vN` available as the first key to start from a generation snapshot.
//! Elo is reported for the **first** side with the second as the anchor, which
//! is the arena's `--agent-a` / `--agent-b` convention.

use duels_core::scoring::GameResult;
use duels_core::{engine, Action, Observation, Player};
use duels_eval::{expected_value, Config, Root};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// `PhasedAgent`'s tie window.
const TIE_EPSILON: f64 = 1e-6;
/// `duels_arena::match_runner`'s per-role agent salts, verbatim.
const AGENT_A_SALT: u64 = 0xA011_7A9E_5B21_0001;
const AGENT_B_SALT: u64 = 0xB022_8C3F_6D42_0002;
/// ...and its engine-stream salt.
const ENGINE_RNG_SALT: u64 = 0x9E37_79B9_7F4A_7C15;
/// `duels_arena::elo`'s Elo-to-logit constant and 95% z.
const ELO_TO_LOGIT: f64 = std::f64::consts::LN_10 / 400.0;
const Z_95: f64 = 1.959_963_984_540_054;

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

/// Play one game with `seat_one` moving first, and report the result plus the
/// victory kind.
fn play_one(
    seat_one: (Config, u64),
    seat_two: (Config, u64),
    setup_seed: u64,
) -> Option<GameResult> {
    let mut driver = [
        Driver::new(seat_one.1, seat_one.0),
        Driver::new(seat_two.1, seat_two.0),
    ];
    let mut state = engine::new_game(setup_seed);
    let mut rng = StdRng::seed_from_u64(setup_seed ^ ENGINE_RNG_SALT);
    while !state.is_over() {
        let legal = engine::legal_actions(&state);
        if legal.is_empty() {
            break;
        }
        let me = state.current_player();
        let obs: Observation = state.observation();
        let action = driver[me.index()].choose(&obs, &legal);
        engine::apply(&mut state, action, &mut rng).expect("the driver plays legally");
    }
    state.result()
}

/// Wins for A, wins for B, draws — and A's wins split by victory kind.
#[derive(Default, Clone, Copy)]
struct Tally {
    a: u32,
    b: u32,
    draws: u32,
    a_military: u32,
    a_science: u32,
    a_civilian: u32,
    a_tiebreak: u32,
    b_military: u32,
    b_science: u32,
}

impl Tally {
    fn merge(&mut self, other: &Tally) {
        self.a += other.a;
        self.b += other.b;
        self.draws += other.draws;
        self.a_military += other.a_military;
        self.a_science += other.a_science;
        self.a_civilian += other.a_civilian;
        self.a_tiebreak += other.a_tiebreak;
        self.b_military += other.b_military;
        self.b_science += other.b_science;
    }

    fn record(&mut self, result: Option<GameResult>, a_seat: Player) {
        use duels_core::scoring::VictoryKind;
        match result {
            Some(GameResult::Win { winner, kind }) if winner == a_seat => {
                self.a += 1;
                match kind {
                    VictoryKind::MilitarySupremacy => self.a_military += 1,
                    VictoryKind::ScientificSupremacy => self.a_science += 1,
                    VictoryKind::CivilianVictory => self.a_civilian += 1,
                    VictoryKind::CivilianTiebreak => self.a_tiebreak += 1,
                }
            }
            Some(GameResult::Win { kind, .. }) => {
                self.b += 1;
                match kind {
                    VictoryKind::MilitarySupremacy => self.b_military += 1,
                    VictoryKind::ScientificSupremacy => self.b_science += 1,
                    _ => {}
                }
            }
            _ => self.draws += 1,
        }
    }
}

/// `duels_arena::elo::fit_elo`, reproduced: Bradley-Terry by Newton's method
/// with one half-game at 50/50 as a weak prior, and a 95% interval from the
/// Fisher information.
fn fit_elo(wins: u32, losses: u32, draws: u32) -> (f64, f64) {
    let w = f64::from(wins) + f64::from(draws) * 0.5 + 0.5;
    let l = f64::from(losses) + f64::from(draws) * 0.5 + 0.5;
    let n = w + l;
    let sigmoid = |x: f64| 1.0 / (1.0 + (-x).exp());
    let mut d = 0.0f64;
    for _ in 0..50 {
        let p = sigmoid(ELO_TO_LOGIT * d);
        let grad = ELO_TO_LOGIT * (w - n * p);
        let hess = -ELO_TO_LOGIT * ELO_TO_LOGIT * n * p * (1.0 - p);
        if hess.abs() < 1e-12 {
            break;
        }
        let step = grad / hess;
        d -= step;
        if step.abs() < 1e-9 {
            break;
        }
    }
    let p = sigmoid(ELO_TO_LOGIT * d);
    let information = ELO_TO_LOGIT * ELO_TO_LOGIT * n * p * (1.0 - p);
    let se = if information > 0.0 {
        1.0 / information.sqrt()
    } else {
        f64::INFINITY
    };
    (d, Z_95 * se)
}

/// Apply one `key=value` override. Kept deliberately in step with
/// `examples/leaf_probe.rs`'s table so a knob screened on one instrument can be
/// screened on the other by the same string.
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
    if key == "count" {
        cfg.count_pricing = if raw == "0" || raw == "off" || raw == "unpriced" {
            duels_eval::CountPricing::Unpriced
        } else {
            duels_eval::CountPricing::Counted
        };
        return Ok(());
    }
    let v: f64 = raw
        .parse()
        .map_err(|_| format!("\"{raw}\" is not a number (key {key})"))?;
    let e = &mut cfg.eval;
    match key {
        "yellow" => e.yellow_equity = v,
        "discardrate" => e.yellow_discard_rate = v,
        "guildproj" => e.guild_projection = v,
        "band" => e.military_band = v,
        "loot" => e.military_loot = v,
        "urgency" => e.military_endgame_urgency = v,
        "lambda" => e.menu.lambda = v,
        "tau" => e.menu.tau = v,
        "coins_div3" => e.coins_div3 = v,
        "vp" => e.vp_projection = v,
        "dev" => e.development = v,
        "sci" => e.science_ladder = v,
        "bill" => e.resource_bill = v,
        "chaineq" => e.chain_equity = v,
        "wprem" => e.wonder_extra_turn_premium = v,
        "wonder_potential" => e.wonder_potential = v,
        "deny" => e.deny = v,
        "start1" => e.next_age_start[0] = v,
        "start2" => e.next_age_start[1] = v,
        "lockin" => e.production_lock_in = v,
        "beta" => e.coin_smooth_beta = v,
        "cref" => e.coin_smooth_ref = v,
        "raceliq" => e.race_card_liquidity = v,
        "denyboost" => e.deny_opponent_commit_boost = v,
        "kappa" => e.military_sigma_scale = v,
        "logistic" => e.military_logistic_scale = v,
        "start3" => e.next_age_start[2] = v,
        "takerate" => e.development_take_rate = v,
        "coinend" => e.coin_endgame_decisions = v,
        "strongtok" => e.science.strong_token_mult = v,
        "pairshare" => e.science.pair_token_share = v,
        "pairtax" => e.science.pair_tempo_tax = v,
        "dead" => e.science.dead_race_scale = v,
        "tokeneq" => e.token_equity = v,
        "tomove" => e.to_move = v,
        "scale" => e.value_scale = v,
        "pairthreat" => e.science.pair_threat_weight = v,
        "ladder1" => e.science.ladder[1] = v,
        "ladder2" => e.science.ladder[2] = v,
        "ladder3" => e.science.ladder[3] = v,
        "ladder4" => e.science.ladder[4] = v,
        "ladder5" => e.science.ladder[5] = v,
        "turns" => e.wonder_turns_per_wonder = v,
        "pref" => e.wonder_p_build_ref = v,
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
    let mut games: u64 = 3200;
    let mut seed: u64 = 1;
    let mut threads: usize = 8;
    let mut sides: Vec<String> = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--games" => games = args.next().and_then(|s| s.parse().ok()).unwrap_or(games),
            "--seed" => seed = args.next().and_then(|s| s.parse().ok()).unwrap_or(seed),
            "--threads" => threads = args.next().and_then(|s| s.parse().ok()).unwrap_or(threads),
            _ => sides.push(a),
        }
    }
    if sides.len() != 2 {
        eprintln!("head_to_head: expected exactly two \"label:key=value,...\" sides");
        return;
    }
    let (label_a, cfg_a) = match parse_variant(&sides[0]) {
        Ok(v) => v,
        Err(e) => return eprintln!("head_to_head: {e}"),
    };
    let (label_b, cfg_b) = match parse_variant(&sides[1]) {
        Ok(v) => v,
        Err(e) => return eprintln!("head_to_head: {e}"),
    };

    // `ceil(games / 2)` paired seeds, exactly as `duels-arena match` does.
    let pairs = games.div_ceil(2);
    let threads = threads.max(1);
    let mut total = Tally::default();
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let t = t as u64;
                let threads = threads as u64;
                scope.spawn(move || {
                    let mut tally = Tally::default();
                    let mut i = t;
                    while i < pairs {
                        let s = seed + i;
                        let a_seed = s ^ AGENT_A_SALT;
                        let b_seed = s ^ AGENT_B_SALT;
                        tally.record(play_one((cfg_a, a_seed), (cfg_b, b_seed), s), Player::One);
                        tally.record(play_one((cfg_b, b_seed), (cfg_a, a_seed), s), Player::Two);
                        i += threads;
                    }
                    tally
                })
            })
            .collect();
        for h in handles {
            total.merge(&h.join().expect("a worker thread panicked"));
        }
    });

    let (elo, ci) = fit_elo(total.a, total.b, total.draws);
    println!(
        "head_to_head: {label_a} vs {label_b}  ({} games = {pairs} paired seeds, base seed {seed})",
        pairs * 2
    );
    println!(
        "results: {label_a} {} wins, {label_b} {} wins, {} draws",
        total.a, total.b, total.draws
    );
    println!(
        "victory kinds: {label_a} (military {}, scientific {}, civilian {}, tiebreak {})   \
         {label_b} (military {}, scientific {})",
        total.a_military,
        total.a_science,
        total.a_civilian,
        total.a_tiebreak,
        total.b_military,
        total.b_science
    );
    println!(
        "elo: {label_a} = {elo:+.1} (anchor: {label_b} = 0.0), 95% CI [{:+.1}, {:+.1}]",
        elo - ci,
        elo + ci
    );
}
