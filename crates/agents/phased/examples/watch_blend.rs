//! Play a game and narrate what the commitment blend is doing, turn by turn.
//!
//! This is the human checkpoint for this crate. There is no mode label to
//! sanity-check — the whole design is continuous — so the thing to read is
//! whether the *numbers* match how you read the position: is the player two
//! symbols from supremacy really only 0.18 committed? Should the points term
//! still be at 94% of its weight there? Is the development term putting its
//! value on the resources you would actually want?
//!
//! ```text
//! cargo run --release -p duels-agent-phased --example watch_blend
//! cargo run --release -p duels-agent-phased --example watch_blend -- 7 phased greedy-ev
//! cargo run --release -p duels-agent-phased --example watch_blend -- 7 --quiet
//! ```
//!
//! Arguments, all optional: `seed`, then the two agents (`phased`,
//! `greedy-ev` or `random`), and `--quiet` to print only the turns where some
//! weight has moved by more than a percentage point since the last printed
//! one.

use duels_agent_greedy_ev::GreedyEvAgent;
use duels_agent_phased::{terms, Config, PhasedAgent, Root};
use duels_agent_random::RandomAgent;
use duels_agents_api::{Agent, Budget};
use duels_core::data::Resource;
use duels_core::{engine, GameState, Player};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn make_agent(name: &str, seed: u64) -> Box<dyn Agent> {
    match name {
        "random" => Box::new(RandomAgent::new(seed)),
        "greedy-ev" => Box::new(GreedyEvAgent::new(seed)),
        "phased" => Box::new(PhasedAgent::new(seed)),
        other => {
            eprintln!("unknown agent {other:?}; using phased");
            Box::new(PhasedAgent::new(seed))
        }
    }
}

/// One printed block: both players' scalars, weights and term contents.
fn report(state: &GameState, root: &Root, config: &Config) {
    let e = &config.eval;
    println!(
        "  {:<4} {:>6} {:>6} {:>6} | {:>6} {:>6} | {:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5}",
        "",
        "c_sci",
        "c_mil",
        "c",
        "c0eff",
        "S(c)",
        "vp",
        "liq",
        "dev",
        "sci",
        "mil",
        "rliq",
        "econ"
    );
    for p in Player::ALL {
        let c = root.commitment(p);
        let w = root.weights(p);
        println!(
            "  {:<4} {:>6.3} {:>6.3} {:>6.3} | {:>6.3} {:>6.3} | {:>5.2} {:>5.2} {:>5.2} {:>5.2} {:>5.2} {:>5.2} {:>5.2}",
            if p == Player::One { "P1" } else { "P2" },
            c.c_sci,
            c.c_mil,
            c.c,
            c.c0_eff,
            c.s,
            w.vp,
            w.liquidity,
            w.development,
            w.science,
            w.military,
            w.race_liquidity,
            w.economy,
        );
    }

    // What the weighted terms are actually worth right now, so a weight can be
    // read against the number it is multiplying.
    for p in Player::ALL {
        let w = root.weights(p);
        let b = duels_core::scoring::breakdown(state, p);
        let dev = terms::development_by_resource(state, p, root.supply(), e.development_take_rate);
        let dev_total: f64 = dev.iter().sum();
        println!(
            "  {:<4} points {:>6.1}  coins {:>5.1}  dev {:>6.2} (w {:>4.1} c {:>4.1} s {:>4.1} g {:>4.1} p {:>4.1})  ladder {:>6.2}  mil {:>5.1}  start {:>4.1}",
            if p == Player::One { "P1" } else { "P2" },
            w.vp * e.vp_projection * terms::card_and_token_vp(&b),
            w.liquidity * e.coins_div3 * f64::from(b.coins),
            w.development * e.development * dev_total,
            dev[Resource::Wood.index()],
            dev[Resource::Clay.index()],
            dev[Resource::Stone.index()],
            dev[Resource::Glass.index()],
            dev[Resource::Papyrus.index()],
            w.science * e.science_ladder * terms::science_ladder(state, p, &e.science),
            w.military * e.military_band * terms::military_band(state, p, root.smoothing())
                + e.military_loot * terms::military_loot(state, p, root.smoothing()),
            terms::next_age_start(state, p, e),
        );

        // The resource bill, split the same way, so the claim that a second
        // grey source is valuable *because it raises the opponent's price*
        // can be checked rather than believed: read P2's row while P1 takes a
        // grey card and watch it climb.
        let bill =
            terms::resource_bill_by_resource(state, p, root.supply(), e.development_take_rate);
        let bill_total: f64 = bill.iter().sum();
        println!(
            "       bill {:>6.2} coins (w {:>4.1} c {:>4.1} s {:>4.1} g {:>4.1} p {:>4.1})   chain equity {:>5.2}   prices {:?}",
            bill_total,
            bill[Resource::Wood.index()],
            bill[Resource::Clay.index()],
            bill[Resource::Stone.index()],
            bill[Resource::Glass.index()],
            bill[Resource::Papyrus.index()],
            duels_agent_phased::menu::chain_equity(state, p, root.menu().chain()),
            duels_core::cost::trade_prices(state, p),
        );
    }
    println!(
        "  supply f_1: wood {:.2} clay {:.2} stone {:.2} glass {:.2} papyrus {:.2}  (pool {} cards)",
        root.supply().f[0][Resource::Wood.index()],
        root.supply().f[0][Resource::Clay.index()],
        root.supply().f[0][Resource::Stone.index()],
        root.supply().f[0][Resource::Glass.index()],
        root.supply().f[0][Resource::Papyrus.index()],
        root.supply().pool_size,
    );
    println!(
        "  next-age starter: {:?}   denial scale x{:.2}",
        terms::projected_starter(state),
        root.deny_scale(),
    );
}

/// How far apart two weight vectors are, for `--quiet`.
fn moved(a: &Root, b: &Root) -> f64 {
    let mut worst: f64 = 0.0;
    for p in Player::ALL {
        let (x, y) = (a.weights(p), b.weights(p));
        for (u, v) in [
            (x.vp, y.vp),
            (x.liquidity, y.liquidity),
            (x.development, y.development),
            (x.science, y.science),
            (x.military, y.military),
            (x.race_liquidity, y.race_liquidity),
            (x.economy, y.economy),
        ] {
            worst = worst.max((u - v).abs());
        }
    }
    worst
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let quiet = args.iter().any(|a| a == "--quiet");
    let positional: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    let seed: u64 = positional.first().and_then(|s| s.parse().ok()).unwrap_or(7);
    let names = [
        positional.get(1).map_or("phased", |s| s.as_str()),
        positional.get(2).map_or("phased", |s| s.as_str()),
    ];

    let config = Config::default();
    let mut agents = [
        make_agent(names[0], seed * 2),
        make_agent(names[1], seed * 2 + 1),
    ];
    let mut state = engine::new_game(seed);
    let mut rng = StdRng::seed_from_u64(seed ^ 0xB1E4D);

    println!("seed {seed}: {} (P1) vs {} (P2)", names[0], names[1]);
    println!("{}", config.params_string());
    println!();

    let mut last: Option<Root> = None;
    for _ in 0..600 {
        if state.is_over() {
            break;
        }
        let legal = engine::legal_actions(&state);
        if legal.is_empty() {
            break;
        }
        let me = state.current_player();
        let root = Root::new(&state, me, config);

        let interesting = match &last {
            None => true,
            Some(prev) => moved(prev, &root) > 0.01,
        };
        // Compare against the last block actually printed, so a blend that
        // drifts a fraction of a percent per turn still surfaces once the
        // drift adds up.
        if !quiet || interesting {
            println!(
                "turn {:>2}  age {}  pawn {:+}  to move {:?}  ({} legal)",
                state.turn(),
                state.age(),
                state.conflict(),
                me,
                legal.len()
            );
            report(&state, &root, &config);
        }

        let obs = state.observation();
        let action = agents[me.index()].choose(&obs, &legal, Budget::Nodes(1));
        if !quiet || interesting {
            println!("  -> {action:?}\n");
        }
        engine::apply_quiet(&mut state, action, &mut rng).expect("agents must play legal moves");
        if interesting {
            last = Some(root);
        }
    }

    println!("result: {:?}", state.result());
    for p in Player::ALL {
        let b = duels_core::scoring::breakdown(&state, p);
        println!("  {p:?}: {} points  {b:?}", b.total);
    }
}
